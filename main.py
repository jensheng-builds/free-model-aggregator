# free-model-aggregator-02 entry: a live free-model aggregator server.
#
# One process serves both client wires (OpenAI Chat Completions and Anthropic
# Messages, JSON and SSE), and every completion it answers is a real call to a
# free OpenRouter model dialled through the grader's gateway. The server's own
# job is routing: pick the highest-rated free model that meets each request's
# requirements, fail over down that ranked list when a provider refuses, cool
# down models that keep refusing, and report every refusal it recovered from
# on the x-makefun-upstream-failures response header.
#
# Trial lifecycle (agent/v1):
#   * read the brief at $MAKEFUN_PROMPT_FILE (free model list + prompt manifest)
#   * listen on 0.0.0.0 at an OS-picked free port (never 8080, the env's port)
#   * POST $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": p, "catalog": c}
#   * serve the three routes until GET $MAKEFUN_ENV_URL/work says done, exit 0
#   * `python3 main.py --list` prints the catalog document and exits 0; the
#     registered catalog is that same document, byte for byte.
#
# Determinism: every choice below is a function of the brief, the requests and
# the gateway answers seen so far. No wall clock, no randomness: a replayed
# trial must leave a byte-identical state export. Listed models therefore
# carry fixed `created` values, and model cooldowns count refusals instead of
# reading the time.
#
# Standard library only: a trial runs `python3 -P -s /bot/main.py`, so this
# file imports nothing outside Python 3.13's stdlib and keeps every helper in
# this one file (with -P the script's directory is not on sys.path).

import json
import math
import os
import signal
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

# --------------------------------------------------------------------------
# Constants and environment
# --------------------------------------------------------------------------

PROTOCOL_VERSION = 1
POLL_INTERVAL_SEC = 0.1
MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_OUTPUT_TOKENS = 8192  # the gateway refuses an output cap above this
DEFAULT_OUTPUT_TOKENS = 1024  # the gateway refuses a request with no cap
MAX_ATTEMPTS_PER_PROMPT = 6  # failover depth; the trial caps calls at 60
MAX_FAILURE_ROWS = 8  # the failures header carries at most 8 rows
STRIKE_PENALTY_MILLI = 150  # rating points a model loses per recorded refusal

ENV_URL = os.environ.get('MAKEFUN_ENV_URL', '').rstrip('/')
PROMPT_FILE = os.environ.get('MAKEFUN_PROMPT_FILE', '')
GATEWAY_URL = os.environ.get('OPENAI_BASE_URL', '').rstrip('/')
GATEWAY_KEY = os.environ.get('OPENAI_API_KEY', '')

SCHEMA_PREFIX = 'free-model-aggregator-02/'
MODELS_SCHEMA = 'free-model-aggregator-02/models/v1'
PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1'

# The catalog names the provider's public endpoint, never the gateway's
# trial-local address (which differs between a live pass and its replay).
PROVIDER_ID = 'openrouter'
PROVIDER_BASE_URL = 'https://openrouter.ai/api/v1'
PROVIDER_KEY_ENV = 'OPENROUTER_API_KEY'
PROVIDER_WIRE = 'openai'

FIXED_OPENAI_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z'

Json = dict[str, object]

# --------------------------------------------------------------------------
# Small JSON value helpers
# --------------------------------------------------------------------------


def is_record(value: object) -> bool:
    return isinstance(value, dict)


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def as_str(value: object) -> str:
    return value if isinstance(value, str) else ''


def as_number_or_none(value: object) -> int | float | None:
    return value if is_number(value) and math.isfinite(value) else None


def as_number_or_zero(value: object) -> int | float:
    return value if is_number(value) else 0


def as_bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def clamp_output_tokens(value: object) -> int:
    """An output cap the gateway accepts: present, positive, at most 8192."""
    number = as_number_or_none(value)
    if number is None or number <= 0:
        return DEFAULT_OUTPUT_TOKENS
    return int(min(number, MAX_OUTPUT_TOKENS))


# --------------------------------------------------------------------------
# The brief: two JSON documents embedded in the prompt file's prose
# --------------------------------------------------------------------------


def _object_end(text: str, start: int) -> int:
    """Index of the brace closing the object that opens at `start`, or -1."""
    depth = 0
    in_string = False
    i = start
    while i < len(text):
        ch = text[i]
        if in_string:
            if ch == '\\':
                i += 2
                continue
            if ch == '"':
                in_string = False
        elif ch == '"':
            in_string = True
        elif ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return -1


def _schema_documents(text: str) -> list[Json]:
    """Every JSON object in `text` carrying a string `schema`, in order."""
    docs: list[Json] = []
    start = text.find('{')
    while start != -1:
        end = _object_end(text, start)
        if end != -1:
            try:
                value = json.loads(text[start : end + 1])
            except ValueError:
                value = None
            if is_record(value) and isinstance(value.get('schema'), str):
                docs.append(value)
        start = text.find('{', start + 1)
    return docs


def _find_document(docs: list[Json], keys: tuple[str, ...]) -> Json | None:
    for doc in docs:
        if SCHEMA_PREFIX not in as_str(doc.get('schema')):
            continue
        if any(isinstance(doc.get(key), list) for key in keys):
            return doc
    return None


def _model_row(value: object) -> Json | None:
    """One free-model row from the brief, normalised to the gateway's id."""
    if not is_record(value):
        return None
    raw_id = as_str(value.get('id')) or as_str(value.get('model'))
    if raw_id == '':
        return None
    provider = as_str(value.get('provider')) or PROVIDER_ID
    model_id = raw_id
    if provider == PROVIDER_ID and not raw_id.lower().startswith('openrouter/'):
        model_id = f'openrouter/{raw_id}'
    rating = as_number_or_none(value.get('rating_milli'))
    return {
        'id': model_id,
        'provider': provider,
        'display_name': as_str(value.get('display_name')) or raw_id,
        'rating_milli': rating if rating is not None else 0,
        'context_length': as_number_or_none(value.get('context_length')),
        'tools': as_bool_or_none(value.get('tools')),
        'vision': as_bool_or_none(value.get('vision')),
        'json': as_bool_or_none(value.get('json')),
        'free': value.get('free') is not False,
    }


def _prompt_row(value: object) -> Json | None:
    if not is_record(value) or as_str(value.get('id')) == '':
        return None
    requires = value.get('requires') if is_record(value.get('requires')) else {}
    return {'id': as_str(value.get('id')), 'requires': requires}


def read_brief(path: str) -> Json:
    """The brief's model list and prompt manifest; empty lists without a file."""
    if path == '':
        return {'models': [], 'prompts': [], 'error': None}
    try:
        with open(path, encoding='utf-8') as handle:
            text = handle.read()
    except OSError as err:
        return {'models': [], 'prompts': [], 'error': f'cannot read the brief at {path}: {err}'}
    docs = _schema_documents(text)
    models_doc = _find_document(docs, ('models', 'ratings'))
    prompts_doc = _find_document(docs, ('prompts', 'manifest'))
    models: list[Json] = []
    if models_doc is not None:
        rows = models_doc.get('models') if isinstance(models_doc.get('models'), list) else models_doc.get('ratings')
        for row in rows if isinstance(rows, list) else []:
            model = _model_row(row)
            if model is not None:
                models.append(model)
    prompts: list[Json] = []
    if prompts_doc is not None:
        rows = prompts_doc.get('prompts') if isinstance(prompts_doc.get('prompts'), list) else prompts_doc.get('manifest')
        for row in rows if isinstance(rows, list) else []:
            prompt = _prompt_row(row)
            if prompt is not None:
                prompts.append(prompt)
    return {'models': models, 'prompts': prompts, 'error': None}


BRIEF = read_brief(PROMPT_FILE)


def free_models() -> list[Json]:
    return [m for m in BRIEF['models'] if m['free']]


# --------------------------------------------------------------------------
# The catalog document (--list output and the register body, one object)
# --------------------------------------------------------------------------


def build_catalog() -> Json:
    seen: set[str] = set()
    models: list[Json] = []
    for model in BRIEF['models']:
        if not model['free'] or model['id'] in seen:
            continue
        seen.add(model['id'])
        models.append({'id': model['id'], 'free': True, 'context_length': model['context_length']})
    if not models:
        return {'providers': []}
    return {
        'providers': [
            {
                'id': PROVIDER_ID,
                'base_url': PROVIDER_BASE_URL,
                'key_env': PROVIDER_KEY_ENV,
                'wire': PROVIDER_WIRE,
                'models': models,
            }
        ]
    }


CATALOG = build_catalog()
CATALOG_JSON = json.dumps(CATALOG, separators=(',', ':'), ensure_ascii=False)


# --------------------------------------------------------------------------
# Routing: ranked candidates, requirement checks, refusal cooldowns
# --------------------------------------------------------------------------

# Refusals seen so far this trial, per model id. A count, never a timestamp:
# the replay must make the same choices from the same recorded answers.
_strikes: dict[str, int] = {}
_strikes_lock = threading.Lock()

# Completion requests answered so far, used to line a request up with the
# brief's prompt manifest (the env sends the prompts in manifest order).
_manifest_index = 0
_manifest_lock = threading.Lock()


def record_refusal(model_id: str) -> None:
    with _strikes_lock:
        _strikes[model_id] = _strikes.get(model_id, 0) + 1


def strikes_of(model_id: str) -> int:
    with _strikes_lock:
        return _strikes.get(model_id, 0)


def effective_rating(model: Json) -> int | float:
    """Published rating minus a penalty per refusal this trial has seen."""
    return model['rating_milli'] - STRIKE_PENALTY_MILLI * strikes_of(model['id'])


def meets_requirements(requires: Json, model: Json) -> bool:
    """Whether `model` can serve a prompt with these requirements.

    An unknown capability (None in the brief) never meets a requirement: a
    JSON-mode prompt dialled to a model without JSON support is a wasted call.
    """
    if requires.get('tools') is True and model['tools'] is not True:
        return False
    if requires.get('vision') is True and model['vision'] is not True:
        return False
    if requires.get('json') is True and model['json'] is not True:
        return False
    min_context = as_number_or_none(requires.get('min_context'))
    if min_context is not None:
        context = model['context_length']
        if context is None or context < min_context:
            return False
    return True


def candidates_for(requires: Json) -> list[Json]:
    """Free models meeting `requires`, best first.

    Order is the effective rating (published rating less refusal penalties),
    descending; ties break on the model id so the order never depends on the
    brief's layout or on set iteration.
    """
    ranked = [m for m in free_models() if meets_requirements(requires, m)]
    ranked.sort(key=lambda m: (-effective_rating(m), m['id']))
    return ranked


def requirements_from_body(body: Json) -> Json:
    """Requirements read off one request body itself."""
    vision = False
    messages = body.get('messages') if isinstance(body.get('messages'), list) else []
    for message in messages:
        if not is_record(message) or not isinstance(message.get('content'), list):
            continue
        for block in message['content']:
            if not is_record(block):
                continue
            if block.get('type') in ('image', 'image_url', 'input_image'):
                vision = True
            source = block.get('source')
            if block.get('type') == 'image' and is_record(source):
                vision = True
    response_format = body.get('response_format')
    response_type = response_format.get('type') if is_record(response_format) else None
    return {
        'tools': isinstance(body.get('tools'), list) and len(body['tools']) > 0,
        'vision': vision,
        'json': response_type in ('json_object', 'json_schema'),
    }


def requirements_for(body: Json) -> Json:
    """The requirements of the request now arriving.

    The union of the brief manifest's row for this arrival (the env sends the
    prompts in manifest order) and what the body itself declares: a model
    that meets the union meets either source on its own.
    """
    global _manifest_index
    with _manifest_lock:
        index = _manifest_index
        _manifest_index += 1
    merged = dict(requirements_from_body(body))
    prompts = BRIEF['prompts']
    if index < len(prompts):
        for key, value in prompts[index]['requires'].items():
            if value is True:
                merged[key] = True
            elif key == 'min_context' and value is not None:
                merged[key] = value
    return merged


# --------------------------------------------------------------------------
# Wire shapes: model lists and error envelopes
# --------------------------------------------------------------------------


def listed_models() -> list[Json]:
    return [
        {'id': m['id'], 'provider': m['provider'], 'display_name': m['display_name']}
        for m in free_models()
    ]


def openai_model_list(models: list[Json]) -> Json:
    return {
        'object': 'list',
        'data': [
            {'id': m['id'], 'object': 'model', 'created': FIXED_OPENAI_CREATED, 'owned_by': m['provider']}
            for m in models
        ],
    }


def anthropic_model_list(models: list[Json]) -> Json:
    data = [
        {
            'type': 'model',
            'id': m['id'],
            'display_name': m['display_name'],
            'created_at': FIXED_ANTHROPIC_CREATED_AT,
        }
        for m in models
    ]
    return {
        'data': data,
        'has_more': False,
        'first_id': data[0]['id'] if data else None,
        'last_id': data[-1]['id'] if data else None,
    }


def openai_error(message: str, error_type: str, code: str | None) -> Json:
    return {'error': {'message': message, 'type': error_type, 'code': code}}


def anthropic_error(error_type: str, message: str) -> Json:
    return {'type': 'error', 'error': {'type': error_type, 'message': message}}


# --------------------------------------------------------------------------
# Anthropic <-> OpenAI translation
# --------------------------------------------------------------------------


def text_of(content: object) -> str:
    """A string content, or the joined text blocks of a block array."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ''
    return '\n'.join(
        b['text'] for b in content if is_record(b) and b.get('type') == 'text' and isinstance(b.get('text'), str)
    )


def _image_part(block: Json) -> Json | None:
    source = block.get('source')
    if not is_record(source):
        return None
    if source.get('type') == 'base64' and isinstance(source.get('data'), str):
        media = source['media_type'] if isinstance(source.get('media_type'), str) else 'image/png'
        return {'type': 'image_url', 'image_url': {'url': f'data:{media};base64,{source["data"]}'}}
    if source.get('type') == 'url' and isinstance(source.get('url'), str):
        return {'type': 'image_url', 'image_url': {'url': source['url']}}
    return None


def _tool_result_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text = text_of(content)
        return text if text != '' else json.dumps(content, separators=(',', ':'), ensure_ascii=False)
    if content is None:
        return ''
    return json.dumps(content, separators=(',', ':'), ensure_ascii=False)


def anthropic_to_openai(body: Json, model: str) -> Json:
    """An Anthropic Messages request rewritten as a Chat Completions request."""
    messages: list[Json] = []
    system = text_of(body.get('system'))
    if system != '':
        messages.append({'role': 'system', 'content': system})
    source = body.get('messages') if isinstance(body.get('messages'), list) else []
    for raw in source:
        if not is_record(raw):
            continue
        is_assistant = raw.get('role') == 'assistant'
        content = raw.get('content')
        if isinstance(content, str):
            messages.append({'role': 'assistant' if is_assistant else 'user', 'content': content})
            continue
        if not isinstance(content, list):
            continue
        if is_assistant:
            tool_calls: list[Json] = []
            for block in content:
                if not is_record(block) or block.get('type') != 'tool_use':
                    continue
                tool_calls.append(
                    {
                        'id': block['id'] if isinstance(block.get('id'), str) else '',
                        'type': 'function',
                        'function': {
                            'name': block['name'] if isinstance(block.get('name'), str) else '',
                            'arguments': json.dumps(
                                block['input'] if is_record(block.get('input')) else {},
                                separators=(',', ':'),
                                ensure_ascii=False,
                            ),
                        },
                    }
                )
            text = text_of(content)
            message: Json = {'role': 'assistant', 'content': text if text != '' else None}
            if tool_calls:
                message['tool_calls'] = tool_calls
            messages.append(message)
            continue
        parts: list[Json] = []
        for block in content:
            if not is_record(block):
                continue
            if block.get('type') == 'tool_result':
                messages.append(
                    {
                        'role': 'tool',
                        'tool_call_id': block['tool_use_id'] if isinstance(block.get('tool_use_id'), str) else '',
                        'content': _tool_result_text(block.get('content')),
                    }
                )
                continue
            if block.get('type') == 'text' and isinstance(block.get('text'), str):
                parts.append({'type': 'text', 'text': block['text']})
                continue
            if block.get('type') == 'image':
                part = _image_part(block)
                if part is not None:
                    parts.append(part)
        if parts:
            messages.append({'role': 'user', 'content': parts})

    out: Json = {'model': model, 'messages': messages, 'max_tokens': clamp_output_tokens(body.get('max_tokens'))}
    if is_number(body.get('temperature')):
        out['temperature'] = body['temperature']
    if is_number(body.get('top_p')):
        out['top_p'] = body['top_p']
    if isinstance(body.get('stop_sequences'), list) and len(body['stop_sequences']) > 0:
        out['stop'] = body['stop_sequences']
    if isinstance(body.get('tools'), list):
        tools: list[Json] = []
        for raw_tool in body['tools']:
            if not is_record(raw_tool) or not isinstance(raw_tool.get('name'), str):
                continue
            tools.append(
                {
                    'type': 'function',
                    'function': {
                        'name': raw_tool['name'],
                        'description': raw_tool['description'] if isinstance(raw_tool.get('description'), str) else '',
                        'parameters': raw_tool['input_schema']
                        if is_record(raw_tool.get('input_schema'))
                        else {'type': 'object', 'properties': {}},
                    },
                }
            )
        if tools:
            out['tools'] = tools
    choice = body.get('tool_choice')
    if is_record(choice):
        if choice.get('type') == 'auto':
            out['tool_choice'] = 'auto'
        elif choice.get('type') == 'any':
            out['tool_choice'] = 'required'
        elif choice.get('type') == 'tool' and isinstance(choice.get('name'), str):
            out['tool_choice'] = {'type': 'function', 'function': {'name': choice['name']}}
        elif choice.get('type') == 'none':
            out['tool_choice'] = 'none'
    if body.get('stream') is True:
        out['stream'] = True
        out['stream_options'] = {'include_usage': True}
    return out


def stop_reason_of(finish: object) -> str:
    if finish == 'length':
        return 'max_tokens'
    if finish in ('tool_calls', 'function_call'):
        return 'tool_use'
    if finish == 'content_filter':
        return 'refusal'
    return 'end_turn'


def _arguments_object(value: object) -> Json:
    if is_record(value):
        return value
    if isinstance(value, str) and value.strip() != '':
        try:
            parsed = json.loads(value)
        except ValueError:
            return {}
        if is_record(parsed):
            return parsed
    return {}


def openai_to_anthropic(body: Json, model: str) -> Json:
    """A Chat Completions answer rewritten as an Anthropic Messages answer."""
    choices = body.get('choices') if isinstance(body.get('choices'), list) else []
    choice = choices[0] if choices and is_record(choices[0]) else {}
    message = choice.get('message') if is_record(choice.get('message')) else {}
    content: list[Json] = []
    raw_content = message.get('content')
    text = raw_content if isinstance(raw_content, str) else text_of(raw_content)
    if text != '':
        content.append({'type': 'text', 'text': text})
    calls = message.get('tool_calls') if isinstance(message.get('tool_calls'), list) else []
    for call in calls:
        if not is_record(call) or not is_record(call.get('function')):
            continue
        fn = call['function']
        content.append(
            {
                'type': 'tool_use',
                'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else 'toolu_unknown',
                'name': fn['name'] if isinstance(fn.get('name'), str) else '',
                'input': _arguments_object(fn.get('arguments')),
            }
        )
    if not content:
        content.append({'type': 'text', 'text': ''})
    usage = body.get('usage') if is_record(body.get('usage')) else {}
    upstream_id = body['id'] if isinstance(body.get('id'), str) and body['id'] != '' else 'aggregate'
    return {
        'id': f'msg_{upstream_id}',
        'type': 'message',
        'role': 'assistant',
        'model': model,
        'content': content,
        'stop_reason': stop_reason_of(choice.get('finish_reason')),
        'stop_sequence': None,
        'usage': {
            'input_tokens': as_number_or_zero(usage.get('prompt_tokens')),
            'output_tokens': as_number_or_zero(usage.get('completion_tokens')),
        },
    }


# --------------------------------------------------------------------------
# Server-sent events: OpenAI passthrough and Anthropic translation
# --------------------------------------------------------------------------


def sse_event(event: str | None, data: object) -> str:
    prefix = '' if event is None else f'event: {event}\n'
    return f'{prefix}data: {json.dumps(data, separators=(",", ":"), ensure_ascii=False)}\n\n'


def iter_sse_frames(chunks):
    """SSE frames (text, without the trailing blank line) from byte chunks."""
    buffer = ''
    for chunk in chunks:
        buffer += chunk.decode('utf-8', 'replace').replace('\r\n', '\n')
        while '\n\n' in buffer:
            frame, buffer = buffer.split('\n\n', 1)
            yield frame
    if buffer.strip() != '':
        yield buffer


def frame_data(frame: str) -> str | None:
    """The joined `data:` payload of one SSE frame, or None."""
    lines = [line[5:].lstrip() for line in frame.split('\n') if line.startswith('data:')]
    return '\n'.join(lines) if lines else None


def openai_stream_frames(chunks, model: str):
    """Gateway SSE frames re-emitted on the OpenAI wire with `model` fixed.

    Every JSON chunk's model field is set to the id actually dialled (the
    provider spells it without the gateway's prefix), and the stream always
    ends with the [DONE] frame the wire expects.
    """
    saw_done = False
    for frame in iter_sse_frames(chunks):
        payload = frame_data(frame)
        if payload is None:
            yield frame + '\n\n'
            continue
        if payload.strip() == '[DONE]':
            saw_done = True
            yield 'data: [DONE]\n\n'
            continue
        try:
            value = json.loads(payload)
        except ValueError:
            yield frame + '\n\n'
            continue
        if is_record(value):
            value = {**value, 'model': model}
            yield sse_event(None, value)
        else:
            yield frame + '\n\n'
    if not saw_done:
        yield 'data: [DONE]\n\n'


def anthropic_stream_events(chunks, model: str):
    """Gateway SSE frames translated to the Anthropic Messages event stream.

    Event order follows the Messages API: message_start, then per content
    block content_block_start / content_block_delta / content_block_stop,
    then message_delta (stop reason and output usage) and message_stop.
    Text and tool-call argument deltas are forwarded as they arrive; the
    stop reason and usage, which OpenAI sends last, close the stream.
    """
    started = False
    finished = False
    next_index = 0
    text_index: int | None = None
    tool_indexes: dict[int, int] = {}
    open_blocks: list[int] = []
    pending_stop = 'end_turn'
    usage_out: int | float = 0
    usage_in: int | float = 0
    message_id = 'msg_aggregate'

    def start_events() -> list[str]:
        return [
            sse_event(
                'message_start',
                {
                    'type': 'message_start',
                    'message': {
                        'id': message_id,
                        'type': 'message',
                        'role': 'assistant',
                        'model': model,
                        'content': [],
                        'stop_reason': None,
                        'stop_sequence': None,
                        'usage': {'input_tokens': usage_in, 'output_tokens': 0},
                    },
                },
            )
        ]

    def finish_events() -> list[str]:
        events: list[str] = []
        for index in open_blocks:
            events.append(sse_event('content_block_stop', {'type': 'content_block_stop', 'index': index}))
        open_blocks.clear()
        events.append(
            sse_event(
                'message_delta',
                {
                    'type': 'message_delta',
                    'delta': {'stop_reason': pending_stop, 'stop_sequence': None},
                    'usage': {'output_tokens': usage_out},
                },
            )
        )
        events.append(sse_event('message_stop', {'type': 'message_stop'}))
        return events

    for frame in iter_sse_frames(chunks):
        payload = frame_data(frame)
        if payload is None or payload.strip() == '[DONE]':
            continue
        try:
            value = json.loads(payload)
        except ValueError:
            continue
        if not is_record(value):
            continue
        if isinstance(value.get('id'), str) and value['id'] != '':
            message_id = f'msg_{value["id"]}'
        usage = value.get('usage')
        if is_record(usage):
            if is_number(usage.get('completion_tokens')):
                usage_out = usage['completion_tokens']
            if is_number(usage.get('prompt_tokens')):
                usage_in = usage['prompt_tokens']
        choices = value.get('choices') if isinstance(value.get('choices'), list) else []
        if not choices or not is_record(choices[0]):
            continue
        choice = choices[0]
        delta = choice.get('delta') if is_record(choice.get('delta')) else {}
        events: list[str] = []
        if not started:
            started = True
            events.extend(start_events())
        text = delta.get('content')
        if isinstance(text, str) and text != '':
            if text_index is None:
                text_index = next_index
                next_index += 1
                open_blocks.append(text_index)
                events.append(
                    sse_event(
                        'content_block_start',
                        {'type': 'content_block_start', 'index': text_index, 'content_block': {'type': 'text', 'text': ''}},
                    )
                )
            events.append(
                sse_event(
                    'content_block_delta',
                    {'type': 'content_block_delta', 'index': text_index, 'delta': {'type': 'text_delta', 'text': text}},
                )
            )
        calls = delta.get('tool_calls') if isinstance(delta.get('tool_calls'), list) else []
        for call in calls:
            if not is_record(call):
                continue
            upstream_index = call['index'] if is_number(call.get('index')) else len(tool_indexes)
            index = tool_indexes.get(upstream_index)
            fn = call.get('function') if is_record(call.get('function')) else {}
            if index is None:
                index = next_index
                next_index += 1
                tool_indexes[upstream_index] = index
                open_blocks.append(index)
                events.append(
                    sse_event(
                        'content_block_start',
                        {
                            'type': 'content_block_start',
                            'index': index,
                            'content_block': {
                                'type': 'tool_use',
                                'id': call['id'] if isinstance(call.get('id'), str) and call['id'] != '' else f'toolu_{index}',
                                'name': fn.get('name') if isinstance(fn.get('name'), str) else '',
                                'input': {},
                            },
                        },
                    )
                )
            if isinstance(fn.get('arguments'), str) and fn['arguments'] != '':
                events.append(
                    sse_event(
                        'content_block_delta',
                        {
                            'type': 'content_block_delta',
                            'index': index,
                            'delta': {'type': 'input_json_delta', 'partial_json': fn['arguments']},
                        },
                    )
                )
        if choice.get('finish_reason') is not None:
            pending_stop = stop_reason_of(choice.get('finish_reason'))
        yield from events
    if not started:
        yield from start_events()
    if not finished:
        yield from finish_events()


# --------------------------------------------------------------------------
# The gateway: one dial, plus the failover loop around it
# --------------------------------------------------------------------------


def log(message: str) -> None:
    sys.stderr.write(f'aggregator: {message}\n')
    sys.stderr.flush()


def gateway_chat_url() -> str:
    if GATEWAY_URL == '':
        return ''
    return f'{GATEWAY_URL}/chat/completions' if GATEWAY_URL.endswith('/v1') else f'{GATEWAY_URL}/v1/chat/completions'


class Upstream:
    """One gateway answer: a status plus a body readable whole or in chunks."""

    def __init__(self, status: int, response) -> None:
        self.status = status
        self._response = response

    @property
    def ok(self) -> bool:
        return 200 <= self.status < 300

    def chunks(self):
        reader = getattr(self._response, 'read1', None) or self._response.read
        while True:
            chunk = reader(65536)
            if not chunk:
                break
            yield chunk

    def text(self) -> str:
        try:
            return self._response.read().decode('utf-8', 'replace')
        finally:
            self.close()

    def close(self) -> None:
        try:
            self._response.close()
        except Exception:
            pass


def dial(model_id: str, body: Json) -> Upstream:
    """POST one Chat Completions request to the gateway, naming `model_id`."""
    url = gateway_chat_url()
    if url == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    payload = json.dumps({**body, 'model': model_id}, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    headers = {'content-type': 'application/json', 'authorization': f'Bearer {GATEWAY_KEY}'}
    if body.get('stream') is True:
        headers['accept'] = 'text/event-stream'
    request = urllib.request.Request(url, data=payload, headers=headers, method='POST')
    try:
        response = urllib.request.urlopen(request, timeout=30)
    except urllib.error.HTTPError as err:
        return Upstream(err.code, err)
    return Upstream(response.status, response)


def error_code_of(status: int, text: str) -> str:
    """The provider's own error code from a refusal body, when it carries one."""
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if is_record(value):
        error = value.get('error')
        if is_record(error):
            for key in ('code', 'type'):
                if isinstance(error.get(key), str) and error[key] != '':
                    return error[key][:64]
        if isinstance(value.get('code'), str) and value['code'] != '':
            return value['code'][:64]
    return f'http_{status}'


def prepare_openai_body(body: Json) -> Json:
    """A client Chat Completions body ready for the gateway.

    The model is chosen by the router, the output cap is clamped to what the
    gateway accepts, and streaming requests always ask for the usage chunk so
    the streamed answer can carry whole usage counts.
    """
    out = dict(body)
    if is_number(out.get('max_completion_tokens')):
        out['max_completion_tokens'] = clamp_output_tokens(out.get('max_completion_tokens'))
    else:
        out['max_tokens'] = clamp_output_tokens(out.get('max_tokens'))
    if out.get('stream') is True:
        options = out.get('stream_options') if is_record(out.get('stream_options')) else {}
        out['stream_options'] = {**options, 'include_usage': True}
    return out


def dial_with_failover(requires: Json, make_body) -> tuple[Upstream | None, Json, list[Json]]:
    """Dial ranked candidates until one answers; collect the refusal trail.

    `make_body(model_id)` builds the gateway body for a candidate. Returns
    (upstream, model, failures): upstream is None when every candidate
    refused (or none exists), and failures holds one row per refused call in
    the order they happened, ready for the x-makefun-upstream-failures
    header. Each refusal also costs its model a cooldown strike, so a model
    that is rate-limiting sinks down the ranking for the prompts after it.
    """
    failures: list[Json] = []
    candidates = candidates_for(requires)[:MAX_ATTEMPTS_PER_PROMPT]
    for model in candidates:
        model_id = model['id']
        try:
            upstream = dial(model_id, make_body(model_id))
        except Exception as err:  # an unreachable gateway is one candidate down
            record_refusal(model_id)
            failures.append({'status': 502, 'code': 'upstream_unreachable', 'model': model_id})
            log(f'gateway unreachable for {model_id}: {err}')
            continue
        if upstream.ok:
            return upstream, model, failures
        text = upstream.text()
        code = error_code_of(upstream.status, text)
        record_refusal(model_id)
        failures.append({'status': upstream.status, 'code': code, 'model': model_id})
        log(f'gateway refused {model_id} with HTTP {upstream.status} ({code}); failing over')
    return None, {}, failures


def failures_header(failures: list[Json]) -> dict[str, str]:
    if not failures:
        return {}
    rows = failures[:MAX_FAILURE_ROWS]
    return {'x-makefun-upstream-failures': json.dumps(rows, separators=(',', ':'), ensure_ascii=False)}


# --------------------------------------------------------------------------
# The HTTP server
# --------------------------------------------------------------------------


class Handler(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    server_version = 'free-model-aggregator/1'
    sys_version = ''

    def log_message(self, format: str, *args: object) -> None:
        pass

    def do_GET(self) -> None:
        self.dispatch()

    def do_POST(self) -> None:
        self.dispatch()

    def dispatch(self) -> None:
        self.headers_sent = False
        self.extra_headers: dict[str, str] = {}
        path = urlsplit(self.path).path
        try:
            if self.command == 'GET' and path == '/v1/models':
                self.models()
                return
            if self.command == 'POST' and path == '/v1/chat/completions':
                self.chat()
                return
            if self.command == 'POST' and path == '/v1/messages':
                self.messages()
                return
            wire = 'anthropic' if path == '/v1/messages' else 'openai'
            self.send_error_body(wire, 404, 'not_found_error' if wire == 'anthropic' else 'not_found', f'no route {self.command} {path}')
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except Exception:
            log(traceback.format_exc())
            if not self.headers_sent:
                wire = 'anthropic' if urlsplit(self.path).path.startswith('/v1/messages') else 'openai'
                try:
                    self.send_error_body(wire, 500, 'api_error' if wire == 'anthropic' else 'internal_error', 'the server failed to answer')
                except (BrokenPipeError, ConnectionResetError):
                    self.close_connection = True

    # -- plumbing -----------------------------------------------------------

    def send_json(self, status: int, body: object) -> None:
        raw = json.dumps(body, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('content-type', 'application/json')
        self.send_header('content-length', str(len(raw)))
        for name, value in self.extra_headers.items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.wfile.write(raw)

    def send_error_body(self, wire: str, status: int, error_type: str, message: str) -> None:
        if wire == 'anthropic':
            self.send_json(status, anthropic_error(error_type, message))
        else:
            self.send_json(status, openai_error(message, error_type, None))

    def stream_headers(self) -> None:
        self.send_response(200)
        self.send_header('content-type', 'text/event-stream')
        self.send_header('cache-control', 'no-cache')
        self.send_header('connection', 'close')
        for name, value in self.extra_headers.items():
            self.send_header(name, value)
        self.headers_sent = True
        self.end_headers()
        self.close_connection = True

    def write_event(self, text: str) -> None:
        self.wfile.write(text.encode('utf-8'))
        self.wfile.flush()

    def read_json(self) -> Json | None:
        length = self.headers.get('content-length')
        if length is None:
            self.close_connection = True
            return None
        try:
            size = int(length)
        except ValueError:
            self.close_connection = True
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            self.close_connection = True
            return None
        raw = self.rfile.read(size) if size > 0 else b''
        if not raw:
            return None
        try:
            value = json.loads(raw.decode('utf-8'))
        except (UnicodeDecodeError, ValueError):
            return None
        return value if is_record(value) else None

    def give_up(self, wire: str, failures: list[Json]) -> None:
        """Every candidate refused: answer 502 in the client's own wire."""
        self.extra_headers = failures_header(failures)
        last = failures[-1] if failures else {}
        status = last.get('status') if is_number(last.get('status')) else 502
        if wire == 'anthropic':
            self.send_error_body(wire, 502, 'api_error', f'every free model refused this request (last: HTTP {status})')
        else:
            self.send_error_body(wire, 502, 'upstream_error', f'every free model refused this request (last: HTTP {status})')

    # -- routes ---------------------------------------------------------------

    def models(self) -> None:
        listing = listed_models()
        if self.headers.get('x-api-key') is not None:
            self.send_json(200, anthropic_model_list(listing))
        else:
            self.send_json(200, openai_model_list(listing))

    def chat(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('openai', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        prepared = prepare_openai_body(body)
        upstream, model, failures = dial_with_failover(requires, lambda _mid: prepared)
        if upstream is None:
            if not candidates_for(requires):
                self.send_error_body('openai', 503, 'no_free_model', 'no free model in the brief meets this request')
            else:
                self.give_up('openai', failures)
            return
        self.extra_headers = failures_header(failures)
        model_id = model['id']
        if body.get('stream') is True:
            self.stream_headers()
            try:
                for frame in openai_stream_frames(upstream.chunks(), model_id):
                    self.write_event(frame)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except ValueError:
            parsed = None
        self.send_json(200, {**parsed, 'model': model_id} if is_record(parsed) else openai_error('the gateway answer was not JSON', 'upstream_error', None))

    def messages(self) -> None:
        body = self.read_json()
        if body is None:
            self.send_error_body('anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object')
            return
        requires = requirements_for(body)
        upstream, model, failures = dial_with_failover(requires, lambda mid: anthropic_to_openai(body, mid))
        if upstream is None:
            if not candidates_for(requires):
                self.send_error_body('anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request')
            else:
                self.give_up('anthropic', failures)
            return
        self.extra_headers = failures_header(failures)
        model_id = model['id']
        if body.get('stream') is True:
            self.stream_headers()
            try:
                for event in anthropic_stream_events(upstream.chunks(), model_id):
                    self.write_event(event)
            finally:
                upstream.close()
            return
        answer = upstream.text()
        try:
            parsed = json.loads(answer)
        except ValueError:
            parsed = None
        if is_record(parsed):
            self.send_json(200, openai_to_anthropic(parsed, model_id))
        else:
            self.send_error_body('anthropic', 502, 'api_error', 'the gateway answer was not JSON')


# --------------------------------------------------------------------------
# Register, poll, start
# --------------------------------------------------------------------------


def register(port: int) -> None:
    body = json.dumps(
        {'protocol_version': PROTOCOL_VERSION, 'port': port, 'catalog': CATALOG},
        separators=(',', ':'),
        ensure_ascii=False,
    )
    request = urllib.request.Request(
        f'{ENV_URL}/register',
        data=body.encode('utf-8'),
        headers={'content-type': 'application/json'},
        method='POST',
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            status = response.status
            reply_text = response.read().decode('utf-8', 'replace')
    except urllib.error.HTTPError as err:
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {err.code}') from err
    if not (200 <= status < 300):
        raise RuntimeError(f'POST {ENV_URL}/register answered HTTP {status}: {reply_text[:500]}')
    try:
        reply = json.loads(reply_text)
    except ValueError:
        reply = None
    if is_record(reply) and is_number(reply.get('requests')):
        log(f'registered: the trial holds {reply["requests"]} prompts')


def poll() -> None:
    while True:
        try:
            with urllib.request.urlopen(f'{ENV_URL}/work', timeout=2) as response:
                if response.status == 200:
                    work = json.loads(response.read().decode('utf-8', 'replace'))
                    if is_record(work) and work.get('done') is True:
                        sys.stderr.flush()
                        os._exit(0)
        except Exception:
            pass
        time.sleep(POLL_INTERVAL_SEC)


def _stop(*_: object) -> None:
    sys.stderr.flush()
    os._exit(0)


def start() -> None:
    if BRIEF['error'] is not None:
        log(BRIEF['error'])
    if '--list' in sys.argv[1:]:
        sys.stdout.write(CATALOG_JSON + '\n')
        sys.stdout.flush()
        return
    if ENV_URL == '':
        raise RuntimeError('MAKEFUN_ENV_URL is not set')
    if gateway_chat_url() == '':
        raise RuntimeError('OPENAI_BASE_URL is not set')
    server = ThreadingHTTPServer(('0.0.0.0', 0), Handler)
    try:
        port = server.server_address[1]
        if port == 8080:
            raise RuntimeError('the OS picked 8080, which the env owns: restart the trial')
        log(f'listening on 0.0.0.0:{port} with {len(listed_models())} free models')
        register(port)
        signal.signal(signal.SIGTERM, _stop)
        signal.signal(signal.SIGINT, _stop)
        threading.Thread(target=poll, name='work-poll', daemon=True).start()
        server.serve_forever()
    finally:
        server.server_close()


def main() -> int:
    try:
        start()
        return 0
    except SystemExit:
        raise
    except BaseException:
        log(traceback.format_exc())
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
