# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/jensheng-builds/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/jensheng-builds/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

A live free-model aggregator: one zero-config server that serves the OpenAI Chat Completions API and the Anthropic Messages API (JSON and streaming) and answers every request with a real completion from a free OpenRouter model. It ranks the free catalog by published quality rating, filters by each request's requirements (tools, vision, JSON mode, context), fails over down the ranked list when a provider refuses, cools down models that keep refusing, and reports recovered refusals on the `x-makefun-upstream-failures` header. The `model` field of every answer names the model that was really dialled.

## Run it

```sh
python3 main.py --list   # print the catalog document
python3 main.py          # serve (trial mode: reads $MAKEFUN_PROMPT_FILE, registers with $MAKEFUN_ENV_URL)
```

In a make.fun trial the server reads the brief at `$MAKEFUN_PROMPT_FILE`, registers its port and catalog with the env at `$MAKEFUN_ENV_URL`, dials free models through the gateway at `$OPENAI_BASE_URL`, and exits 0 when the env's `/work` says done. Model calls go out on the trial's key; the entrant's own OpenRouter key (`OPENROUTER_API_KEY`, pushed to the competition as an entry secret) never enters the sandbox and is never committed to this repository.

---

<sub>Entered on [make.fun](https://make.fun/?ref=me77q2r5&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
