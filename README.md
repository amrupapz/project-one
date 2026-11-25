# LLM regression lab: Phase 1, email classifier

Classifies a customer support email into `billing | technical | account | general`
with a one-sentence summary, using versioned prompts in `prompts/`.

## Setup

```sh
uv sync                     # Python 3.11, creates .venv
cp .env.example .env        # add your OpenRouter key
set -a; source .env; set +a # export OPENROUTER_API_KEY
```

## Run

```sh
.venv/bin/python -m app.classify --prompt prompts/v1.yaml "I was charged twice this month"
```

## Test

```sh
.venv/bin/pytest -q
```

## Prompt files

`prompts/<version>.yaml` with fields `version` (`v1`, `v2`, ...; must match the
file name), `created_at`, `model_name` (OpenRouter id), `temperature` (0-2,
default 0), `system_prompt`, and optional `few_shot_examples` (`input`, `output`).
`v2` is deliberately weaker than `v1` for regression testing.

Instructor mode (JSON vs tools) is set by `INSTRUCTOR_MODE` in `app/llm.py`.
