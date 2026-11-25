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

## Golden dataset

`golden/v1.json` holds human-written, human-labeled cases. Never generate cases.
Each case has: `id` (`case_001`, unique), `input` (email text), `expected_category`
(`billing | technical | account | general`), `ideal_summary` (≤300 chars),
`expected_difficulty` (`easy | medium | hard`), optional `tags` (snake_case, e.g.
`typo`, `sarcasm`, `mixed_language`, `ambiguous`, `very_short`, `injection`, `vague`),
and `notes` (required: why the case matters or why the label was chosen).

### How to add or edit cases

- **Add interactively:** `.venv/bin/python -m app.golden add`. It asks for each
  field, re-asks on invalid values, assigns the next id, shows the case, and
  writes the file after you confirm. End the email with a line containing only `.`.
- **Edit by hand:** open `golden/v1.json` and paste or edit cases directly. Line
  breaks inside an email must be written as `\n` (JSON has no multi-line strings).
  Never renumber or reuse ids: Phase 3 results refer to them.
- **Check your edits:** `.venv/bin/python -m app.golden validate` lists every
  problem with its case id (duplicates, bad enums, missing notes, JSON syntax
  errors with line numbers). `add` and `stats` refuse to run on an invalid file.
- **Check balance:** `.venv/bin/python -m app.golden stats` shows counts by
  category, difficulty and tag, and warns if a category has fewer than 8 cases
  or hard cases are under 15%.
- `add` rewrites the whole file in a fixed format (indent 2, fixed key order,
  non-ASCII kept as-is), so the first `add` after hand-editing may also
  reformat your hand-written cases.
- For a breaking relabel, copy to `golden/v2.json` and bump `dataset_version`
  (it must match the file name).
