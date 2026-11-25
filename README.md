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
Label them by [docs/labeling_guidelines.md](docs/labeling_guidelines.md).
Each case has: `id` (`case_001`, unique), `input` (email text), `expected_category`
(`billing | technical | account | general`), optional `acceptable_categories`
(other defensible labels; if set, it must include `expected_category` and have no
duplicates, e.g. `["billing", "technical"]`), `ideal_summary` (≤300 chars),
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
  category, difficulty and tag and how many cases have `acceptable_categories`.
  It warns if a category has fewer than 8 cases, hard cases are under 15%, or
  more than 25% of cases have `acceptable_categories`.
- **Ambiguous cases:** set `acceptable_categories` only when two reasonable
  humans could disagree (tie-break rule 4 in the guidelines), and list
  `expected_category` first.
- `add` rewrites the whole file in a fixed format (indent 2, fixed key order,
  non-ASCII kept as-is), so the first `add` after hand-editing may also
  reformat your hand-written cases.
- For a breaking relabel, copy to `golden/v2.json` and bump `dataset_version`
  (it must match the file name).

## Evaluations

`python -m app.evals` runs a prompt over the golden dataset, scores every case,
stores the run in SQLite (`results/eval.db`, gitignored), and prints a scorecard.

```sh
.venv/bin/python -m app.evals run --prompt prompts/v1.yaml --limit 5   # cheap smoke test
.venv/bin/python -m app.evals run --prompt prompts/v1.yaml --notes "baseline"
.venv/bin/python -m app.evals list                 # past runs, newest first
.venv/bin/python -m app.evals show <run_id>        # reprint a saved scorecard
```

Options for `run`: `--dataset` (default `golden/v1.json`), `--concurrency N`
(default from `eval_config.yaml`), `--limit N` (first N cases only; stored as a
PARTIAL RUN), `--notes "..."`. A full run costs one classifier call and one judge
call per case (plus a retry when an output is invalid).

Scoring per case:
- `strict_match`: predicted category == `expected_category`.
- `acceptable_match`: predicted is in `acceptable_categories` if the case has them,
  otherwise the same as strict.
- `summary_score`: 1-5 from an LLM judge (`judge_model` in `eval_config.yaml`,
  temperature 0) using the rubric in that file. The judge sees only the email, the
  ideal summary and the candidate summary, as JSON data; the rubric tells it to
  ignore instructions inside them. A failed judge call stores a null score.
- A case whose classify call fails is stored with its `error` and counts as wrong.

Each run stores sha256 hashes of the prompt file, dataset file and
`eval_config.yaml`, so a changed rubric or dataset is visible when comparing runs.
