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
- For a breaking relabel, copy to the next free version (`golden/v3.json`; `v2` is
  the held-out set, see below) and bump `dataset_version` (it must match the file name).

## Datasets

| File | Role | Cases |
|---|---|---|
| `golden/v1.json` | **Main dataset.** Use it to develop and tune prompts and for regression runs. | 68 |
| `golden/v2.json` | **Held-out check.** Run it only to confirm that a prompt tuned on v1 generalizes. Never tune prompts on it, and never add its cases to v1. | 12 |

```sh
.venv/bin/python -m app.evals run --prompt prompts/v3.yaml --dataset golden/v2.json
```

v2 scores are never compared directly with v1 scores: they are different cases of
different difficulty, so a number from one says nothing about the other. Compare a
held-out run only with another held-out run. `compare` enforces this, because it
refuses runs whose dataset files differ and picks its default baseline from runs on the
same dataset file. With 12 cases, one case is about 8.3 pp, so a held-out run can
show a large difference but cannot show a small one. Treat it as a sanity check, not a
precise score.

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
PARTIAL RUN), `--notes "..."`, `--no-judge` (skip the summary judge: every
`summary_score` is null, the run is stored with `judge_enabled = 0` and its scorecard
says NOT JUDGED; takes seconds instead of minutes). A full run costs one classifier
call and one judge call per case (plus a retry when an output is invalid).

Scoring per case:
- `strict_match`: predicted category == `expected_category`.
- `acceptable_match`: predicted is in `acceptable_categories` if the case has them,
  otherwise the same as strict.
- `summary_score`: 1-5 from an LLM judge (`judge_model` in `eval_config.yaml`,
  temperature 0) using the rubric in that file. The judge sees only the email, the
  ideal summary and the candidate summary, as JSON data; the rubric tells it to
  ignore instructions inside them. A failed judge call stores a null score.
- A case whose classify call fails is stored with its `error` and counts as wrong.

Judge requests are rate-limited across the whole run to `judge_requests_per_minute`
(default 15; OpenRouter allows new accounts 20/min on some models), so a full run of
68 cases takes about 4-5 minutes. A 429 is retried up to 3 times, honoring
Retry-After (capped at 60 s); every queued judge call waits out that pause. After
that the case gets a null score, and the scorecard warns when more than 5% of judge
calls failed. Failures print as one short line each, e.g. `judge failed: case_012 (429)`.

Each run stores sha256 hashes of the prompt file, dataset file and
`eval_config.yaml`, so a changed rubric or dataset is visible when comparing runs.

The database schema is versioned (`PRAGMA user_version`, now 2). Opening a version-1
database migrates it in place (adds `runs.judge_enabled`, existing runs count as
judged) after copying it to `results/eval.db.bak` (or `.bak.1`, ... if that exists).

## Comparing runs

```sh
.venv/bin/python -m app.evals compare                      # latest run vs previous full run
.venv/bin/python -m app.evals compare --baseline 20261125-112215-81ac --candidate 20261125-112841-1718
.venv/bin/python -m app.evals compare --html /tmp/report.html --no-alert
```

Defaults: the candidate is the latest run; the baseline is the most recent earlier
**full** run on the same dataset file (a `--no-judge` run is never picked as the
baseline of a judged candidate). The command prints a summary, writes a
self-contained HTML report (default `results/report_<candidate>.html`: run metadata,
scorecard, every regression and improvement with old and new output side by side,
and trend charts of strict accuracy and judge score over the last `trend_runs` full
runs), and exits with the status:

| Exit | Status | Meaning |
|---|---|---|
| 0 | PASS | strict accuracy dropped less than `strict_drop_warning_pp` (3) |
| 1 | WARNING | strict drop ≥ 3 pp, or mean judge score dropped ≥ `judge_score_drop_warning` (0.3) |
| 2 | CRITICAL | strict drop ≥ `strict_drop_critical_pp` (8) |
| 3 | error | refused comparison, unknown run id, no baseline, bad arguments |

Thresholds live under `compare:` in `eval_config.yaml`. The status line always
carries the flip counts, e.g. `PASS (strict -2.9 pp; 4 regressions, 2 improvements)`,
and the report lists every regression whatever the status. Acceptable-match flips
are listed separately and don't drive the status.

How runs are matched and when the result is flagged:
- Metrics are computed on the case ids both runs share.
- Different `dataset_sha256`: refused (exit 3).
- A partial run (`--limit`) or a different `eval_config_sha256`: compared, with a
  prominent WARNING.
- Judge coverage below `min_judge_coverage` (95%) in either run, or a `--no-judge`
  run on either side: the judge delta is shown as unreliable or n/a and is not used
  for the status.
- Noise: with 68 cases one case is about 1.5 pp, so WARNING needs a net loss of 3
  cases (4.4 pp; 2 cases is 2.9 pp) and CRITICAL a net loss of 6 (8.8 pp). Check the
  noise level with `repeat`:

```sh
.venv/bin/python -m app.evals repeat --prompt prompts/v1.yaml --n 5 --no-judge
```

`repeat` runs the prompt N times (sequentially, never judged, not saved) and prints
the strict accuracy of each run, their mean, stdev and spread, and every case whose
prediction changed between runs. A compare delta inside that spread is likely noise.

### Slack alerts

Set `SLACK_WEBHOOK_URL` (an incoming webhook) and `compare` posts the status line,
the strict / acceptable / judge numbers, any warnings and the report path. It never
sends email text. Unset means no alert; `--no-alert` skips it; a failed post prints a
warning and does not change the exit code. Other channels can implement the
`Alerter` protocol in `app/evals/alerts.py`.
