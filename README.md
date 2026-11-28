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

`compare` also prints the slow-drift line for the candidate's prompt (see
[Slow-drift detection](#slow-drift-detection)) and exits with the **worse** of the two
statuses, so a PASS compare can still exit 1 on DRIFT WARNING.

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

`repeat` runs the prompt N times (sequentially, never judged, not saved by default)
and prints the strict accuracy of each run, their mean, stdev and spread, and every
case whose prediction changed between runs. A compare delta inside that spread is
likely noise. With `--save`, each run is stored in `eval.db` with the notes
`repeat k/N` (shown by `list`). Saved repeat runs are **real runs** for drift
purposes: they count towards the moving average like any other full run.

### Slack alerts

Set `SLACK_WEBHOOK_URL` (an incoming webhook) and `compare` posts the status line,
the strict / acceptable / judge numbers, any warnings, the drift line when drift is
not OK, and the report path. It never
sends email text. Unset means no alert; `--no-alert` skips it; a failed post prints a
warning and does not change the exit code. Other channels can implement the
`Alerter` protocol in `app/evals/alerts.py`.

## CI: regression check on pull requests

`.github/workflows/prompt-regression.yml` runs on pull requests that touch
`prompts/**`, `golden/**`, `eval_config.yaml` or `app/**`:

1. Installs with uv (Python from `.python-version`, dependencies cached on `uv.lock`).
2. Runs the unit tests and validates every `golden/*.json`. No API key is needed for this.
3. Runs `python -m app.ci --base-sha <base>`. For each `prompts/*.yaml` file the PR
   **adds or modifies**, it runs the eval on the base branch's version of that file
   and on the PR's version, with the PR's code, `eval_config.yaml` and
   `golden/v1.json`, in a fresh `results/ci/eval.db`, then compares the two runs.
   A **new** prompt file is compared against the base branch's copy of
   `baseline_prompt` (set in `eval_config.yaml`, default `prompts/v1.yaml`). If no
   prompt file changed, the eval is skipped.
4. Posts one PR comment (updated in place on every push) with the status, the
   delta tables, the flipped cases and the strict-vs-acceptable note, and uploads
   the HTML reports as the `regression-reports` artifact.
5. Fails the check on **CRITICAL** (exit 2) or an error (exit 3). **WARNING** passes
   but is shown in the comment.

The judge is off on pull requests to save cost. To score summaries, run the workflow
manually (Actions > prompt-regression > Run workflow) on the PR's branch with
`judge` enabled; it updates the same PR comment. The merge gate is still the
automatic pull_request run.

Run the same check locally against `main`:

```sh
.venv/bin/python -m app.ci --base-sha "$(git merge-base main HEAD)"   # add --judge to score summaries
```

Limits:
- PRs from forks get no secrets, so their check fails with a "missing
  OPENROUTER_API_KEY" message. The workflow uses `pull_request`, never
  `pull_request_target`, so fork code can never reach the key.
- Edit a prompt by adding a new version file rather than changing an existing one.
  Runs record `prompt_file_sha256`, but two different files both called `v2` are
  easy to confuse.

## Slow-drift detection

Each compare looks at two runs, so a series of small drops that each stay under the
3 pp WARNING threshold never fails one. The drift check compares the mean of the last
7 qualifying runs with the mean of the 7 qualifying runs before them:

```sh
.venv/bin/python -m app.evals drift --prompt-version v3                     # dataset default golden/v1.json
.venv/bin/python -m app.evals drift --prompt-version v3 --dataset golden/v2.json
```

It prints the window mean, reference mean and delta for strict and acceptable
accuracy, the status and the run ids used. A run qualifies when it is a **full** run
(no `--limit`), has **no errored cases**, and matches the prompt version, the dataset
file (`dataset_sha256`), and the prompt file (`prompt_file_sha256`) and classifier
model of the newest such run. An edited prompt file or a model switch therefore starts
a new history instead of showing up as drift. Prompt versions and datasets are never
mixed.

| Exit | Status | Meaning |
|---|---|---|
| 0 | NOT ENOUGH HISTORY | fewer than `min_history` (14 = 2 × window) qualifying runs; never warns |
| 0 | OK | both means within the thresholds |
| 1 | DRIFT WARNING | strict or acceptable 7-run mean dropped ≥ `drop_warning_pp` (2) |
| 2 | DRIFT CRITICAL | dropped ≥ `drop_critical_pp` (4), or the strict window mean is below `floor_strict_pct` (off by default) |
| 3 | error | bad arguments, missing dataset file |

Settings live under `drift:` in `eval_config.yaml`. They are tighter than the per-run
thresholds because a mean of 7 runs wobbles much less than one run. The mean judge
score is shown only if at least 95% of the runs in both windows have one, and it never
changes the status. `repeat` runs are never judged, so a history built with
`repeat --save` shows no judge drift.

The `compare` output and HTML report carry one line, e.g. `Drift (7-run average):
DRIFT WARNING (strict 90.0% vs 93.5% over the previous 7 runs, -3.5 pp; ...)`,
computed as of the candidate run. The strict-accuracy trend chart gets a dashed
7-run moving-average line for the candidate's prompt.

To build history, run the production prompt repeatedly and save the runs:

```sh
.venv/bin/python -m app.evals repeat --prompt prompts/v3.yaml --n 14 --save
.venv/bin/python -m app.evals drift --prompt-version v3
```

Back-to-back runs only measure noise at one moment, so right after this the check
says OK. Drift shows up as new runs arrive over days or weeks, for example a daily
`.venv/bin/python -m app.evals run --prompt prompts/v3.yaml --no-judge` (or
`repeat --n 2 --save`). Each new run pushes the oldest one out of the reference window.

**Not wired into CI.** The pull-request workflow uses a fresh `results/ci/eval.db`
with no history, so the drift check would always say NOT ENOUGH HISTORY. Running it
in CI would need the run history to be stored between workflow runs (for example as
a cached or committed database file, or a scheduled job writing to persistent
storage, since `results/` is gitignored). That is out of scope for now.
