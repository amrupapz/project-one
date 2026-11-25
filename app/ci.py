"""CI regression check for pull requests: python -m app.ci --base-sha SHA

For every prompts/*.yaml file the PR adds or modifies, run the eval on the base branch's
version (or, for a new file, on the base branch's copy of `baseline_prompt`) and on the
PR's version, with the PR's code, eval_config.yaml and dataset, then compare the runs.

Writes <out>/comment.md (a PR comment, Markdown) and <out>/report_<prompt>.html, and
exits with the compare status: 0 PASS, 1 WARNING, 2 CRITICAL, 3 error. If no prompt
changed, the comment says the eval was skipped and the exit code is 0.
"""

import argparse
import asyncio
import logging
import os
import re
import subprocess
import sys
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

import instructor
from pydantic import ValidationError

from app.errors import MissingAPIKeyError
from app.evals.compare import Comparison, CompareRefused, Delta, Flip, Status, compare_runs
from app.evals.config import BASELINE_PROMPT_PATTERN, DEFAULT_EVAL_CONFIG_PATH, REPO_ROOT, EvalConfig
from app.evals.judge_client import make_judge_client
from app.evals.models import EvalRun
from app.evals.report_html import render_report, trend_points
from app.evals.runner import execute_run
from app.evals.store import list_runs, save_run
from app.golden.io import GoldenDatasetError
from app.llm import API_KEY_ENV_VAR, make_client

COMMENT_MARKER = "<!-- llm-regression-check -->"
EXIT_ERROR = 3
MAX_FLIP_ROWS = 30
MAX_COMMENT_CHARS = 60_000  # GitHub's limit is 65,536
PROMPT_RE = re.compile(BASELINE_PROMPT_PATTERN)


# --- which prompts changed, and what to compare them against (pure) ------------------------

@dataclass(frozen=True)
class PromptChange:
    path: str  # repo-relative, e.g. prompts/v3.yaml
    kind: Literal["added", "modified"]


@dataclass(frozen=True)
class PlannedComparison:
    candidate_path: str  # read from the PR checkout
    baseline_path: str  # read from the base commit
    reason: Literal["base version", "baseline_prompt"]


def parse_name_status(output: str) -> list[PromptChange]:
    """Parse `git diff --name-status --no-renames`. Keeps added/modified prompts/*.yaml
    files directly under prompts/; deleted files are skipped."""
    changes = []
    for line in output.splitlines():
        if not line.strip():
            continue
        status, _, path = line.partition("\t")
        if not PROMPT_RE.match(path):
            continue
        if status == "A":
            changes.append(PromptChange(path, "added"))
        elif status == "M":
            changes.append(PromptChange(path, "modified"))
    return sorted(changes, key=lambda c: c.path)


def plan_comparisons(changes: list[PromptChange], baseline_prompt: str) -> list[PlannedComparison]:
    """A modified prompt is compared with its own base version; a new one with the base
    branch's copy of baseline_prompt (so a PR that also edits that file can't move the baseline)."""
    return [
        PlannedComparison(c.path, c.path, "base version")
        if c.kind == "modified"
        else PlannedComparison(c.path, baseline_prompt, "baseline_prompt")
        for c in changes
    ]


# --- git -------------------------------------------------------------------------------------

class Git(Protocol):
    def changed_files(self, base_sha: str) -> str: ...
    def show(self, base_sha: str, path: str) -> bytes: ...


class GitRepo:
    def __init__(self, root: Path = REPO_ROOT) -> None:
        self.root = root

    def _run(self, *args: str) -> subprocess.CompletedProcess[bytes]:
        return subprocess.run(["git", *args], cwd=self.root, capture_output=True, check=False)

    def changed_files(self, base_sha: str) -> str:
        result = self._run("diff", "--name-status", "--no-renames", base_sha, "HEAD", "--", "prompts/")
        if result.returncode != 0:
            raise RuntimeError(f"git diff against {base_sha} failed: {result.stderr.decode().strip()}")
        return result.stdout.decode()

    def show(self, base_sha: str, path: str) -> bytes:
        result = self._run("show", f"{base_sha}:{path}")
        if result.returncode != 0:
            raise FileNotFoundError(f"{path} does not exist on the base commit {base_sha[:12]}")
        return result.stdout


# --- running ---------------------------------------------------------------------------------

@dataclass
class PromptResult:
    plan: PlannedComparison
    comparison: Comparison | None = None
    report_path: Path | None = None
    error: str | None = None

    @property
    def exit_code(self) -> int:
        return EXIT_ERROR if self.comparison is None else int(self.comparison.status)


@dataclass
class CiOutcome:
    results: list[PromptResult] = field(default_factory=list)
    error: str | None = None  # a job-level problem, e.g. the API key is missing

    @property
    def exit_code(self) -> int:
        if self.error is not None:
            return EXIT_ERROR
        return max((r.exit_code for r in self.results), default=0)


RunFn = Callable[[Path, str], Coroutine[Any, Any, EvalRun]]  # (prompt file, notes) -> saved run


def overall_label(code: int) -> str:
    return {0: "PASS", 1: "WARNING", 2: "CRITICAL"}.get(code, "ERROR")


async def run_comparisons(
    plans: list[PlannedComparison],
    *,
    git: Git,
    base_sha: str,
    work_dir: Path,
    run_prompt: RunFn,
    eval_config: EvalConfig,
    db_path: Path,
    head_root: Path = REPO_ROOT,
) -> list[PromptResult]:
    """Run each side once (a base prompt shared by several new files runs only once),
    compare, and write one HTML report per changed prompt. One prompt's failure does not
    stop the others."""
    runs: dict[tuple[str, str], EvalRun] = {}

    async def run_side(side: Literal["base", "head"], path: str) -> EvalRun:
        key = (side, path)
        if key not in runs:
            if side == "base":
                # Keep the file name: PromptConfig requires the version id to match it.
                local = work_dir / "base" / path
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_bytes(git.show(base_sha, path))
                notes = f"ci baseline: {path} at {base_sha[:12]}"
            else:
                local = head_root / path
                notes = f"ci candidate: {path} from the PR"
            print(f"Running {side} {path}", file=sys.stderr, flush=True)
            runs[key] = await run_prompt(local, notes)
        return runs[key]

    results = []
    for plan in plans:
        result = PromptResult(plan)
        try:
            baseline = await run_side("base", plan.baseline_path)
            candidate = await run_side("head", plan.candidate_path)
            result.comparison = compare_runs(baseline, candidate, eval_config.compare)
            trend = trend_points(
                list_runs(db_path), candidate.info, eval_config.compare.trend_runs, eval_config.compare.min_judge_coverage
            )
            result.report_path = work_dir / f"report_{Path(plan.candidate_path).stem}.html"
            result.report_path.write_text(
                render_report(result.comparison, trend, generated_at=datetime.now(UTC)), encoding="utf-8"
            )
        except (CompareRefused, GoldenDatasetError, ValidationError, ValueError, FileNotFoundError, RuntimeError) as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            print(f"error: {plan.candidate_path}: {result.error}", file=sys.stderr, flush=True)
        results.append(result)
    return results


# --- PR comment (pure) -------------------------------------------------------------------------

STATUS_ICON = {"PASS": "✅", "WARNING": "⚠️", "CRITICAL": "❌", "ERROR": "❌"}


def _pct(v: float | None) -> str:
    return "–" if v is None else f"{v:.1%}"


def _pp(d: Delta) -> str:
    return "–" if d.change is None else f"{d.change * 100:+.1f} pp"


def _num(v: float | None, fmt: str) -> str:
    return "–" if v is None else format(v, fmt)


def _signed(v: float | None, fmt: str) -> str:
    return "–" if v is None else ("+" if v >= 0 else "") + format(v, fmt)


def _outcome(case: Any) -> str:
    if case.error is not None:
        return "ERROR"
    return case.predicted_category.value if case.predicted_category else "–"


def _code_block(text: str, limit: int = 800) -> str:
    text = text if len(text) <= limit else text[:limit] + " …"
    return "```\n" + text.replace("```", "'''") + "\n```"


def _flip_table(cmp: Comparison) -> list[str]:
    strict = {f.case_id: "regressed" for f in cmp.strict_regressions} | {
        f.case_id: "improved" for f in cmp.strict_improvements
    }
    acceptable = {f.case_id: "regressed" for f in cmp.acceptable_regressions} | {
        f.case_id: "improved" for f in cmp.acceptable_improvements
    }
    flips: dict[str, Flip] = {
        f.case_id: f
        for f in [*cmp.strict_regressions, *cmp.strict_improvements, *cmp.acceptable_regressions, *cmp.acceptable_improvements]
    }
    lines = [
        f"**Flipped cases**: strict {len(cmp.strict_regressions)} regressed / {len(cmp.strict_improvements)} improved; "
        f"acceptable {len(cmp.acceptable_regressions)} regressed / {len(cmp.acceptable_improvements)} improved.",
    ]
    if not flips:
        return lines
    ordered = sorted(flips.values(), key=lambda f: f.case_id)
    lines += ["", "| Case | Expected | Baseline → Candidate | Strict | Acceptable |", "|---|---|---|---|---|"]
    for f in ordered[:MAX_FLIP_ROWS]:
        lines.append(
            f"| {f.case_id} | {f.expected_category.value} | {_outcome(f.baseline)} → {_outcome(f.candidate)} "
            f"| {strict.get(f.case_id, '–')} | {acceptable.get(f.case_id, '–')} |"
        )
    if len(ordered) > MAX_FLIP_ROWS:
        lines.append(f"\n…and {len(ordered) - MAX_FLIP_ROWS} more in the HTML report.")
    return lines


def _comparison_section(result: PromptResult) -> list[str]:
    plan = result.plan
    against = (
        f"base `{plan.baseline_path}`"
        if plan.reason == "base version"
        else f"base `{plan.baseline_path}` (`baseline_prompt`; this is a new file)"
    )
    lines = [f"### `{plan.candidate_path}` vs {against}", ""]
    if result.comparison is None:
        return lines + ["**ERROR**: the comparison could not be completed.", "", _code_block(result.error or "unknown error"), ""]

    cmp = result.comparison
    lines.append(f"**{cmp.headline}**")
    lines += [f"- reason: {r}" for r in cmp.status_reasons]
    lines += [f"- ⚠️ {w}" for w in cmp.warnings]
    judge_note = "" if cmp.judge_used_in_status else " (not used for status)"
    lines += [
        "",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
        f"| Strict accuracy | {_pct(cmp.strict.baseline)} | {_pct(cmp.strict.candidate)} | {_pp(cmp.strict)} |",
        f"| Acceptable accuracy | {_pct(cmp.acceptable.baseline)} | {_pct(cmp.acceptable.candidate)} | {_pp(cmp.acceptable)} |",
        f"| Mean judge score{judge_note} | {_num(cmp.judge_score.baseline, '.2f')} | {_num(cmp.judge_score.candidate, '.2f')} | {_signed(cmp.judge_score.change, '.2f')} |",
        f"| Latency p50 (ms) | {_num(cmp.latency_p50_ms.baseline, ',.0f')} | {_num(cmp.latency_p50_ms.candidate, ',.0f')} | {_signed(cmp.latency_p50_ms.change, ',.0f')} |",
        f"| Latency p95 (ms) | {_num(cmp.latency_p95_ms.baseline, ',.0f')} | {_num(cmp.latency_p95_ms.candidate, ',.0f')} | {_signed(cmp.latency_p95_ms.change, ',.0f')} |",
        f"| Tokens in | {_num(cmp.input_tokens.baseline, ',.0f')} | {_num(cmp.input_tokens.candidate, ',.0f')} | {_signed(cmp.input_tokens.change, ',.0f')} |",
        f"| Tokens out | {_num(cmp.output_tokens.baseline, ',.0f')} | {_num(cmp.output_tokens.candidate, ',.0f')} | {_signed(cmp.output_tokens.change, ',.0f')} |",
        "",
        "<details><summary>Strict accuracy by category and difficulty</summary>",
        "",
        "| Group | n | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, g in [*cmp.by_category.items(), *cmp.by_difficulty.items()]:
        lines.append(f"| {key.value} | {g.n} | {_pct(g.strict.baseline)} | {_pct(g.strict.candidate)} | {_pp(g.strict)} |")
    lines += ["", "</details>", ""]
    lines += _flip_table(cmp)
    lines += [
        "",
        "> **Strict vs acceptable:** the status is decided by **strict** accuracy (prediction equals "
        "`expected_category`). Acceptable accuracy also counts a case's `acceptable_categories` as correct "
        "and is shown for context: a strict regression that is still acceptable is a softer change. "
        f"With {cmp.shared_cases} shared cases one case is about {cmp.points_per_case:.1f} pp, so small "
        "deltas can be noise.",
        "",
    ]
    lines += [f"_Note: {n}_" for n in cmp.notes]
    return lines


def format_pr_comment(
    outcome: CiOutcome,
    *,
    judge: bool,
    dataset: str,
    base_sha: str,
    run_url: str | None,
) -> str:
    label = "ERROR" if outcome.error else overall_label(outcome.exit_code)
    if not outcome.error and not outcome.results:
        label = "PASS"
    lines = [COMMENT_MARKER, f"## {STATUS_ICON[label]} LLM regression check: {label}", ""]

    meta = [f"dataset `{dataset}`", f"base `{base_sha[:7]}`"]
    meta.append(
        "judge **on**"
        if judge
        else "judge **off** (cost control; run this workflow manually with `judge` enabled to score summaries)"
    )
    if run_url:
        meta.append(f"[workflow run and HTML reports]({run_url})")
    lines += [" · ".join(meta), ""]

    if outcome.error:
        lines += [_code_block(outcome.error), ""]
    elif not outcome.results:
        lines += ["No prompt files were added or changed, so the eval was skipped. "
                  "Unit tests and dataset validation ran in this job.", ""]
    else:
        if len(outcome.results) > 1:
            lines += ["| Prompt | Status |", "|---|---|"]
            for r in outcome.results:
                lines.append(f"| `{r.plan.candidate_path}` | {overall_label(r.exit_code)} |")
            lines.append("")
        for r in outcome.results:
            lines += _comparison_section(r)
        lines.append("Gate: **CRITICAL** or **ERROR** fails this check; **WARNING** passes.")

    text = "\n".join(lines).rstrip() + "\n"
    if len(text) > MAX_COMMENT_CHARS:
        text = text[: MAX_COMMENT_CHARS - 200] + "\n\n…(truncated; see the HTML reports)\n"
    return text


# --- entry point --------------------------------------------------------------------------------

def _progress(done: int, total: int) -> None:
    if done == total:
        print(f"  {done}/{total} cases", file=sys.stderr, flush=True)


def _failure(case_id: str, stage: str, code: str) -> None:
    print(f"  {stage} failed: {case_id} ({code})", file=sys.stderr, flush=True)


def run_url_from_env() -> str | None:
    server, repo, run_id = (os.environ.get(k) for k in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID"))
    return f"{server}/{repo}/actions/runs/{run_id}" if server and repo and run_id else None


async def run_ci(
    *,
    base_sha: str,
    out_dir: Path,
    eval_config_path: Path,
    dataset_path: Path,
    judge: bool,
    git: Git,
    head_root: Path = REPO_ROOT,
    classifier_client: instructor.AsyncInstructor | None = None,
    judge_client: instructor.AsyncInstructor | None = None,
) -> CiOutcome:
    eval_config = EvalConfig.from_yaml(eval_config_path)
    plans = plan_comparisons(parse_name_status(git.changed_files(base_sha)), eval_config.baseline_prompt)
    if not plans:
        print("No prompt files changed; eval skipped.", file=sys.stderr)
        return CiOutcome()

    if not os.environ.get(API_KEY_ENV_VAR) and classifier_client is None:
        message = (
            f"{API_KEY_ENV_VAR} is not set. Add it as a repository secret (Settings > Secrets and variables > "
            "Actions). Pull requests from forks never receive secrets."
        )
        print(f"::error title=Missing secret::{message}")  # GitHub annotation; the key itself is never printed
        return CiOutcome(error=message)

    logging.getLogger("instructor").setLevel(logging.CRITICAL)  # its error logs include full response bodies
    try:
        classifier_client = classifier_client or make_client()
        if judge and judge_client is None:
            judge_client = make_judge_client(eval_config)  # one limiter shared by every judged run in the job
    except MissingAPIKeyError as exc:
        return CiOutcome(error=str(exc))

    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / "eval.db"  # fresh per job: CI never depends on results/eval.db

    async def run_prompt(prompt_path: Path, notes: str) -> EvalRun:
        run = await execute_run(
            prompt_path,
            dataset_path,
            eval_config_path,
            notes=notes,
            judge=judge,
            classifier_client=classifier_client,
            judge_client=judge_client,
            on_progress=_progress,
            on_failure=_failure,
        )
        save_run(db_path, run)
        return run

    results = await run_comparisons(
        plans, git=git, base_sha=base_sha, work_dir=out_dir, run_prompt=run_prompt,
        eval_config=eval_config, db_path=db_path, head_root=head_root,
    )
    return CiOutcome(results=results)


def main(argv: list[str] | None = None, git: Git | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.ci", description=__doc__.splitlines()[0])
    parser.add_argument("--base-sha", required=True, help="commit the PR is compared against")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "ci", help="output folder")
    parser.add_argument("--judge", action="store_true", help="also run the LLM judge (slower, costs more)")
    parser.add_argument("--dataset", type=Path, default=REPO_ROOT / "golden" / "v1.json")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH)
    args = parser.parse_args(argv)

    try:
        outcome = asyncio.run(run_ci(
            base_sha=args.base_sha, out_dir=args.out, eval_config_path=args.eval_config,
            dataset_path=args.dataset, judge=args.judge, git=git or GitRepo(),
        ))
    except (RuntimeError, ValueError, ValidationError, GoldenDatasetError, FileNotFoundError) as exc:
        outcome = CiOutcome(error=f"{type(exc).__name__}: {exc}")
        print(f"error: {outcome.error}", file=sys.stderr)

    dataset = args.dataset.resolve()
    dataset_label = str(dataset.relative_to(REPO_ROOT)) if dataset.is_relative_to(REPO_ROOT) else str(args.dataset)
    comment = format_pr_comment(
        outcome, judge=args.judge, dataset=dataset_label, base_sha=args.base_sha, run_url=run_url_from_env()
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "comment.md").write_text(comment, encoding="utf-8")
    print(comment)
    return outcome.exit_code


if __name__ == "__main__":
    sys.exit(main())
