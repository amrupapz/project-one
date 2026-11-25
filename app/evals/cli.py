"""python -m app.evals {run,list,show,compare,repeat}"""

import argparse
import asyncio
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from app.errors import MissingAPIKeyError
from app.evals.alerts import AlertError, SlackAlerter
from app.evals.compare import compare_runs, format_comparison
from app.evals.config import DEFAULT_EVAL_CONFIG_PATH, REPO_ROOT, EvalConfig
from app.evals.judge_client import make_judge_client
from app.evals.repeat import format_spread, repeat_runs, repeat_spread
from app.evals.report_html import render_report, trend_points
from app.evals.runner import execute_run
from app.evals.scorecard import compute_scorecard, format_scorecard
from app.evals.store import find_default_baseline, list_runs, load_run, save_run
from app.golden.io import GoldenDatasetError
from app.llm import make_client

DEFAULT_DATASET_PATH = REPO_ROOT / "golden" / "v1.json"

# compare exits 0 PASS / 1 WARNING / 2 CRITICAL, so its errors (and argparse usage errors,
# which would otherwise exit 2) use 3 and can never be mistaken for a status.
EXIT_ERROR = 3


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        self.print_usage(sys.stderr)
        self.exit(EXIT_ERROR, f"{self.prog}: error: {message}\n")


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _progress(done: int, total: int) -> None:
    if sys.stderr.isatty():
        print(f"\r  {done}/{total} cases", end="" if done < total else "\n", file=sys.stderr, flush=True)
    elif done == total:  # logs and pipes: no live counter, just the final count
        print(f"  {done}/{total} cases", file=sys.stderr, flush=True)


def _failure(case_id: str, stage: str, code: str) -> None:
    # One short line per failure. On a terminal, first clear the progress counter line.
    clear = "\r\033[K" if sys.stderr.isatty() else ""
    print(f"{clear}  {stage} failed: {case_id} ({code})", file=sys.stderr, flush=True)


def _quiet_library_logs() -> None:
    # instructor logs every failed API call with the full response body at ERROR level.
    # Failures are reported compactly by _failure() and stored in the run instead.
    logging.getLogger("instructor").setLevel(logging.CRITICAL)


def cmd_run(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    _quiet_library_logs()
    # Build both clients up front so a missing key fails before any work starts.
    classifier_client = make_client()
    judge_client = None if args.no_judge else make_judge_client(eval_config)
    run = asyncio.run(
        execute_run(
            args.prompt,
            args.dataset,
            args.eval_config,
            concurrency=args.concurrency,
            limit=args.limit,
            notes=args.notes,
            judge=not args.no_judge,
            classifier_client=classifier_client,
            judge_client=judge_client,
            on_progress=_progress,
            on_failure=_failure,
        )
    )
    save_run(eval_config.db_path, run)
    print(format_scorecard(run.info, compute_scorecard(run.cases)))
    print(f"\nSaved run {run.info.run_id} to {eval_config.db_path}")
    return 0


def _acc(value: float | None) -> str:
    return "-" if value is None else f"{value:.1%}"


def cmd_list(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    runs = list_runs(eval_config.db_path)
    if not runs:
        print(f"No runs yet in {eval_config.db_path}")
        return 0
    header = f"{'run_id':<22} {'started (UTC)':<19} {'prompt':<7} {'model':<28} {'cases':>5} {'strict':>7} {'accept':>7} {'errors':>6}"
    print(header)
    for r in runs:
        partial = f"  PARTIAL RUN (limit {r.info.case_limit})" if r.info.is_partial else ""
        print(
            f"{r.info.run_id:<22} {r.info.started_at:%Y-%m-%d %H:%M:%S} {r.info.prompt_version:<7} "
            f"{r.info.classifier_model:<28} {r.case_count:>5} {_acc(r.strict_accuracy):>7} "
            f"{_acc(r.acceptable_accuracy):>7} {r.errors:>6}{partial}"
        )
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    run = load_run(eval_config.db_path, args.run_id)
    print(format_scorecard(run.info, compute_scorecard(run.cases)))
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    runs = list_runs(eval_config.db_path)
    if not runs:
        raise ValueError(f"no runs in {eval_config.db_path}")
    candidate = load_run(eval_config.db_path, args.candidate or runs[0].info.run_id)
    if args.baseline:
        baseline = load_run(eval_config.db_path, args.baseline)
    else:
        default = find_default_baseline(runs, candidate.info)
        if default is None:
            raise ValueError(
                f"no earlier full run on the same dataset as {candidate.info.run_id}"
                + (" (no-judge runs are skipped for a judged candidate)" if candidate.info.judge_enabled else "")
                + "; pass --baseline RUN_ID"
            )
        baseline = load_run(eval_config.db_path, default.run_id)

    cmp = compare_runs(baseline, candidate, eval_config.compare)
    print(format_comparison(cmp))

    report_path = args.html or eval_config.db_path.parent / f"report_{candidate.info.run_id}.html"
    trend = trend_points(runs, candidate.info, eval_config.compare.trend_runs)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(cmp, trend, generated_at=datetime.now(UTC)), encoding="utf-8")
    print(f"\nReport: {report_path}")

    if not args.no_alert:
        alerter = SlackAlerter.from_env()
        if alerter is not None:
            try:
                alerter.send(cmp, report_path)
                print("Slack alert sent.")
            except AlertError as exc:  # an alert problem never changes the status
                print(f"warning: {exc}", file=sys.stderr)
    return int(cmp.status)


def cmd_repeat(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    _quiet_library_logs()
    classifier_client = make_client()

    async def one(i: int):
        print(f"Run {i + 1}/{args.n}", file=sys.stderr, flush=True)
        return await execute_run(
            args.prompt,
            args.dataset,
            args.eval_config,
            concurrency=args.concurrency,
            limit=args.limit,
            judge=False,
            classifier_client=classifier_client,
            on_progress=_progress,
            on_failure=_failure,
        )

    runs = asyncio.run(repeat_runs(one, args.n))
    print(format_spread(repeat_spread(runs), runs[0].info.prompt_version))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(prog="python -m app.evals", description="Evaluate a prompt against the golden dataset.")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH, help="default: eval_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True, parser_class=_Parser)

    run = sub.add_parser("run", help="run an eval, save it, print the scorecard")
    run.add_argument("--prompt", type=Path, required=True, help="e.g. prompts/v1.yaml")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="default: golden/v1.json")
    run.add_argument("--concurrency", type=_positive_int, help="default: from eval_config.yaml")
    run.add_argument("--limit", type=_positive_int, help="only the first N cases (a PARTIAL run)")
    run.add_argument("--notes", help="free text stored with the run")
    run.add_argument("--no-judge", action="store_true", help="skip the summary judge (null scores; fast, cheap)")
    run.set_defaults(func=cmd_run)

    sub.add_parser("list", help="list past runs, newest first").set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="print the scorecard of a saved run")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    compare = sub.add_parser("compare", help="compare two runs; exit 0 PASS, 1 WARNING, 2 CRITICAL, 3 error")
    compare.add_argument("--baseline", metavar="RUN_ID", help="default: most recent earlier full run on the same dataset")
    compare.add_argument("--candidate", metavar="RUN_ID", help="default: latest run")
    compare.add_argument("--html", type=Path, help="report path (default: results/report_<candidate>.html)")
    compare.add_argument("--no-alert", action="store_true", help="don't post to Slack even if SLACK_WEBHOOK_URL is set")
    compare.set_defaults(func=cmd_compare, error_exit=EXIT_ERROR)

    repeat = sub.add_parser("repeat", help="run one prompt N times without the judge and report the spread")
    repeat.add_argument("--prompt", type=Path, required=True)
    repeat.add_argument("--n", type=_positive_int, default=5, help="repetitions (>= 2, default 5)")
    repeat.add_argument("--no-judge", action="store_true", help="accepted for clarity; repeat never calls the judge")
    repeat.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="default: golden/v1.json")
    repeat.add_argument("--concurrency", type=_positive_int, help="default: from eval_config.yaml")
    repeat.add_argument("--limit", type=_positive_int, help="only the first N cases")
    repeat.set_defaults(func=cmd_repeat)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MissingAPIKeyError, GoldenDatasetError, ValidationError, ValueError, KeyError, FileNotFoundError, OSError) as exc:
        message = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        print(f"error: {message}", file=sys.stderr)
        return getattr(args, "error_exit", 1)
