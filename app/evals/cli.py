"""python -m app.evals {run,list,show}"""

import argparse
import asyncio
import sys
from pathlib import Path

from pydantic import ValidationError

from app.errors import MissingAPIKeyError
from app.evals.config import DEFAULT_EVAL_CONFIG_PATH, REPO_ROOT, EvalConfig
from app.evals.runner import execute_run
from app.evals.scorecard import compute_scorecard, format_scorecard
from app.evals.store import list_runs, load_run, save_run
from app.golden.io import GoldenDatasetError
from app.llm import make_client

DEFAULT_DATASET_PATH = REPO_ROOT / "golden" / "v1.json"


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def _progress(done: int, total: int) -> None:
    print(f"\r  {done}/{total} cases", end="" if done < total else "\n", file=sys.stderr, flush=True)


def cmd_run(args: argparse.Namespace) -> int:
    eval_config = EvalConfig.from_yaml(args.eval_config)
    # Build both clients up front so a missing key fails before any work starts.
    classifier_client = make_client()
    judge_client = make_client()
    run = asyncio.run(
        execute_run(
            args.prompt,
            args.dataset,
            args.eval_config,
            concurrency=args.concurrency,
            limit=args.limit,
            notes=args.notes,
            classifier_client=classifier_client,
            judge_client=judge_client,
            on_progress=_progress,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.evals", description="Evaluate a prompt against the golden dataset.")
    parser.add_argument("--eval-config", type=Path, default=DEFAULT_EVAL_CONFIG_PATH, help="default: eval_config.yaml")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run an eval, save it, print the scorecard")
    run.add_argument("--prompt", type=Path, required=True, help="e.g. prompts/v1.yaml")
    run.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH, help="default: golden/v1.json")
    run.add_argument("--concurrency", type=_positive_int, help="default: from eval_config.yaml")
    run.add_argument("--limit", type=_positive_int, help="only the first N cases (a PARTIAL run)")
    run.add_argument("--notes", help="free text stored with the run")
    run.set_defaults(func=cmd_run)

    sub.add_parser("list", help="list past runs, newest first").set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="print the scorecard of a saved run")
    show.add_argument("run_id")
    show.set_defaults(func=cmd_show)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (MissingAPIKeyError, GoldenDatasetError, ValidationError, ValueError, KeyError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
