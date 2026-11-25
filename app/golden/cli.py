"""python -m app.golden {validate,stats,add} [path]"""

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypeVar

from pydantic import TypeAdapter, ValidationError

from app.golden.io import GoldenDatasetError, find_issues, load_golden, save_golden
from app.golden.models import (
    Difficulty,
    EmailText,
    GoldenCase,
    GoldenDataset,
    NonEmptyStr,
    TagList,
)
from app.golden.stats import compute_stats
from app.schemas import Category, Summary

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "golden" / "v1.json"
END_OF_EMAIL = "."
TAG_EXAMPLES = "typo, sarcasm, mixed_language, ambiguous, very_short, injection, vague"

InputFn = Callable[[str], str]
T = TypeVar("T")


class Cancelled(Exception):
    pass


# --- validate ---------------------------------------------------------------

def cmd_validate(path: Path) -> int:
    issues = find_issues(path)
    if issues:
        for issue in issues:
            print(issue)
        print(f"\n{len(issues)} problem(s) in {path}", file=sys.stderr)
        return 1
    dataset = load_golden(path)
    print(f"OK: {path} is valid ({len(dataset.cases)} cases)")
    return 0


# --- stats ------------------------------------------------------------------

def _table(title: str, counts: dict[Any, int], total: int) -> None:
    print(f"\n{title}:")
    if not counts:
        print("  (none)")
    width = max((len(str(k)) for k in counts), default=0)
    for key, n in counts.items():
        pct = f"{n / total:6.1%}" if total else "     -"
        print(f"  {str(key):<{width}}  {n:>4}  {pct}")


def cmd_stats(path: Path) -> int:
    dataset = load_golden(path)
    stats = compute_stats(dataset)
    print(f"{path}  (dataset {dataset.dataset_version})")
    print(f"Total cases: {stats.total}")
    _table("By category", {k.value: v for k, v in stats.by_category.items()}, stats.total)
    _table("By difficulty", {k.value: v for k, v in stats.by_difficulty.items()}, stats.total)
    _table("By tag", stats.by_tag, stats.total)
    print(f"  ({stats.untagged} case(s) have no tags)")
    if stats.warnings:
        print("\nWarnings:")
        for warning in stats.warnings:
            print(f"  - {warning}")
    return 0


# --- add --------------------------------------------------------------------

def _ask(input_fn: InputFn, prompt: str) -> str:
    try:
        return input_fn(prompt)
    except (EOFError, KeyboardInterrupt):
        raise Cancelled from None


def _ask_until_valid(input_fn: InputFn, prompt: str, adapter: TypeAdapter[T], transform: Callable[[str], Any] = lambda s: s) -> T:
    while True:
        raw = _ask(input_fn, prompt)
        try:
            return adapter.validate_python(transform(raw))
        except ValidationError as exc:
            print("  invalid: " + "; ".join(err["msg"] for err in exc.errors()))


def _read_email(input_fn: InputFn, existing_inputs: set[str]) -> str:
    while True:
        print(f'Email text (paste it, then a line with only "{END_OF_EMAIL}" to finish):')
        lines: list[str] = []
        while True:
            try:
                line = input_fn("")
            except EOFError:
                break
            except KeyboardInterrupt:
                raise Cancelled from None
            if line == END_OF_EMAIL:
                break
            lines.append(line)
        text = "\n".join(lines).strip("\n")
        try:
            TypeAdapter(EmailText).validate_python(text)
        except ValidationError:
            print("  invalid: email must not be empty")
            if not lines:
                raise Cancelled from None  # EOF with nothing typed
            continue
        if text.strip() in existing_inputs:
            print("  invalid: an existing case already has exactly this input")
            continue
        return text


def _split_tags(raw: str) -> list[str]:
    return [t.strip() for t in raw.split(",") if t.strip()]


def prompt_new_case(dataset: GoldenDataset, input_fn: InputFn = input) -> GoldenCase:
    """Ask for each field, re-asking on invalid values. Raises Cancelled on EOF/Ctrl-C."""
    case_id = dataset.next_case_id()
    print(f"New case: {case_id}")
    email = _read_email(input_fn, {c.input.strip() for c in dataset.cases})
    categories = "/".join(c.value for c in Category)
    category = _ask_until_valid(input_fn, f"Category [{categories}]: ", TypeAdapter(Category), str.strip)
    summary = _ask_until_valid(input_fn, "Ideal summary (one sentence, max 300 chars): ", TypeAdapter(Summary))
    difficulties = "/".join(d.value for d in Difficulty)
    difficulty = _ask_until_valid(input_fn, f"Difficulty [{difficulties}]: ", TypeAdapter(Difficulty), str.strip)
    tags = _ask_until_valid(input_fn, f"Tags, comma-separated, optional (e.g. {TAG_EXAMPLES}): ", TypeAdapter(TagList), _split_tags)
    notes = _ask_until_valid(input_fn, "Notes (why this case matters / why this label): ", TypeAdapter(NonEmptyStr))
    return GoldenCase(
        id=case_id,
        input=email,
        expected_category=category,
        ideal_summary=summary,
        expected_difficulty=difficulty,
        tags=tags,
        notes=notes,
    )


def append_case(dataset: GoldenDataset, case: GoldenCase) -> GoldenDataset:
    """Return a new dataset with `case` appended; raises ValidationError on duplicates."""
    return GoldenDataset.model_validate({**dataset.model_dump(), "cases": [*dataset.cases, case]})


def cmd_add(path: Path, input_fn: InputFn = input) -> int:
    dataset = load_golden(path)  # refuse to append to a file that is already invalid
    try:
        case = prompt_new_case(dataset, input_fn)
        print("\n" + json.dumps(case.model_dump(mode="json"), indent=2, ensure_ascii=False))
        answer = _ask(input_fn, f"Append to {path}? [y/N]: ")
    except Cancelled:
        print("\nCancelled; nothing written.", file=sys.stderr)
        return 1
    if answer.strip().lower() not in {"y", "yes"}:
        print("Not saved.")
        return 1
    save_golden(append_case(dataset, case), path)
    print(f"Added {case.id} ({len(dataset.cases) + 1} cases total).")
    return 0


# --- entry point ------------------------------------------------------------

def main(argv: list[str] | None = None, input_fn: InputFn = input) -> int:
    parser = argparse.ArgumentParser(prog="python -m app.golden", description="Golden dataset tools.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in [
        ("validate", "check the file and print every problem with its case id"),
        ("stats", "counts by category, difficulty and tag, plus balance warnings"),
        ("add", "interactively add one case"),
    ]:
        cmd = sub.add_parser(name, help=help_text)
        cmd.add_argument("path", nargs="?", type=Path, default=DEFAULT_PATH, help=f"dataset file (default: {DEFAULT_PATH.parent.name}/{DEFAULT_PATH.name})")
    args = parser.parse_args(argv)

    if args.command == "validate":
        return cmd_validate(args.path)
    try:
        if args.command == "stats":
            return cmd_stats(args.path)
        return cmd_add(args.path, input_fn)
    except GoldenDatasetError as exc:
        print(exc, file=sys.stderr)
        print("Run `python -m app.golden validate` and fix these first.", file=sys.stderr)
        return 1
