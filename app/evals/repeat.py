"""Noise check: run the same prompt N times and measure how much strict accuracy moves."""

import statistics
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from app.evals.models import CaseResult, EvalRun


@dataclass(frozen=True)
class UnstableCase:
    case_id: str
    expected: str
    outcomes: list[str]  # one per repetition: predicted category or "ERROR"
    strict_passes: int


@dataclass(frozen=True)
class RepeatSpread:
    accuracies: list[float]  # strict accuracy per repetition
    case_count: int
    unstable: list[UnstableCase]  # cases whose outcome differs between repetitions, dataset order

    @property
    def mean(self) -> float:
        return statistics.fmean(self.accuracies)

    @property
    def stdev(self) -> float:
        return statistics.stdev(self.accuracies) if len(self.accuracies) > 1 else 0.0

    @property
    def spread_pp(self) -> float:
        return (max(self.accuracies) - min(self.accuracies)) * 100

    @property
    def strict_flippers(self) -> list[UnstableCase]:
        """Cases that passed strict in some repetitions and failed in others."""
        return [u for u in self.unstable if 0 < u.strict_passes < len(self.accuracies)]


def _outcome(c: CaseResult) -> str:
    if c.error is not None:
        return "ERROR"
    return c.predicted_category.value if c.predicted_category else "-"


def repeat_spread(runs: list[EvalRun]) -> RepeatSpread:
    """Pure: takes the N runs of one prompt over the same cases."""
    if not runs:
        raise ValueError("need at least one run")
    ids = [c.case_id for c in runs[0].cases]
    if any([c.case_id for c in r.cases] != ids for r in runs):
        raise ValueError("all repetitions must cover the same cases in the same order")
    accuracies = [sum(c.strict_match for c in r.cases) / len(r.cases) if r.cases else 0.0 for r in runs]
    unstable = []
    for i, case_id in enumerate(ids):
        results = [r.cases[i] for r in runs]
        outcomes = [_outcome(c) for c in results]
        if len(set(outcomes)) > 1:
            unstable.append(
                UnstableCase(
                    case_id=case_id,
                    expected=results[0].expected_category.value,
                    outcomes=outcomes,
                    strict_passes=sum(c.strict_match for c in results),
                )
            )
    return RepeatSpread(accuracies=accuracies, case_count=len(ids), unstable=unstable)


def format_spread(spread: RepeatSpread, prompt_version: str, saved: bool = False) -> str:
    n = len(spread.accuracies)
    lines = [
        f"Repeat: prompt {prompt_version}, {n} runs x {spread.case_count} cases (no judge, " + ("saved" if saved else "not saved") + ")",
        "Strict accuracy per run: " + ", ".join(f"{a:.1%}" for a in spread.accuracies),
        f"Mean {spread.mean:.1%}  stdev {spread.stdev * 100:.2f} pp  "
        f"min {min(spread.accuracies):.1%}  max {max(spread.accuracies):.1%}  spread {spread.spread_pp:.1f} pp",
        f"One case is {100 / spread.case_count:.1f} pp. A compare delta inside this spread is likely noise."
        if spread.case_count
        else "",
        "",
        f"Cases whose prediction changed between runs ({len(spread.unstable)}; "
        f"{len(spread.strict_flippers)} flip strict pass/fail):",
    ]
    if not spread.unstable:
        lines.append("  (none: every case got the same prediction in every run)")
    for u in spread.unstable:
        lines.append(
            f"  {u.case_id}  expected {u.expected:<9}  {' '.join(f'{o:<9}' for o in u.outcomes).rstrip()}  "
            f"(strict {u.strict_passes}/{n})"
        )
    return "\n".join(lines)


async def repeat_runs(make_run: Callable[[int], Awaitable[EvalRun]], n: int) -> list[EvalRun]:
    """Run sequentially so repetitions don't compete for rate limits."""
    if n < 2:
        raise ValueError("n must be >= 2 to measure spread")
    return [await make_run(i) for i in range(n)]
