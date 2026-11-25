"""Aggregate metrics for one run, computed from its case results."""

import math
from dataclasses import dataclass

from app.evals.models import CaseResult, RunInfo
from app.golden.models import Difficulty
from app.schemas import Category


@dataclass(frozen=True)
class GroupAccuracy:
    total: int
    strict_correct: int
    acceptable_correct: int

    @property
    def strict(self) -> float | None:
        return self.strict_correct / self.total if self.total else None

    @property
    def acceptable(self) -> float | None:
        return self.acceptable_correct / self.total if self.total else None


@dataclass(frozen=True)
class Scorecard:
    overall: GroupAccuracy
    by_category: dict[Category, GroupAccuracy]
    by_difficulty: dict[Difficulty, GroupAccuracy]
    mean_judge_score: float | None
    judged: int
    latency_p50_ms: float | None
    latency_p95_ms: float | None
    input_tokens: int
    output_tokens: int
    errors: int
    failures: list[CaseResult]  # strict misses and errors, dataset order


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank percentile: the smallest value with at least p% of values at or below it."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(1, math.ceil(p / 100 * len(ordered)))
    return ordered[rank - 1]


def _group(cases: list[CaseResult]) -> GroupAccuracy:
    return GroupAccuracy(
        total=len(cases),
        strict_correct=sum(c.strict_match for c in cases),
        acceptable_correct=sum(c.acceptable_match for c in cases),
    )


def compute_scorecard(cases: list[CaseResult]) -> Scorecard:
    scores = [c.summary_score for c in cases if c.summary_score is not None]
    latencies = [c.latency_ms for c in cases if c.latency_ms is not None]
    return Scorecard(
        overall=_group(cases),
        by_category={cat: _group([c for c in cases if c.expected_category == cat]) for cat in Category},
        by_difficulty={d: _group([c for c in cases if c.expected_difficulty == d]) for d in Difficulty},
        mean_judge_score=sum(scores) / len(scores) if scores else None,
        judged=len(scores),
        latency_p50_ms=percentile(latencies, 50),
        latency_p95_ms=percentile(latencies, 95),
        input_tokens=sum(c.input_tokens or 0 for c in cases),
        output_tokens=sum(c.output_tokens or 0 for c in cases),
        errors=sum(1 for c in cases if c.error is not None),
        failures=[c for c in cases if not c.strict_match],
    )


def _pct(value: float | None) -> str:
    return "    -" if value is None else f"{value:6.1%}"


def _ms(value: float | None) -> str:
    return "-" if value is None else f"{value:,.0f} ms"


def format_scorecard(info: RunInfo, card: Scorecard) -> str:
    lines = [f"Run {info.run_id}  ({info.started_at:%Y-%m-%d %H:%M:%S %Z})"]
    if info.is_partial:
        lines.append(f"PARTIAL RUN: first {info.case_limit} cases only; not comparable to a full run")
    lines += [
        f"Prompt {info.prompt_version} on {info.classifier_model} | judge {info.judge_model} "
        f"| dataset {info.dataset_version} | concurrency {info.concurrency}",
    ]
    if info.notes:
        lines.append(f"Notes: {info.notes}")
    o = card.overall
    lines += [
        "",
        f"Strict accuracy:      {_pct(o.strict)}  ({o.strict_correct}/{o.total})",
        f"Acceptable accuracy:  {_pct(o.acceptable)}  ({o.acceptable_correct}/{o.total})",
        "Mean judge score:     "
        + ("-" if card.mean_judge_score is None else f"{card.mean_judge_score:.2f} / 5")
        + f"  ({card.judged}/{o.total} scored)",
        f"Latency p50 / p95:    {_ms(card.latency_p50_ms)} / {_ms(card.latency_p95_ms)}",
        f"Tokens in / out:      {card.input_tokens:,} / {card.output_tokens:,}",
        f"Errors:               {card.errors}",
        "",
        f"{'By expected category':<22}{'n':>4}  {'strict':>6}  {'accept':>6}",
    ]
    for cat, g in card.by_category.items():
        lines.append(f"  {cat.value:<20}{g.total:>4}  {_pct(g.strict)}  {_pct(g.acceptable)}")
    lines.append(f"{'By difficulty':<22}{'n':>4}  {'strict':>6}  {'accept':>6}")
    for diff, g in card.by_difficulty.items():
        lines.append(f"  {diff.value:<20}{g.total:>4}  {_pct(g.strict)}  {_pct(g.acceptable)}")

    lines += ["", f"Failing cases ({len(card.failures)}, strict):"]
    if not card.failures:
        lines.append("  (none)")
    for c in card.failures:
        if c.error is not None:
            lines.append(f"  {c.case_id}  expected {c.expected_category.value:<9}  ERROR: {c.error[:120]}")
        else:
            note = "  (in acceptable_categories)" if c.acceptable_match else ""
            lines.append(
                f"  {c.case_id}  expected {c.expected_category.value:<9}  predicted {c.predicted_category.value}{note}"
            )
    return "\n".join(lines)
