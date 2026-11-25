"""Coverage counts and balance warnings for a golden dataset."""

from collections import Counter
from dataclasses import dataclass, field

from app.golden.models import Difficulty, GoldenDataset
from app.schemas import Category

MIN_CASES_PER_CATEGORY = 8
MIN_HARD_FRACTION = 0.15
MAX_ACCEPTABLE_FRACTION = 0.25


@dataclass(frozen=True)
class GoldenStats:
    total: int
    by_category: dict[Category, int]
    by_difficulty: dict[Difficulty, int]
    by_tag: dict[str, int]  # most common first
    untagged: int
    with_acceptable_categories: int
    warnings: list[str] = field(default_factory=list)


def compute_stats(dataset: GoldenDataset) -> GoldenStats:
    cases = dataset.cases
    total = len(cases)
    by_category = {c: 0 for c in Category} | Counter(case.expected_category for case in cases)
    by_difficulty = {d: 0 for d in Difficulty} | Counter(case.expected_difficulty for case in cases)
    tag_counts = Counter(tag for case in cases for tag in case.tags)
    by_tag = dict(sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])))

    warnings = [
        f"category {cat.value!r} has {n} case(s); aim for at least {MIN_CASES_PER_CATEGORY}"
        for cat, n in by_category.items()
        if n < MIN_CASES_PER_CATEGORY
    ]
    if total:
        hard_fraction = by_difficulty[Difficulty.HARD] / total
        if hard_fraction < MIN_HARD_FRACTION:
            warnings.append(
                f"hard cases are {hard_fraction:.1%} of the total; aim for at least {MIN_HARD_FRACTION:.0%}"
            )

    with_acceptable = sum(1 for case in cases if case.acceptable_categories)
    if total and with_acceptable / total > MAX_ACCEPTABLE_FRACTION:
        warnings.append(
            f"{with_acceptable / total:.1%} of cases have acceptable_categories; keep it at or under "
            f"{MAX_ACCEPTABLE_FRACTION:.0%} or the category labels lose meaning"
        )

    return GoldenStats(
        total=total,
        by_category=by_category,
        by_difficulty=by_difficulty,
        by_tag=by_tag,
        untagged=sum(1 for case in cases if not case.tags),
        with_acceptable_categories=with_acceptable,
        warnings=warnings,
    )
