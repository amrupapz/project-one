"""Compare two stored runs: deltas, flipped cases, warnings and a PASS/WARNING/CRITICAL status.

Everything here is pure: it takes EvalRun records and returns plain data.
Metrics are computed on the case ids both runs share, so a partial run is compared
like for like (and flagged with a warning).
"""

from dataclasses import dataclass, field
from enum import IntEnum

from app.evals.config import CompareConfig
from app.evals.models import CaseResult, EvalRun, RunInfo
from app.evals.scorecard import Scorecard, compute_scorecard
from app.golden.models import Difficulty
from app.schemas import Category


class Status(IntEnum):
    """Value is the CLI exit code."""

    PASS = 0
    WARNING = 1
    CRITICAL = 2


class CompareRefused(ValueError):
    """The two runs cannot be compared meaningfully."""


@dataclass(frozen=True)
class Delta:
    baseline: float | None
    candidate: float | None

    @property
    def change(self) -> float | None:
        if self.baseline is None or self.candidate is None:
            return None
        return self.candidate - self.baseline


@dataclass(frozen=True)
class GroupDelta:
    n: int
    strict: Delta
    acceptable: Delta


@dataclass(frozen=True)
class Flip:
    """A case whose pass/fail outcome differs between the runs."""

    baseline: CaseResult
    candidate: CaseResult

    @property
    def case_id(self) -> str:
        return self.candidate.case_id

    @property
    def input(self) -> str:
        return self.candidate.input

    @property
    def expected_category(self) -> Category:
        return self.candidate.expected_category

    @property
    def acceptable_categories(self) -> list[Category]:
        return self.candidate.acceptable_categories

    @property
    def ideal_summary(self) -> str:
        return self.candidate.ideal_summary


@dataclass(frozen=True)
class JudgeCoverage:
    enabled: bool
    scored: int
    total: int

    @property
    def fraction(self) -> float:
        return self.scored / self.total if self.total else 0.0


@dataclass(frozen=True)
class Comparison:
    baseline: RunInfo
    candidate: RunInfo
    shared_cases: int
    baseline_only: list[str]
    candidate_only: list[str]
    baseline_card: Scorecard
    candidate_card: Scorecard
    strict: Delta
    acceptable: Delta
    strict_drop_pp: float  # baseline minus candidate, in percentage points (positive = worse)
    by_category: dict[Category, GroupDelta]
    by_difficulty: dict[Difficulty, GroupDelta]
    judge_score: Delta
    judge_coverage: tuple[JudgeCoverage, JudgeCoverage]  # (baseline, candidate)
    judge_used_in_status: bool
    latency_p50_ms: Delta
    latency_p95_ms: Delta
    input_tokens: Delta
    output_tokens: Delta
    strict_regressions: list[Flip]
    strict_improvements: list[Flip]
    acceptable_regressions: list[Flip]
    acceptable_improvements: list[Flip]
    status: Status
    status_reasons: list[str]
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def points_per_case(self) -> float:
        return 100 / self.shared_cases

    @property
    def headline(self) -> str:
        """Status with the strict delta and flip counts, so a PASS with regressions is never bare."""
        return (
            f"{self.status.name} (strict {-self.strict_drop_pp:+.1f} pp; "
            f"{_plural(len(self.strict_regressions), 'regression')}, "
            f"{_plural(len(self.strict_improvements), 'improvement')})"
        )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def _flips(pairs: list[tuple[CaseResult, CaseResult]], attr: str) -> tuple[list[Flip], list[Flip]]:
    regressions = [Flip(b, c) for b, c in pairs if getattr(b, attr) and not getattr(c, attr)]
    improvements = [Flip(b, c) for b, c in pairs if not getattr(b, attr) and getattr(c, attr)]
    return regressions, improvements


def _coverage(info: RunInfo, card: Scorecard) -> JudgeCoverage:
    return JudgeCoverage(enabled=info.judge_enabled, scored=card.judged, total=card.overall.total)


def compute_status(
    strict_drop_pp: float, judge_drop: float | None, config: CompareConfig
) -> tuple[Status, list[str]]:
    """Strict accuracy drives the status. A judge-score drop (only when judge_drop is
    given, i.e. usable in both runs) can raise PASS to WARNING, never to CRITICAL."""
    drop = round(strict_drop_pp, 9)  # guard against float noise at exact threshold values
    reasons: list[str] = []
    if drop >= config.strict_drop_critical_pp:
        status = Status.CRITICAL
        reasons.append(f"strict accuracy dropped {drop:.1f} pp (critical at {config.strict_drop_critical_pp:g})")
    elif drop >= config.strict_drop_warning_pp:
        status = Status.WARNING
        reasons.append(f"strict accuracy dropped {drop:.1f} pp (warning at {config.strict_drop_warning_pp:g})")
    else:
        status = Status.PASS
    if judge_drop is not None and round(judge_drop, 9) >= config.judge_score_drop_warning:
        reasons.append(
            f"mean judge score dropped {judge_drop:.2f} (warning at {config.judge_score_drop_warning:g})"
        )
        status = max(status, Status.WARNING)
    return status, reasons


def compare_runs(baseline: EvalRun, candidate: EvalRun, config: CompareConfig | None = None) -> Comparison:
    config = config or CompareConfig()
    b_info, c_info = baseline.info, candidate.info
    if b_info.run_id == c_info.run_id:
        raise CompareRefused(f"baseline and candidate are the same run ({b_info.run_id})")
    if b_info.dataset_sha256 != c_info.dataset_sha256:
        raise CompareRefused(
            f"refusing to compare: the runs used different dataset files "
            f"(baseline {b_info.dataset_version} sha256 {b_info.dataset_sha256[:12]}, "
            f"candidate {c_info.dataset_version} sha256 {c_info.dataset_sha256[:12]}). "
            "Accuracy on different cases or labels is not comparable."
        )

    b_by_id = {c.case_id: c for c in baseline.cases}
    c_by_id = {c.case_id: c for c in candidate.cases}
    shared_ids = [c.case_id for c in candidate.cases if c.case_id in b_by_id]  # candidate (dataset) order
    if not shared_ids:
        raise CompareRefused("refusing to compare: the runs share no case ids")
    pairs = [(b_by_id[i], c_by_id[i]) for i in shared_ids]
    b_cases = [b for b, _ in pairs]
    c_cases = [c for _, c in pairs]
    b_card, c_card = compute_scorecard(b_cases), compute_scorecard(c_cases)
    n = len(pairs)

    warnings: list[str] = []
    notes: list[str] = []
    for role, info in (("baseline", b_info), ("candidate", c_info)):
        if info.is_partial:
            warnings.append(
                f"PARTIAL RUN: {role} {info.run_id} ran only the first {info.case_limit} cases; "
                f"metrics cover the {n} cases both runs share"
            )
    if b_info.eval_config_sha256 != c_info.eval_config_sha256:
        warnings.append(
            "eval_config.yaml differs between the runs (judge model, rubric or settings changed); "
            "judge scores may not be comparable"
        )

    # Judge: usable for the status only if both runs were judged with enough coverage.
    coverage = (_coverage(b_info, b_card), _coverage(c_info, c_card))
    judge_usable = True
    if not b_info.judge_enabled and not c_info.judge_enabled:
        judge_usable = False
        notes.append("Both runs were --no-judge runs; no judge-score comparison.")
    elif not b_info.judge_enabled:
        judge_usable = False
        warnings.append(
            f"baseline {b_info.run_id} is a --no-judge run and must not be used as a baseline for a "
            "judged run; judge-score delta is not computed"
        )
    elif not c_info.judge_enabled:
        judge_usable = False
        warnings.append(
            f"candidate {c_info.run_id} is a --no-judge run but the baseline was judged; "
            "judge-score delta is not computed"
        )
    else:
        for role, cov in zip(("baseline", "candidate"), coverage):
            if cov.fraction < config.min_judge_coverage:
                judge_usable = False
                warnings.append(
                    f"{role} judge coverage is only {cov.scored}/{cov.total}; judge-score delta is unreliable "
                    "and not used for the status"
                )

    judge_score = Delta(b_card.mean_judge_score, c_card.mean_judge_score)
    if not (b_info.judge_enabled and c_info.judge_enabled):
        judge_score = Delta(
            b_card.mean_judge_score if b_info.judge_enabled else None,
            c_card.mean_judge_score if c_info.judge_enabled else None,
        )

    strict_drop_pp = (b_card.overall.strict_correct - c_card.overall.strict_correct) * 100 / n
    judge_drop = -judge_score.change if judge_usable and judge_score.change is not None else None
    status, reasons = compute_status(strict_drop_pp, judge_drop, config)

    strict_reg, strict_imp = _flips(pairs, "strict_match")
    acc_reg, acc_imp = _flips(pairs, "acceptable_match")

    def group_delta(b, c) -> GroupDelta:
        return GroupDelta(n=c.total, strict=Delta(b.strict, c.strict), acceptable=Delta(b.acceptable, c.acceptable))

    return Comparison(
        baseline=b_info,
        candidate=c_info,
        shared_cases=n,
        baseline_only=[i for i in b_by_id if i not in c_by_id],
        candidate_only=[i for i in c_by_id if i not in b_by_id],
        baseline_card=b_card,
        candidate_card=c_card,
        strict=Delta(b_card.overall.strict, c_card.overall.strict),
        acceptable=Delta(b_card.overall.acceptable, c_card.overall.acceptable),
        strict_drop_pp=strict_drop_pp,
        by_category={k: group_delta(b_card.by_category[k], c_card.by_category[k]) for k in Category},
        by_difficulty={k: group_delta(b_card.by_difficulty[k], c_card.by_difficulty[k]) for k in Difficulty},
        judge_score=judge_score,
        judge_coverage=coverage,
        judge_used_in_status=judge_usable,
        latency_p50_ms=Delta(b_card.latency_p50_ms, c_card.latency_p50_ms),
        latency_p95_ms=Delta(b_card.latency_p95_ms, c_card.latency_p95_ms),
        input_tokens=Delta(b_card.input_tokens, c_card.input_tokens),
        output_tokens=Delta(b_card.output_tokens, c_card.output_tokens),
        strict_regressions=strict_reg,
        strict_improvements=strict_imp,
        acceptable_regressions=acc_reg,
        acceptable_improvements=acc_imp,
        status=status,
        status_reasons=reasons,
        warnings=warnings,
        notes=notes,
    )


# --- terminal output ---------------------------------------------------------------------

def _pct(v: float | None) -> str:
    return "-" if v is None else f"{v:.1%}"


def _pp(d: Delta) -> str:
    return "-" if d.change is None else f"{d.change * 100:+.1f} pp"


def _num(v: float | None, fmt: str) -> str:
    return "-" if v is None else format(v, fmt)


def _signed(v: float | None, fmt: str) -> str:
    return "-" if v is None else ("+" if v >= 0 else "") + format(v, fmt)


def _outcome(c: CaseResult) -> str:
    if c.error is not None:
        return "ERROR"
    return c.predicted_category.value if c.predicted_category else "-"


def format_comparison(cmp: Comparison) -> str:
    b, c = cmp.baseline, cmp.candidate
    lines = [
        f"STATUS: {cmp.headline}",
        f"Baseline  {b.run_id}  prompt {b.prompt_version} on {b.classifier_model}"
        + ("" if b.judge_enabled else "  [NO JUDGE]"),
        f"Candidate {c.run_id}  prompt {c.prompt_version} on {c.classifier_model}"
        + ("" if c.judge_enabled else "  [NO JUDGE]"),
    ]
    for w in cmp.warnings:
        lines.append(f"WARNING: {w}")
    for r in cmp.status_reasons:
        lines.append(f"  reason: {r}")
    lines.append("")

    judge_row = (
        f"{_num(cmp.judge_score.baseline, '.2f'):>10} {_num(cmp.judge_score.candidate, '.2f'):>10} "
        f"{_signed(cmp.judge_score.change, '.2f'):>10}"
    )
    if not cmp.judge_used_in_status:
        judge_row += "  (not used for status)"
    lines += [
        f"{'':<22}{'baseline':>10} {'candidate':>10} {'delta':>10}",
        f"{'Strict accuracy':<22}{_pct(cmp.strict.baseline):>10} {_pct(cmp.strict.candidate):>10} {_pp(cmp.strict):>10}",
        f"{'Acceptable accuracy':<22}{_pct(cmp.acceptable.baseline):>10} {_pct(cmp.acceptable.candidate):>10} {_pp(cmp.acceptable):>10}",
        f"{'Mean judge score':<22}{judge_row}",
        f"{'Latency p50 (ms)':<22}{_num(cmp.latency_p50_ms.baseline, ',.0f'):>10} {_num(cmp.latency_p50_ms.candidate, ',.0f'):>10} {_signed(cmp.latency_p50_ms.change, ',.0f'):>10}",
        f"{'Latency p95 (ms)':<22}{_num(cmp.latency_p95_ms.baseline, ',.0f'):>10} {_num(cmp.latency_p95_ms.candidate, ',.0f'):>10} {_signed(cmp.latency_p95_ms.change, ',.0f'):>10}",
        f"{'Tokens in':<22}{_num(cmp.input_tokens.baseline, ',.0f'):>10} {_num(cmp.input_tokens.candidate, ',.0f'):>10} {_signed(cmp.input_tokens.change, ',.0f'):>10}",
        f"{'Tokens out':<22}{_num(cmp.output_tokens.baseline, ',.0f'):>10} {_num(cmp.output_tokens.candidate, ',.0f'):>10} {_signed(cmp.output_tokens.change, ',.0f'):>10}",
        "",
        f"{'Strict by category':<22}{'n':>4}  {'base':>6} {'cand':>6} {'delta':>9}",
    ]
    for key, g in [*cmp.by_category.items(), (None, None), *cmp.by_difficulty.items()]:
        if key is None:
            lines.append(f"{'Strict by difficulty':<22}{'n':>4}  {'base':>6} {'cand':>6} {'delta':>9}")
            continue
        lines.append(
            f"  {key.value:<20}{g.n:>4}  {_pct(g.strict.baseline):>6} {_pct(g.strict.candidate):>6} {_pp(g.strict):>9}"
        )

    lines += [
        "",
        f"Flipped cases: strict {len(cmp.strict_regressions)} regressed / {len(cmp.strict_improvements)} improved; "
        f"acceptable {len(cmp.acceptable_regressions)} regressed / {len(cmp.acceptable_improvements)} improved.",
        f"Note: with {cmp.shared_cases} cases one case is about {cmp.points_per_case:.1f} pp, "
        "so small differences can be noise (see `repeat`).",
    ]
    for title, flips in (
        ("Strict regressions", cmp.strict_regressions),
        ("Strict improvements", cmp.strict_improvements),
        ("Acceptable regressions", cmp.acceptable_regressions),
        ("Acceptable improvements", cmp.acceptable_improvements),
    ):
        if not flips:
            continue
        lines.append(f"{title}:")
        for f in flips:
            lines.append(
                f"  {f.case_id}  expected {f.expected_category.value:<9}  "
                f"{_outcome(f.baseline):>9} -> {_outcome(f.candidate)}"
            )
    for note in cmp.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)
