"""Slow-drift check: is the recent moving average lower than the one before it?

A per-run compare looks at two runs. A series of small drops, each under the compare
thresholds, never fails it. Here the mean of the last `window` qualifying runs is
compared with the mean of the `window` qualifying runs before them.

A run qualifies when it is a full run (no --limit) with no errored cases, on the same
dataset file, prompt version, prompt file and classifier model. The prompt file and model
come from the newest such run, so an edited prompt or a model switch starts a new history
instead of showing up as drift.

Everything here is pure: it takes RunSummary records (newest first, as list_runs returns
them) and returns plain data.
"""

import statistics
from dataclasses import dataclass, field
from enum import IntEnum

from app.evals.compare import Delta
from app.evals.config import DriftConfig
from app.evals.models import RunInfo
from app.evals.store import RunSummary


class DriftStatus(IntEnum):
    """Value is the CLI exit code, same convention as compare. NOT ENOUGH HISTORY exits 0."""

    OK = 0
    WARNING = 1
    CRITICAL = 2


LABELS = {DriftStatus.OK: "OK", DriftStatus.WARNING: "DRIFT WARNING", DriftStatus.CRITICAL: "DRIFT CRITICAL"}
NOT_ENOUGH = "NOT ENOUGH HISTORY"


@dataclass(frozen=True)
class Selection:
    """The qualifying runs, oldest first, and what was skipped among the same prompt + dataset."""

    runs: list[RunSummary]
    anchor: RunInfo | None  # newest qualifying run: defines prompt file and model
    skipped_partial: int = 0
    skipped_errors: int = 0
    skipped_other_setup: int = 0  # other prompt file or classifier model


@dataclass(frozen=True)
class Drift:
    prompt_version: str
    dataset_sha256: str
    anchor: RunInfo | None
    window: int
    have: int
    need: int
    window_ids: list[str]
    reference_ids: list[str]
    strict: Delta  # baseline = reference mean, candidate = window mean
    acceptable: Delta
    judge: Delta | None  # None = not enough judge coverage in one of the windows
    status: DriftStatus | None  # None = not enough history
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def enough_history(self) -> bool:
        return self.status is not None

    @property
    def label(self) -> str:
        return NOT_ENOUGH if self.status is None else LABELS[self.status]

    @property
    def exit_code(self) -> int:
        return 0 if self.status is None else int(self.status)


def _has_judge(r: RunSummary, min_coverage: float) -> bool:
    return (
        r.info.judge_enabled
        and r.mean_judge_score is not None
        and r.case_count > 0
        and r.judged / r.case_count >= min_coverage
    )


def select_runs(
    runs: list[RunSummary], prompt_version: str, dataset_sha256: str, as_of: RunInfo | None = None
) -> Selection:
    """Qualifying runs, oldest first. With as_of, runs started after it are ignored."""
    same = [
        r
        for r in runs
        if r.info.prompt_version == prompt_version
        and r.info.dataset_sha256 == dataset_sha256
        and (as_of is None or (r.info.started_at, r.info.run_id) <= (as_of.started_at, as_of.run_id))
    ]
    clean = [r for r in same if not r.info.is_partial and r.errors == 0 and r.case_count > 0]
    if not clean:
        return Selection(runs=[], anchor=None, skipped_partial=sum(r.info.is_partial for r in same),
                         skipped_errors=sum(not r.info.is_partial and r.errors > 0 for r in same))
    clean.sort(key=lambda r: (r.info.started_at, r.info.run_id))
    anchor = clean[-1].info
    picked = [
        r for r in clean
        if r.info.prompt_file_sha256 == anchor.prompt_file_sha256 and r.info.classifier_model == anchor.classifier_model
    ]
    return Selection(
        runs=picked,
        anchor=anchor,
        skipped_partial=sum(r.info.is_partial for r in same),
        skipped_errors=sum(not r.info.is_partial and r.errors > 0 for r in same),
        skipped_other_setup=len(clean) - len(picked),
    )


def _mean(values: list[float]) -> float:
    return statistics.fmean(values)


def compute_drift(
    runs: list[RunSummary],
    prompt_version: str,
    dataset_sha256: str,
    config: DriftConfig | None = None,
    as_of: RunInfo | None = None,
) -> Drift:
    config = config or DriftConfig()
    sel = select_runs(runs, prompt_version, dataset_sha256, as_of)
    w = config.window
    notes = []
    skipped = []
    if sel.skipped_partial:
        skipped.append(f"{sel.skipped_partial} partial")
    if sel.skipped_errors:
        skipped.append(f"{sel.skipped_errors} with errors")
    if sel.skipped_other_setup:
        skipped.append(f"{sel.skipped_other_setup} with another prompt file or classifier model")
    if skipped:
        notes.append("Skipped runs: " + ", ".join(skipped) + ".")

    have = len(sel.runs)
    common = dict(prompt_version=prompt_version, dataset_sha256=dataset_sha256, anchor=sel.anchor,
                  window=w, have=have, need=config.min_history)
    if have < config.min_history:
        return Drift(**common, window_ids=[], reference_ids=[], strict=Delta(None, None),
                     acceptable=Delta(None, None), judge=None, status=None, notes=notes)

    window = sel.runs[-w:]
    reference = sel.runs[-2 * w:-w] if have >= 2 * w else sel.runs[: have - w]
    strict = Delta(_mean([r.strict_accuracy for r in reference]), _mean([r.strict_accuracy for r in window]))
    acceptable = Delta(_mean([r.acceptable_accuracy for r in reference]),
                       _mean([r.acceptable_accuracy for r in window]))

    judge = None
    judged_w = [r for r in window if _has_judge(r, config.min_judge_coverage)]
    judged_r = [r for r in reference if _has_judge(r, config.min_judge_coverage)]
    if len(judged_w) >= config.min_judge_coverage * len(window) and len(judged_r) >= config.min_judge_coverage * len(reference):
        judge = Delta(_mean([r.mean_judge_score for r in judged_r]), _mean([r.mean_judge_score for r in judged_w]))
    else:
        notes.append(
            f"Judge score not shown: judged runs {len(judged_r)}/{len(reference)} (reference), "
            f"{len(judged_w)}/{len(window)} (window); needs {config.min_judge_coverage:.0%} in both."
        )

    status = DriftStatus.OK
    reasons = []
    for name, d in (("strict", strict), ("acceptable", acceptable)):
        drop = round(-d.change * 100, 9)  # guard against float noise at exact thresholds
        if drop >= config.drop_critical_pp:
            level, at = DriftStatus.CRITICAL, config.drop_critical_pp
        elif drop >= config.drop_warning_pp:
            level, at = DriftStatus.WARNING, config.drop_warning_pp
        else:
            continue
        status = max(status, level)
        reasons.append(f"{name} {w}-run mean dropped {drop:.1f} pp ({LABELS[level].lower()} at {at:g})")
    if config.floor_strict_pct is not None and round(strict.candidate * 100, 9) < config.floor_strict_pct:
        status = DriftStatus.CRITICAL
        reasons.append(f"strict {w}-run mean {strict.candidate:.1%} is below the floor {config.floor_strict_pct:g}%")

    return Drift(**common, window_ids=[r.info.run_id for r in window], reference_ids=[r.info.run_id for r in reference],
                 strict=strict, acceptable=acceptable, judge=judge, status=status, reasons=reasons, notes=notes)


def strict_moving_average(
    runs: list[RunSummary], prompt_version: str, dataset_sha256: str, window: int, as_of: RunInfo | None = None
) -> dict[str, float]:
    """run_id -> mean strict accuracy of the `window` qualifying runs ending at that run.
    Only runs with a full trailing window get a value."""
    picked = select_runs(runs, prompt_version, dataset_sha256, as_of).runs
    return {
        picked[i].info.run_id: _mean([r.strict_accuracy for r in picked[i - window + 1: i + 1]])
        for i in range(window - 1, len(picked))
    }


# --- output ------------------------------------------------------------------------------

def _pp(change: float | None) -> str:
    return "-" if change is None else f"{change * 100:+.1f} pp"


def drift_line(d: Drift) -> str:
    """One line for compare, the HTML report and alerts."""
    head = f"Drift ({d.window}-run average): {d.label}"
    if not d.enough_history:
        return f"{head} (have {d.have}, need {d.need})"
    return (
        f"{head} (strict {d.strict.candidate:.1%} vs {d.strict.baseline:.1%} over the previous "
        f"{len(d.reference_ids)} runs, {_pp(d.strict.change)}; acceptable {_pp(d.acceptable.change)})"
    )


def format_drift(d: Drift) -> str:
    lines = [f"STATUS: {d.label}"]
    setup = f"Prompt {d.prompt_version}, dataset sha256 {d.dataset_sha256[:12]}"
    if d.anchor is not None:
        setup += (f" ({d.anchor.dataset_version}); prompt file sha256 {d.anchor.prompt_file_sha256[:12]}, "
                  f"model {d.anchor.classifier_model}")
    lines.append(setup)
    if not d.enough_history:
        lines.append(f"Not enough history (have {d.have}, need {d.need} qualifying full runs without errors).")
        lines += d.notes
        return "\n".join(lines)
    for r in d.reasons:
        lines.append(f"  reason: {r}")
    lines += [
        "",
        f"{'':<22}{'reference':>10} {'window':>10} {'delta':>10}",
        f"{'Strict accuracy':<22}{d.strict.baseline:>10.1%} {d.strict.candidate:>10.1%} {_pp(d.strict.change):>10}",
        f"{'Acceptable accuracy':<22}{d.acceptable.baseline:>10.1%} {d.acceptable.candidate:>10.1%} {_pp(d.acceptable.change):>10}",
    ]
    if d.judge is not None:
        lines.append(f"{'Mean judge score':<22}{d.judge.baseline:>10.2f} {d.judge.candidate:>10.2f} "
                     f"{d.judge.change:>+10.2f}  (shown only, not used for status)")
    lines += [
        "",
        f"Window ({len(d.window_ids)} runs, newest last): " + ", ".join(d.window_ids),
        f"Reference ({len(d.reference_ids)} runs): " + ", ".join(d.reference_ids),
    ]
    lines += d.notes
    return "\n".join(lines)
