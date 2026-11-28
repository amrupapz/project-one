"""Slow-drift detection: window math, run selection, thresholds, CLI, report, alerts. No API calls."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from app.evals import EvalRun, save_run
from app.evals.compare import Status, compare_runs
from app.evals.config import DriftConfig, EvalConfig
from app.evals.drift import DriftStatus, compute_drift, drift_line, format_drift, strict_moving_average
from app.evals.store import list_runs
from tests.test_evals import make_case, make_info

T0 = datetime(2026, 11, 1, 12, 0, tzinfo=UTC)
DATASET = "b" * 64  # make_info's default dataset_sha256


def make_run(i: int, correct: int, total: int = 200, *, accept_extra: int = 0, judged: bool = False,
             errors: int = 0, **info_overrides: Any) -> EvalRun:
    """Run number i (one hour apart). `correct` cases strict-correct; `accept_extra` of the
    wrong ones are still acceptable; `errors` of the wrong ones errored."""
    run_info = make_info(f"202611{1 + i // 24:02d}-{i % 24:02d}0000-r{i:03d}", started_at=T0 + timedelta(hours=i),
                         judge_enabled=judged, **info_overrides)
    cases = []
    for k in range(total):
        cid = f"case_{k + 1:03d}"
        if k < correct:
            cases.append(make_case(cid, "billing", "billing"))
        elif k < correct + errors:
            cases.append(make_case(cid, "billing", None))
        elif k < correct + errors + accept_extra:
            cases.append(make_case(cid, "billing", "account", acceptable_categories=["account"]))
        else:
            cases.append(make_case(cid, "billing", "account"))
    if not judged:
        cases = [c.model_copy(update={"summary_score": None, "judge_reason": None}) for c in cases]
    return EvalRun(info=run_info, cases=cases)


def store(tmp_path: Path, runs: list[EvalRun]):
    db = tmp_path / "eval.db"
    for r in runs:
        save_run(db, r)
    return list_runs(db)


def corrects_to_runs(corrects: list[int], **kw: Any) -> list[EvalRun]:
    return [make_run(i, c, **kw) for i, c in enumerate(corrects)]


# --- window math ---------------------------------------------------------------------------

def test_window_and_reference_means(tmp_path):
    # 200 cases: 1 case = 0.5 pp. Reference 7 runs at 180 (90%), window 7 runs at 178 (89%).
    runs = store(tmp_path, corrects_to_runs([180] * 7 + [178] * 7))
    d = compute_drift(runs, "v1", DATASET)
    assert d.strict.baseline == pytest.approx(0.90) and d.strict.candidate == pytest.approx(0.89)
    assert d.strict.change == pytest.approx(-0.01)
    assert d.status is DriftStatus.OK
    assert d.window_ids == [r.info.run_id for r in reversed(runs)][7:]
    assert d.reference_ids == [r.info.run_id for r in reversed(runs)][:7]


def test_only_last_two_windows_count(tmp_path):
    # Old, much better runs before the reference window do not affect the result.
    runs = store(tmp_path, corrects_to_runs([200] * 5 + [180] * 14))
    d = compute_drift(runs, "v1", DATASET)
    assert d.have == 19 and d.strict.change == pytest.approx(0.0) and d.status is DriftStatus.OK
    assert len(d.reference_ids) == len(d.window_ids) == 7


def test_custom_window(tmp_path):
    runs = store(tmp_path, corrects_to_runs([180] * 3 + [170] * 3))
    d = compute_drift(runs, "v1", DATASET, DriftConfig(window=3, min_history=6))
    assert len(d.window_ids) == 3 and d.strict.change == pytest.approx(-0.05)
    assert d.status is DriftStatus.CRITICAL


def test_strict_moving_average(tmp_path):
    runs = store(tmp_path, corrects_to_runs([180, 182, 184, 186]))
    ma = strict_moving_average(runs, "v1", DATASET, window=3)
    ids = [r.info.run_id for r in reversed(runs)]
    assert list(ma) == ids[2:]
    assert ma[ids[2]] == pytest.approx(0.91) and ma[ids[3]] == pytest.approx(0.92)


# --- not enough history ----------------------------------------------------------------------

@pytest.mark.parametrize("n", [0, 1, 7, 8, 13])
def test_not_enough_history_never_warns(tmp_path, n):
    runs = store(tmp_path, corrects_to_runs([200] * (n // 2) + [100] * (n - n // 2))) if n else []
    d = compute_drift(runs, "v1", DATASET)
    assert d.status is None and not d.enough_history and d.exit_code == 0
    assert d.label == "NOT ENOUGH HISTORY"
    assert drift_line(d) == f"Drift (7-run average): NOT ENOUGH HISTORY (have {n}, need 14)"
    assert f"have {n}, need 14" in format_drift(d)


def test_exactly_min_history_is_enough(tmp_path):
    runs = store(tmp_path, corrects_to_runs([180] * 14))
    assert compute_drift(runs, "v1", DATASET).status is DriftStatus.OK


# --- which runs count ------------------------------------------------------------------------

def test_other_prompt_versions_and_datasets_are_excluded(tmp_path):
    good = corrects_to_runs([180] * 14)
    # Interleave terrible runs of another prompt version and another dataset.
    noise = [make_run(100 + i, 20, prompt_version="v2") for i in range(7)]
    noise += [make_run(200 + i, 20, dataset_sha256="d" * 64) for i in range(7)]
    runs = store(tmp_path, good + noise)
    d = compute_drift(runs, "v1", DATASET)
    assert d.have == 14 and d.strict.change == pytest.approx(0.0) and d.status is DriftStatus.OK
    assert compute_drift(runs, "v2", DATASET).have == 7
    assert compute_drift(runs, "v1", "d" * 64).have == 7


def test_partial_and_errored_runs_are_excluded(tmp_path):
    good = corrects_to_runs([180] * 14)
    bad = [make_run(50 + i, 20, case_limit=200) for i in range(5)]  # partial
    bad += [make_run(80 + i, 150, errors=1) for i in range(3)]  # one errored case
    runs = store(tmp_path, good + bad)
    d = compute_drift(runs, "v1", DATASET)
    assert d.have == 14 and d.status is DriftStatus.OK
    assert "Skipped runs: 5 partial, 3 with errors." in d.notes


def test_other_prompt_file_or_model_starts_a_new_history(tmp_path):
    old_file = corrects_to_runs([180] * 14)
    new_file = [make_run(20 + i, 100, prompt_file_sha256="e" * 64) for i in range(3)]
    other_model = [make_run(40 + i, 100, classifier_model="openai/gpt-4o") for i in range(1)]
    runs = store(tmp_path, old_file + new_file)
    d = compute_drift(runs, "v1", DATASET)
    assert d.status is None and d.have == 3  # the newest prompt file defines the history
    assert "14 with another prompt file or classifier model" in d.notes[0]

    runs = store(tmp_path / "b", old_file + other_model)
    assert compute_drift(runs, "v1", DATASET).have == 1


def test_as_of_ignores_newer_runs(tmp_path):
    evals = corrects_to_runs([180] * 14 + [100] * 7)
    runs = store(tmp_path, evals)
    assert compute_drift(runs, "v1", DATASET).status is DriftStatus.CRITICAL
    assert compute_drift(runs, "v1", DATASET, as_of=evals[13].info).status is DriftStatus.OK


# --- thresholds ------------------------------------------------------------------------------

# 1 case = 0.5 pp. drop_cases cases fewer in every window run.
@pytest.mark.parametrize(("drop_cases", "expected"), [
    (0, DriftStatus.OK), (3, DriftStatus.OK), (4, DriftStatus.WARNING),  # 1.5 / 2.0 pp
    (7, DriftStatus.WARNING), (8, DriftStatus.CRITICAL), (40, DriftStatus.CRITICAL),  # 3.5 / 4.0 / 20 pp
])
def test_strict_thresholds(tmp_path, drop_cases, expected):
    runs = store(tmp_path, corrects_to_runs([180] * 7 + [180 - drop_cases] * 7))
    d = compute_drift(runs, "v1", DATASET)
    assert d.status is expected and d.exit_code == int(expected)
    if expected is not DriftStatus.OK:
        assert d.reasons[0].startswith(f"strict 7-run mean dropped {drop_cases * 0.5:.1f} pp")


def test_acceptable_drop_alone_warns(tmp_path):
    # Strict stays at 170; acceptable drops from 180 to 175 (2.5 pp).
    runs = store(tmp_path, [make_run(i, 170, accept_extra=10) for i in range(7)]
                 + [make_run(7 + i, 170, accept_extra=5) for i in range(7)])
    d = compute_drift(runs, "v1", DATASET)
    assert d.strict.change == pytest.approx(0.0)
    assert d.status is DriftStatus.WARNING and d.reasons == ["acceptable 7-run mean dropped 2.5 pp (drift warning at 2)"]


def test_improvement_is_ok(tmp_path):
    runs = store(tmp_path, corrects_to_runs([150] * 7 + [190] * 7))
    assert compute_drift(runs, "v1", DATASET).status is DriftStatus.OK


def test_floor(tmp_path):
    runs = store(tmp_path, corrects_to_runs([170] * 14))  # flat at 85%
    assert compute_drift(runs, "v1", DATASET).status is DriftStatus.OK  # floor off by default
    assert compute_drift(runs, "v1", DATASET, DriftConfig(floor_strict_pct=85.0)).status is DriftStatus.OK
    d = compute_drift(runs, "v1", DATASET, DriftConfig(floor_strict_pct=85.5))
    assert d.status is DriftStatus.CRITICAL and "below the floor 85.5%" in d.reasons[0]


def test_judge_shown_only_with_coverage_and_never_sets_status(tmp_path):
    judged = [make_run(i, 180, judged=True) for i in range(14)]
    runs = store(tmp_path, judged)
    d = compute_drift(runs, "v1", DATASET)
    assert d.judge is not None and d.judge.change == pytest.approx(0.0)
    assert "Mean judge score" in format_drift(d) and "not used for status" in format_drift(d)

    # One unjudged run in the window: 6/7 < 95%, so the judge is not shown.
    mixed = judged[:13] + [make_run(13, 180)]
    d = compute_drift(store(tmp_path / "b", mixed), "v1", DATASET)
    assert d.judge is None and "judged runs 7/7 (reference), 6/7 (window)" in d.notes[0]


def test_judge_drop_never_changes_status(tmp_path):
    good = [make_run(i, 180, judged=True) for i in range(7)]
    worse_judge = [make_run(7 + i, 180, judged=True) for i in range(7)]
    worse_judge = [EvalRun(info=r.info, cases=[c.model_copy(update={"summary_score": 1}) for c in r.cases])
                   for r in worse_judge]
    d = compute_drift(store(tmp_path, good + worse_judge), "v1", DATASET)
    assert d.judge.change == pytest.approx(-3.0) and d.status is DriftStatus.OK


# --- the key rule: drift fires although every per-run compare passes ---------------------------

def test_slow_slide_passes_every_compare_but_triggers_drift(tmp_path):
    # 14 runs, each 1 case (0.5 pp) worse than the one before: 95.0% down to 88.5%.
    evals = corrects_to_runs([190 - i for i in range(14)])
    for before, after in zip(evals, evals[1:]):
        cmp = compare_runs(before, after)
        assert cmp.strict_drop_pp == pytest.approx(0.5) and cmp.status is Status.PASS
    d = compute_drift(store(tmp_path, evals), "v1", DATASET)
    # Reference mean 93.5%, window mean 90.0%: 3.5 pp, between warning (2) and critical (4).
    assert d.strict.baseline == pytest.approx(0.935) and d.strict.candidate == pytest.approx(0.90)
    assert d.status is DriftStatus.WARNING and d.exit_code == 1
    assert drift_line(d) == ("Drift (7-run average): DRIFT WARNING (strict 90.0% vs 93.5% over the previous "
                             "7 runs, -3.5 pp; acceptable -3.5 pp)")


# --- config ----------------------------------------------------------------------------------

def test_drift_config_validation():
    assert DriftConfig().window == 7 and DriftConfig().min_history == 14
    with pytest.raises(ValidationError):
        DriftConfig(drop_warning_pp=4, drop_critical_pp=2)
    with pytest.raises(ValidationError):
        DriftConfig(window=7, min_history=7)
    with pytest.raises(ValidationError):
        DriftConfig(bogus=1)


def test_repo_eval_config_has_drift_section():
    from app.evals.config import DEFAULT_EVAL_CONFIG_PATH
    drift = EvalConfig.from_yaml(DEFAULT_EVAL_CONFIG_PATH).drift
    assert (drift.window, drift.min_history, drift.drop_warning_pp, drift.drop_critical_pp) == (7, 14, 2.0, 4.0)
    assert drift.floor_strict_pct is None
