"""Evaluation engine: run a prompt over the golden dataset, score it, store it.

    from app.evals import execute_run, run_eval, save_run, load_run, list_runs, compute_scorecard
    from app.evals import compare_runs, Status   # Phase 4: comparison
"""

from app.evals.compare import Comparison, CompareRefused, Status, compare_runs, format_comparison
from app.evals.config import CompareConfig, EvalConfig
from app.evals.models import CaseResult, EvalRun, RunInfo
from app.evals.runner import execute_run, run_eval
from app.evals.scorecard import Scorecard, compute_scorecard, format_scorecard
from app.evals.scoring import JudgeOutcome, acceptable_match, judge_summary, strict_match
from app.evals.store import RunSummary, list_runs, load_run, save_run

__all__ = [
    "CaseResult",
    "CompareConfig",
    "CompareRefused",
    "Comparison",
    "EvalConfig",
    "EvalRun",
    "JudgeOutcome",
    "RunInfo",
    "RunSummary",
    "Scorecard",
    "Status",
    "acceptable_match",
    "compare_runs",
    "compute_scorecard",
    "execute_run",
    "format_comparison",
    "format_scorecard",
    "judge_summary",
    "list_runs",
    "load_run",
    "run_eval",
    "save_run",
    "strict_match",
]
