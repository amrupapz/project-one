"""Evaluation engine: run a prompt over the golden dataset, score it, store it.

    from app.evals import execute_run, run_eval, save_run, load_run, list_runs, compute_scorecard
"""

from app.evals.config import EvalConfig
from app.evals.models import CaseResult, EvalRun, RunInfo
from app.evals.runner import execute_run, run_eval
from app.evals.scorecard import Scorecard, compute_scorecard, format_scorecard
from app.evals.scoring import JudgeOutcome, acceptable_match, judge_summary, strict_match
from app.evals.store import RunSummary, list_runs, load_run, save_run

__all__ = [
    "CaseResult",
    "EvalConfig",
    "EvalRun",
    "JudgeOutcome",
    "RunInfo",
    "RunSummary",
    "Scorecard",
    "acceptable_match",
    "compute_scorecard",
    "execute_run",
    "format_scorecard",
    "judge_summary",
    "list_runs",
    "load_run",
    "run_eval",
    "save_run",
    "strict_match",
]
