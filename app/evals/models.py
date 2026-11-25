"""Records for one eval run. These map 1:1 to the SQLite tables."""

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.golden.models import Difficulty
from app.schemas import Category


class RunInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    started_at: datetime
    prompt_version: str
    prompt_file_sha256: str
    classifier_model: str
    judge_model: str
    dataset_version: str
    dataset_sha256: str
    eval_config_sha256: str
    concurrency: int = Field(ge=1)
    case_limit: int | None = Field(default=None, ge=1)  # None = full run
    notes: str | None = None

    @property
    def is_partial(self) -> bool:
        return self.case_limit is not None


class CaseResult(BaseModel):
    """One golden case's outcome. Snapshot of the case labels is included so later
    phases can report on a run without reopening the dataset file."""

    model_config = ConfigDict(frozen=True)

    case_id: str
    input: str
    expected_category: Category
    acceptable_categories: list[Category]
    expected_difficulty: Difficulty
    ideal_summary: str
    predicted_category: Category | None = None
    predicted_summary: str | None = None
    strict_match: bool = False
    acceptable_match: bool = False
    summary_score: int | None = Field(default=None, ge=1, le=5)
    judge_reason: str | None = None
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    error: str | None = None  # set when classify() failed; the case then counts as wrong


class EvalRun(BaseModel):
    info: RunInfo
    cases: list[CaseResult]  # dataset order
