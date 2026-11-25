"""Evaluation settings loaded from eval_config.yaml."""

import hashlib
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from app.config import MODEL_NAME_PATTERN, NonEmptyStr

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "eval_config.yaml"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CompareConfig(BaseModel):
    """Thresholds for `python -m app.evals compare`. Drops are baseline minus candidate."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    strict_drop_warning_pp: float = Field(default=3.0, gt=0)
    strict_drop_critical_pp: float = Field(default=8.0, gt=0)
    # A drop in mean judge score (1-5 scale) of at least this raises PASS to WARNING; never CRITICAL.
    judge_score_drop_warning: float = Field(default=0.3, gt=0)
    # Below this fraction of scored cases, a run's judge mean is not used for the status.
    min_judge_coverage: float = Field(default=0.95, gt=0, le=1)
    trend_runs: int = Field(default=10, ge=2, le=100)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.strict_drop_critical_pp < self.strict_drop_warning_pp:
            raise ValueError("strict_drop_critical_pp must be >= strict_drop_warning_pp")
        return self


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_model: Annotated[str, StringConstraints(pattern=MODEL_NAME_PATTERN)]
    # Caps output per judge call so OpenRouter doesn't reserve credit for the model maximum.
    judge_max_tokens: int = Field(default=300, ge=1, le=4096)
    # Shared limit on judge requests (including retries) across a whole run.
    judge_requests_per_minute: int = Field(default=15, ge=1, le=600)
    default_concurrency: int = Field(default=8, ge=1, le=64)
    judge_rubric: NonEmptyStr
    db_path: Path = Path("results/eval.db")
    compare: CompareConfig = CompareConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate; a relative db_path is resolved against the config file's folder."""
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        config = cls.model_validate(data)
        if not config.db_path.is_absolute():
            config = config.model_copy(update={"db_path": path.resolve().parent / config.db_path})
        return config
