"""Evaluation settings loaded from eval_config.yaml."""

import hashlib
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.config import MODEL_NAME_PATTERN, NonEmptyStr

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL_CONFIG_PATH = REPO_ROOT / "eval_config.yaml"


def file_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    judge_model: Annotated[str, StringConstraints(pattern=MODEL_NAME_PATTERN)]
    default_concurrency: int = Field(default=8, ge=1, le=64)
    judge_rubric: NonEmptyStr
    db_path: Path = Path("results/eval.db")

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
