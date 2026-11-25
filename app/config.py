"""Prompt configuration loaded from versioned YAML files."""

from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from app.schemas import ClassifyOutput

VERSION_PATTERN = r"^v[1-9][0-9]*$"
# OpenRouter ids look like "provider/model", optionally with a ":variant" suffix.
MODEL_NAME_PATTERN = r"^[\w.-]+/[\w.:-]+$"

NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class FewShotExample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    input: NonEmptyStr
    output: ClassifyOutput


class PromptConfig(BaseModel):
    # extra="forbid" so a typo like "temprature" fails instead of silently using the default.
    model_config = ConfigDict(extra="forbid", frozen=True, protected_namespaces=())

    version: Annotated[str, StringConstraints(pattern=VERSION_PATTERN)]
    created_at: datetime
    model_name: Annotated[str, StringConstraints(pattern=MODEL_NAME_PATTERN)]
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    system_prompt: NonEmptyStr
    few_shot_examples: list[FewShotExample] = Field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Load and validate a prompt file. Raises pydantic.ValidationError or ValueError."""
        path = Path(path)
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"{path}: expected a YAML mapping at the top level")
        config = cls.model_validate(data)
        if path.stem != config.version:
            raise ValueError(
                f"{path}: version {config.version!r} does not match file name {path.stem!r}"
            )
        return config
