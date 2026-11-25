"""Output models for the classifier."""

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints

SUMMARY_MAX_CHARS = 300

Summary = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=SUMMARY_MAX_CHARS),
]


class Category(StrEnum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"


class ClassifyOutput(BaseModel):
    """What the model must return. Also the schema for few-shot example outputs.

    Extra keys from the model are ignored rather than rejected.
    """

    category: Category
    summary: Summary = Field(description="One sentence summarizing the customer's request.")


class ClassifyResult(ClassifyOutput):
    """Validated model output plus run metadata used for scoring in later phases."""

    model_config = ConfigDict(protected_namespaces=())

    prompt_version: str
    model_name: str
    latency_ms: float = Field(ge=0)
    input_tokens: int | None = None
    output_tokens: int | None = None
