"""Pydantic models for the human-labeled golden dataset."""

from datetime import datetime
from enum import StrEnum
from typing import Annotated, Self

from pydantic import AfterValidator, BaseModel, ConfigDict, StringConstraints, ValidationInfo, field_validator, model_validator

from app.schemas import Category, Summary

CASE_ID_PATTERN = r"^case_[0-9]{3,}$"
DATASET_VERSION_PATTERN = r"^v[1-9][0-9]*$"
TAG_PATTERN = r"^[a-z0-9]+(_[a-z0-9]+)*$"  # snake_case, so "Typo" and "typo" can't drift apart


class Difficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


def case_number(case_id: str) -> int:
    return int(case_id.removeprefix("case_"))


def format_case_id(number: int) -> str:
    return f"case_{number:03d}"


def _canonical_case_id(value: str) -> str:
    number = case_number(value)
    if number < 1 or format_case_id(number) != value:
        raise ValueError(f"must look like case_001 (got {value!r}, expected {format_case_id(max(number, 1))!r})")
    return value


def _not_blank(value: str) -> str:
    if not value.strip():
        raise ValueError("must not be empty or whitespace")
    return value


def _unique_tags(tags: list[str]) -> list[str]:
    seen = [t for i, t in enumerate(tags) if t in tags[:i]]
    if seen:
        raise ValueError(f"duplicate tags: {sorted(set(seen))}")
    return tags


CaseId = Annotated[str, StringConstraints(pattern=CASE_ID_PATTERN), AfterValidator(_canonical_case_id)]
# Email text is stored exactly as written; only an all-blank input is rejected.
EmailText = Annotated[str, AfterValidator(_not_blank)]
Tag = Annotated[str, StringConstraints(pattern=TAG_PATTERN)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
TagList = Annotated[list[Tag], AfterValidator(_unique_tags)]


def check_acceptable_categories(expected: Category | None, categories: list[Category]) -> list[Category]:
    """Empty is fine. Otherwise: no duplicates, and expected_category must be included."""
    if not categories:
        return categories
    dupes = sorted({c.value for i, c in enumerate(categories) if c in categories[:i]})
    if dupes:
        raise ValueError(f"duplicate categories: {dupes}")
    if expected is not None and expected not in categories:
        raise ValueError(f"must include expected_category {expected.value!r}")
    return categories


class GoldenCase(BaseModel):
    """One human-labeled case. Field order here is the key order written to disk."""

    model_config = ConfigDict(extra="forbid")

    id: CaseId
    input: EmailText
    expected_category: Category
    # Other labels a reasonable human could defend; see docs/labeling_guidelines.md.
    acceptable_categories: list[Category] = []
    ideal_summary: Summary
    expected_difficulty: Difficulty
    tags: TagList = []
    notes: NonEmptyStr

    @field_validator("acceptable_categories")
    @classmethod
    def _acceptable_includes_expected(cls, value: list[Category], info: ValidationInfo) -> list[Category]:
        # expected_category is validated first (field order); it is absent here if it was invalid.
        return check_acceptable_categories(info.data.get("expected_category"), value)


class GoldenHeader(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: Annotated[str, StringConstraints(pattern=DATASET_VERSION_PATTERN)]
    created_at: datetime
    description: NonEmptyStr


def duplicate_messages(cases: list[tuple[str, str | None]]) -> list[tuple[str, str]]:
    """(case id, message) pairs for repeated ids and exact-duplicate inputs.

    Takes (id, input) per case; input may be None when it is missing.
    Inputs are compared after stripping leading/trailing whitespace.
    """
    problems: list[tuple[str, str]] = []
    first_id: dict[str, int] = {}
    for i, (case_id, _) in enumerate(cases):
        if case_id in first_id:
            problems.append((case_id, f"duplicate id (also used by cases[{first_id[case_id]}])"))
        else:
            first_id[case_id] = i
    first_input: dict[str, str] = {}
    for case_id, text in cases:
        if text is None or not text.strip():
            continue
        key = text.strip()
        if key in first_input:
            problems.append((case_id, f"input is an exact duplicate of {first_input[key]}"))
        else:
            first_input[key] = case_id
    return problems


class GoldenDataset(GoldenHeader):
    cases: list[GoldenCase] = []

    @model_validator(mode="after")
    def _no_duplicates(self) -> Self:
        problems = duplicate_messages([(c.id, c.input) for c in self.cases])
        if problems:
            raise ValueError("; ".join(f"{cid}: {msg}" for cid, msg in problems))
        return self

    def next_case_id(self) -> str:
        """One past the highest existing number, so deleted ids are never reused."""
        return format_case_id(max((case_number(c.id) for c in self.cases), default=0) + 1)
