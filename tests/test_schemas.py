import pytest
from pydantic import ValidationError

from app.schemas import SUMMARY_MAX_CHARS, Category, ClassifyOutput


@pytest.mark.parametrize("category", ["billing", "technical", "account", "general"])
def test_valid_categories(category):
    out = ClassifyOutput(category=category, summary="The customer needs help.")
    assert out.category == Category(category)


@pytest.mark.parametrize("category", ["shipping", "Billing", "", None])
def test_invalid_category_rejected(category):
    with pytest.raises(ValidationError):
        ClassifyOutput(category=category, summary="The customer needs help.")


@pytest.mark.parametrize("summary", ["", "   ", "x" * (SUMMARY_MAX_CHARS + 1)])
def test_invalid_summary_rejected(summary):
    with pytest.raises(ValidationError, match="summary"):
        ClassifyOutput(category="general", summary=summary)


def test_summary_is_stripped_and_max_length_allowed():
    assert ClassifyOutput(category="general", summary="  hi  ").summary == "hi"
    ClassifyOutput(category="general", summary="x" * SUMMARY_MAX_CHARS)


def test_missing_summary_rejected():
    with pytest.raises(ValidationError, match="summary"):
        ClassifyOutput.model_validate({"category": "billing"})


def test_parses_model_json():
    out = ClassifyOutput.model_validate_json('{"category": "technical", "summary": "App crashes."}')
    assert out.category is Category.TECHNICAL
