"""Per-case scoring: category matches (pure) and the LLM summary judge."""

import json
from typing import Annotated

import instructor
import openai
from instructor.core.exceptions import InstructorError
from pydantic import BaseModel, Field, StringConstraints

from app.llm import MAX_RETRIES
from app.schemas import Category

JUDGE_TEMPERATURE = 0.0


def strict_match(predicted: Category | None, expected: Category) -> bool:
    return predicted is not None and predicted == expected


def acceptable_match(predicted: Category | None, expected: Category, acceptable: list[Category]) -> bool:
    """predicted is in acceptable_categories if the case has them, else same as strict_match."""
    if predicted is None:
        return False
    return predicted in acceptable if acceptable else predicted == expected


class JudgeVerdict(BaseModel):
    # reason comes first so the judge explains before it commits to a number.
    reason: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=300)]
    score: int = Field(ge=1, le=5)


class JudgeOutcome(BaseModel):
    """score is None when the judge call failed; reason then holds the error text."""

    score: int | None
    reason: str


def build_judge_messages(email: str, ideal_summary: str, candidate_summary: str, rubric: str) -> list[dict[str, str]]:
    """The rubric is the only instruction text. The case is passed as one JSON object, so
    nothing inside the email (e.g. an injected instruction or a fake closing tag) can
    escape into the instructions. No prompt version or model name is included."""
    payload = json.dumps(
        {"email": email, "ideal_summary": ideal_summary, "candidate_summary": candidate_summary},
        ensure_ascii=False,
        indent=2,
    )
    return [
        {"role": "system", "content": rubric},
        {"role": "user", "content": payload},
    ]


async def judge_summary(
    email: str,
    ideal_summary: str,
    candidate_summary: str,
    *,
    rubric: str,
    judge_model: str,
    max_tokens: int,
    client: instructor.AsyncInstructor,
) -> JudgeOutcome:
    """Score candidate_summary 1-5. Never raises for API or output errors."""
    try:
        verdict = await client.chat.completions.create(
            model=judge_model,
            temperature=JUDGE_TEMPERATURE,
            max_tokens=max_tokens,
            messages=build_judge_messages(email, ideal_summary, candidate_summary, rubric),
            response_model=JudgeVerdict,
            max_retries=MAX_RETRIES,
        )
    except (InstructorError, openai.OpenAIError) as exc:
        return JudgeOutcome(score=None, reason=f"judge failed: {type(exc).__name__}: {exc}"[:1000])
    return JudgeOutcome(score=verdict.score, reason=verdict.reason)
