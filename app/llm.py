"""OpenRouter client wrapped with instructor for structured output."""

import os

import instructor
from openai import AsyncOpenAI

from app.errors import MissingAPIKeyError

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
API_KEY_ENV_VAR = "OPENROUTER_API_KEY"

# JSON mode works across more OpenRouter models than tool calling.
# Switch to instructor.Mode.TOOLS for models with reliable function calling.
INSTRUCTOR_MODE = instructor.Mode.JSON

# Instructor re-asks the model once on a validation failure (2 attempts total).
MAX_RETRIES = 1

# Cap on output tokens per classifier call. Without it OpenRouter reserves credit for
# the model's maximum output on every request (402 in_flight_budget_exhausted).
# The JSON answer is a category plus one sentence, well under this.
CLASSIFIER_MAX_TOKENS = 300


def make_openrouter_client(*, max_retries: int | None = None) -> AsyncOpenAI:
    """Raw OpenRouter client with the key from env. max_retries=None keeps the SDK default."""
    api_key = os.environ.get(API_KEY_ENV_VAR)
    if not api_key:
        raise MissingAPIKeyError(f"{API_KEY_ENV_VAR} is not set (see .env.example)")
    if max_retries is None:
        return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key)
    return AsyncOpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, max_retries=max_retries)


def make_client(openai_client: AsyncOpenAI | None = None) -> instructor.AsyncInstructor:
    """Wrap an AsyncOpenAI client. Builds an OpenRouter client from env if none is given."""
    return instructor.from_openai(openai_client or make_openrouter_client(), mode=INSTRUCTOR_MODE)
