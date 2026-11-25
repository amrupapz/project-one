from collections.abc import Callable
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
import yaml
from openai import AsyncOpenAI
from openai.types import CompletionUsage
from openai.types.chat import ChatCompletion, ChatCompletionMessage
from openai.types.chat.chat_completion import Choice

VALID_PROMPT: dict = {
    "version": "v1",
    "created_at": "2026-11-25T00:00:00Z",
    "model_name": "openai/gpt-4o-mini",
    "temperature": 0,
    "system_prompt": "Classify the email.",
    "few_shot_examples": [
        {
            "input": "I was charged twice this month.",
            "output": {"category": "billing", "summary": "The customer was double charged."},
        }
    ],
}


@pytest.fixture
def write_prompt(tmp_path: Path) -> Callable[..., Path]:
    """Write a prompt dict to tmp_path/<name>.yaml and return the path."""

    def _write(data: dict, name: str | None = None) -> Path:
        path = tmp_path / f"{name or data.get('version', 'v1')}.yaml"
        path.write_text(yaml.safe_dump(data), encoding="utf-8")
        return path

    return _write


def make_completion(content: str, prompt_tokens: int = 50, completion_tokens: int = 20) -> ChatCompletion:
    return ChatCompletion(
        id="cmpl-test",
        object="chat.completion",
        created=0,
        model="openai/gpt-4o-mini",
        choices=[
            Choice(
                index=0,
                finish_reason="stop",
                message=ChatCompletionMessage(role="assistant", content=content),
            )
        ],
        usage=CompletionUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


@pytest.fixture
def fake_openai(monkeypatch: pytest.MonkeyPatch) -> Callable[..., tuple[AsyncOpenAI, AsyncMock]]:
    """An AsyncOpenAI client whose chat.completions.create returns canned responses in order."""

    def _make(*responses: ChatCompletion | Exception) -> tuple[AsyncOpenAI, AsyncMock]:
        client = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid/v1")
        mock = AsyncMock(side_effect=list(responses))
        monkeypatch.setattr(client.chat.completions, "create", mock)
        return client, mock

    return _make
