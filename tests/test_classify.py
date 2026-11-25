import json

import httpx2
import openai
import pytest

from app.classify import build_messages, classify, main
from app.config import PromptConfig
from app.errors import ClassificationError, MissingAPIKeyError
from app.llm import make_client
from tests.conftest import VALID_PROMPT, make_completion

GOOD = json.dumps({"category": "billing", "summary": "The customer wants a refund."})


@pytest.fixture
def config() -> PromptConfig:
    return PromptConfig.model_validate(VALID_PROMPT)


async def test_classify_success(config, fake_openai):
    raw, mock = fake_openai(make_completion(GOOD, prompt_tokens=120, completion_tokens=18))
    result = await classify("Please refund me.", config, client=make_client(raw))

    assert result.category == "billing"
    assert result.summary == "The customer wants a refund."
    assert result.prompt_version == "v1"
    assert result.model_name == "openai/gpt-4o-mini"
    assert result.input_tokens == 120
    assert result.output_tokens == 18
    assert result.latency_ms >= 0
    assert mock.await_count == 1

    kwargs = mock.await_args.kwargs
    assert kwargs["model"] == config.model_name
    assert kwargs["temperature"] == config.temperature
    assert kwargs["messages"][-1] == {"role": "user", "content": "Please refund me."}


def test_build_messages_includes_few_shots(config):
    msgs = build_messages("hello", config)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert json.loads(msgs[2]["content"])["category"] == "billing"


@pytest.mark.parametrize(
    "bad_content",
    [
        "I'm sorry, I can't help with that.",  # refusal / not JSON
        '{"category": "shipping", "summary": "x"}',  # invalid enum
        '{"category": "billing", "summary": ""}',  # empty summary
    ],
)
async def test_retries_once_then_succeeds(config, fake_openai, bad_content):
    raw, mock = fake_openai(
        make_completion(bad_content, prompt_tokens=100, completion_tokens=10),
        make_completion(GOOD, prompt_tokens=130, completion_tokens=12),
    )
    result = await classify("Please refund me.", config, client=make_client(raw))
    assert result.category == "billing"
    assert mock.await_count == 2
    # Token counts include the failed attempt, since Phase 3 scores cost.
    assert (result.input_tokens, result.output_tokens) == (230, 22)


async def test_fails_with_clear_error_after_one_retry(config, fake_openai):
    refusal = make_completion("I can't help with that.")
    raw, mock = fake_openai(refusal, refusal, make_completion(GOOD))
    with pytest.raises(ClassificationError, match="no valid output"):
        await classify("Please refund me.", config, client=make_client(raw))
    assert mock.await_count == 2  # exactly one retry, third response never requested


async def test_api_error_becomes_classification_error(config, fake_openai):
    request = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")
    error = openai.APIConnectionError(request=request)
    raw, _ = fake_openai(error, error)
    with pytest.raises(ClassificationError):
        await classify("Please refund me.", config, client=make_client(raw))


async def test_empty_email_rejected(config, fake_openai):
    raw, mock = fake_openai()
    with pytest.raises(ValueError):
        await classify("   ", config, client=make_client(raw))
    mock.assert_not_awaited()


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        make_client()


def test_cli_prints_json(monkeypatch, capsys, fake_openai):
    raw, _ = fake_openai(make_completion(GOOD))
    monkeypatch.setattr("app.classify.make_client", lambda: make_client(raw))
    assert main(["--prompt", "prompts/v1.yaml", "Please refund me."]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["category"] == "billing"
    assert printed["prompt_version"] == "v1"


def test_cli_reports_error(monkeypatch, capsys):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    assert main(["--prompt", "prompts/v1.yaml", "hi"]) == 1
    assert "OPENROUTER_API_KEY" in capsys.readouterr().err
