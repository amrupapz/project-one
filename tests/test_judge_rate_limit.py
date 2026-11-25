"""Judge rate limiting, 429 retries, failure warning and compact failure logs.

All timing runs on a fake clock, so nothing actually sleeps.
"""

import asyncio
import heapq
import itertools
import json
from typing import Any

import httpx2
import openai
import pytest
from openai import AsyncOpenAI

from app.evals import EvalConfig, compute_scorecard, format_scorecard, judge_summary, run_eval
from app.evals.judge_client import (
    JUDGE_MAX_429_RETRIES,
    JUDGE_MAX_RETRY_WAIT_S,
    RateLimiter,
    error_code,
    make_judge_client,
    retry_after_seconds,
)
from app.golden.models import GoldenDataset
from app.llm import make_client
from tests.conftest import make_completion
from tests.golden_fixtures import fixture_case, fixture_dataset
from tests.test_evals import RUBRIC, classifier_reply, judge_reply, make_case, make_info, prompt_config  # noqa: F401

REQUEST = httpx2.Request("POST", "https://example.invalid/v1/chat/completions")


class FakeClock:
    """Virtual time. sleep() parks the caller; drive() wakes sleepers in time order
    whenever every task is blocked, so concurrent waits behave like real ones."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []
        self._sleepers: list[tuple[float, int, asyncio.Future[None]]] = []
        self._seq = itertools.count()

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        heapq.heappush(self._sleepers, (self.now + seconds, next(self._seq), future))
        await future

    async def drive(self, awaitable: Any) -> Any:
        task = asyncio.ensure_future(awaitable)
        while not task.done():
            for _ in range(50):  # let every runnable task proceed until it blocks
                await asyncio.sleep(0)
            if self._sleepers and not task.done():
                wake_at, _, future = heapq.heappop(self._sleepers)
                self.now = max(self.now, wake_at)
                future.set_result(None)
        return task.result()


def rate_limit_error(headers: dict[str, str] | None = None) -> openai.RateLimitError:
    response = httpx2.Response(429, headers=headers or {}, request=REQUEST)
    return openai.RateLimitError("new accounts are limited to 20 requests per minute", response=response, body=None)


def judge_setup(monkeypatch, replies, *, per_minute: int = 15):
    """A judge client on a fake clock. `replies` is a function (call_number) -> reply or exception.
    Returns (client, clock, call_times)."""
    clock = FakeClock()
    raw = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid/v1")
    call_times: list[float] = []

    async def create(**kwargs: Any):
        call_times.append(clock.now)
        reply = replies(len(call_times))
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(raw.chat.completions, "create", create)
    config = EvalConfig(judge_model="anthropic/claude-haiku-4.5", judge_rubric=RUBRIC, judge_requests_per_minute=per_minute)
    return make_judge_client(config, openai_client=raw, clock=clock), clock, call_times, config


async def judge_once(client, config, email: str = "FIXTURE email"):
    return await judge_summary(
        email, "FIXTURE ideal.", "FIXTURE candidate.",
        rubric=config.judge_rubric, judge_model=config.judge_model,
        max_tokens=config.judge_max_tokens, client=client,
    )


# --- config -----------------------------------------------------------------------

def test_requests_per_minute_default_and_bounds():
    assert EvalConfig(judge_model="a/b", judge_rubric="r").judge_requests_per_minute == 15
    for bad in (0, 601):
        with pytest.raises(ValueError):
            EvalConfig(judge_model="a/b", judge_rubric="r", judge_requests_per_minute=bad)


# --- rate limiter -------------------------------------------------------------------

@pytest.mark.parametrize("per_minute", [15, 20, 3])
async def test_judge_calls_never_exceed_per_minute_limit(monkeypatch, per_minute):
    client, clock, call_times, config = judge_setup(monkeypatch, lambda n: judge_reply(), per_minute=per_minute)
    n_calls = 40

    async def all_at_once():
        return await asyncio.gather(*(judge_once(client, config, f"FIXTURE email {i}") for i in range(n_calls)))

    outcomes = await clock.drive(all_at_once())
    assert all(o.score == 4 for o in outcomes)  # calls waited their turn instead of failing
    assert len(call_times) == n_calls
    times = sorted(call_times)
    for i in range(len(times) - per_minute):
        assert times[i + per_minute] - times[i] >= 60.0  # no 60s window holds more than per_minute
    assert times[-1] == pytest.approx((n_calls - 1) * 60.0 / per_minute)  # and no slower than needed


async def test_limiter_lets_first_call_through_immediately():
    clock = FakeClock()
    limiter = RateLimiter(15, clock)
    await clock.drive(limiter.acquire())
    assert clock.sleeps == []


async def test_limiter_is_shared_across_a_run_but_not_the_classifier(monkeypatch, prompt_config):
    judge_client, clock, judge_times, config = judge_setup(monkeypatch, lambda n: judge_reply(), per_minute=6)
    classifier_times: list[float] = []
    raw_classifier = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid/v1")

    async def classify_create(**kwargs):
        classifier_times.append(clock.now)  # same virtual clock as the judge
        return classifier_reply("general")

    monkeypatch.setattr(raw_classifier.chat.completions, "create", classify_create)
    dataset = GoldenDataset.model_validate(fixture_dataset(*(fixture_case(n) for n in range(1, 9))))

    results = await clock.drive(run_eval(
        prompt_config, dataset, eval_config=config, concurrency=8,
        classifier_client=make_client(raw_classifier), judge_client=judge_client,
    ))
    assert all(r.summary_score == 4 for r in results)
    assert classifier_times == [0.0] * 8  # the classifier is not rate-limited
    assert sorted(judge_times) == [i * 10.0 for i in range(8)]  # 6/min = one every 10s, across all cases


# --- 429 handling ---------------------------------------------------------------------

@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"retry-after": "7"}, 7.0),
        ({"retry-after": "1.5"}, 1.5),
        ({"retry-after-ms": "2500"}, 2.5),
        ({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}, 0.0),  # a date in the past
        ({"retry-after": "soon"}, None),
        ({}, None),
    ],
)
def test_retry_after_parsing(headers, expected):
    assert retry_after_seconds(rate_limit_error(headers)) == expected


async def test_retry_after_is_honored(monkeypatch):
    replies = lambda n: rate_limit_error({"retry-after": "30"}) if n == 1 else judge_reply(5)
    client, clock, call_times, config = judge_setup(monkeypatch, replies)
    outcome = await clock.drive(judge_once(client, config))
    assert outcome.score == 5
    assert call_times == [0.0, 30.0]


async def test_retry_after_pauses_other_queued_calls_too(monkeypatch):
    replies = lambda n: rate_limit_error({"retry-after": "30"}) if n == 1 else judge_reply(5)
    client, clock, call_times, config = judge_setup(monkeypatch, replies, per_minute=60)

    async def three():
        return await asyncio.gather(*(judge_once(client, config, f"FIXTURE {i}") for i in range(3)))

    outcomes = await clock.drive(three())
    assert [o.score for o in outcomes] == [5, 5, 5]
    assert call_times[0] == 0.0
    assert all(t >= 30.0 for t in call_times[1:])  # nobody hammers the API during the pause


async def test_retry_wait_is_capped(monkeypatch):
    replies = lambda n: rate_limit_error({"retry-after": "3600"}) if n == 1 else judge_reply()
    client, clock, call_times, config = judge_setup(monkeypatch, replies)
    await clock.drive(judge_once(client, config))
    assert call_times == [0.0, JUDGE_MAX_RETRY_WAIT_S]


async def test_backoff_without_retry_after(monkeypatch):
    replies = lambda n: rate_limit_error() if n <= 3 else judge_reply()
    client, clock, call_times, config = judge_setup(monkeypatch, replies, per_minute=600)
    outcome = await clock.drive(judge_once(client, config))
    assert outcome.score == 4
    assert [b - a for a, b in zip(call_times, call_times[1:])] == [5.0, 10.0, 20.0]


async def test_retries_stop_after_three_and_score_is_null(monkeypatch):
    client, clock, call_times, config = judge_setup(monkeypatch, lambda n: rate_limit_error({"retry-after": "1"}))
    outcome = await clock.drive(judge_once(client, config))
    assert len(call_times) == 1 + JUDGE_MAX_429_RETRIES == 4
    assert outcome.score is None
    assert outcome.error_code == "429"
    assert outcome.reason.startswith("judge failed (429)")
    assert len(outcome.reason) <= 300 and "\n" not in outcome.reason


async def test_non_429_errors_are_not_retried(monkeypatch):
    error = openai.InternalServerError("boom", response=httpx2.Response(500, request=REQUEST), body=None)
    client, clock, call_times, config = judge_setup(monkeypatch, lambda n: error)
    outcome = await clock.drive(judge_once(client, config))
    assert len(call_times) == 1 and outcome.error_code == "500"


async def test_reask_after_invalid_output_also_waits_for_a_slot(monkeypatch):
    replies = lambda n: make_completion("not json") if n == 1 else judge_reply(3)
    client, clock, call_times, config = judge_setup(monkeypatch, replies, per_minute=15)
    outcome = await clock.drive(judge_once(client, config))
    assert outcome.score == 3
    assert call_times == [0.0, 4.0]


# --- compact failure reporting ------------------------------------------------------------

def test_error_code():
    assert error_code(rate_limit_error()) == "429"
    assert error_code(openai.APIConnectionError(request=REQUEST)) == "connection"
    wrapped = RuntimeError("outer")
    wrapped.__cause__ = rate_limit_error()
    assert error_code(wrapped) == "429"
    assert error_code(ValueError("x")) == "ValueError"


async def test_failures_reported_as_one_short_line(monkeypatch, prompt_config):
    judge_client, clock, _, config = judge_setup(
        monkeypatch, lambda n: rate_limit_error({"retry-after": "1"}), per_minute=600
    )
    raw_classifier = AsyncOpenAI(api_key="test-key", base_url="https://example.invalid/v1")

    async def classify_create(**kwargs):
        return classifier_reply("general")

    monkeypatch.setattr(raw_classifier.chat.completions, "create", classify_create)
    dataset = GoldenDataset.model_validate(fixture_dataset(fixture_case(1), fixture_case(2)))
    reported: list[tuple[str, str, str]] = []
    results = await clock.drive(run_eval(
        prompt_config, dataset, eval_config=config,
        classifier_client=make_client(raw_classifier), judge_client=judge_client,
        on_failure=lambda *args: reported.append(args),
    ))
    assert sorted(reported) == [("case_001", "judge", "429"), ("case_002", "judge", "429")]
    assert all(r.summary_score is None and r.strict_match for r in results)


def test_cli_prints_compact_failure_line(capsys):
    from app.evals.cli import _failure

    _failure("case_012", "judge", "429")
    err = capsys.readouterr().err
    assert err.strip() == "judge failed: case_012 (429)"


# --- scorecard warning ------------------------------------------------------------------------

def _cases(scored: int, judge_failed: int, classify_errors: int = 0):
    cases = [make_case(f"case_{i:03d}") for i in range(1, scored + 1)]
    cases += [make_case(f"case_{100 + i:03d}", summary_score=None, judge_reason="judge failed (429): x") for i in range(judge_failed)]
    cases += [make_case(f"case_{200 + i:03d}", predicted=None) for i in range(classify_errors)]
    return cases


@pytest.mark.parametrize(
    ("scored", "judge_failed", "classify_errors", "warn"),
    [
        (56, 12, 0, True),   # the reported run: 12 of 68 failed
        (19, 1, 0, False),   # exactly 5% is not more than 5%
        (28, 2, 0, True),    # 6.7%
        (40, 0, 0, False),
        (19, 1, 10, False),  # classify errors are not judge calls: 1 of 20 = 5%
        (0, 0, 5, False),    # judge never called
    ],
)
def test_judge_failure_warning(scored, judge_failed, classify_errors, warn):
    card = compute_scorecard(_cases(scored, judge_failed, classify_errors))
    text = format_scorecard(make_info(), card)
    expected = f"WARNING: judge failed on {judge_failed} of {scored + judge_failed} cases; judge mean is unreliable"
    assert (expected in text) is warn
    assert ("WARNING: judge failed" in text) is warn
