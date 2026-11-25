"""Transport for judge calls: a shared rate limiter, 429 retries, compact error codes.

Applies only to the judge. Every HTTP request the judge makes (including instructor's
re-ask after an invalid answer, and 429 retries) waits for a slot from one limiter.
"""

import asyncio
import email.utils
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from typing import Any, Protocol

import instructor
import openai
from instructor.core.exceptions import InstructorError
from openai import AsyncOpenAI

from app.evals.config import EvalConfig
from app.llm import make_client, make_openrouter_client

JUDGE_MAX_429_RETRIES = 3
JUDGE_MAX_RETRY_WAIT_S = 60.0
FALLBACK_BACKOFF_S = 5.0  # first wait when a 429 has no Retry-After; doubles per retry


class Clock(Protocol):
    def monotonic(self) -> float: ...
    def sleep(self, seconds: float) -> Awaitable[None]: ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    async def sleep(self, seconds: float) -> None:
        await asyncio.sleep(seconds)


class RateLimiter:
    """Spaces calls evenly: at most `per_minute` start in any 60-second window.

    Callers wait for their slot instead of failing. defer() pushes every later slot
    back, so one 429's Retry-After pauses all queued calls, not just the one that failed.
    """

    def __init__(self, per_minute: int, clock: Clock | None = None) -> None:
        if per_minute < 1:
            raise ValueError("per_minute must be >= 1")
        self.interval = 60.0 / per_minute
        self.clock = clock or SystemClock()
        self._next_slot = float("-inf")
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = self.clock.monotonic()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self.interval
        if slot > now:
            await self.clock.sleep(slot - now)

    def defer(self, seconds: float) -> None:
        self._next_slot = max(self._next_slot, self.clock.monotonic() + seconds)


def retry_after_seconds(exc: openai.APIStatusError) -> float | None:
    """Seconds from Retry-After / retry-after-ms headers, or None if absent or unparseable."""
    headers = exc.response.headers
    if (ms := headers.get("retry-after-ms")) is not None:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - datetime.now(UTC)).total_seconds())


def retry_delay(exc: openai.APIStatusError, retry_number: int) -> float:
    """Wait before retry `retry_number` (1-based): Retry-After if given, else backoff; capped."""
    header = retry_after_seconds(exc)
    delay = header if header is not None else FALLBACK_BACKOFF_S * 2 ** (retry_number - 1)
    return min(delay, JUDGE_MAX_RETRY_WAIT_S)


def rate_limited(
    create: Callable[..., Awaitable[Any]],
    limiter: RateLimiter,
    *,
    max_429_retries: int = JUDGE_MAX_429_RETRIES,
) -> Callable[..., Awaitable[Any]]:
    """Wrap a raw chat.completions.create: wait for a slot before each request and
    retry a 429 up to max_429_retries times. Other errors pass straight through."""

    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        retries = 0
        while True:
            await limiter.acquire()
            try:
                return await create(*args, **kwargs)
            except openai.RateLimitError as exc:
                if retries >= max_429_retries:
                    raise
                retries += 1
                limiter.defer(retry_delay(exc, retries))

    return wrapped


def make_judge_client(
    eval_config: EvalConfig,
    *,
    openai_client: AsyncOpenAI | None = None,
    clock: Clock | None = None,
) -> instructor.AsyncInstructor:
    """One judge client per run, so all its calls share one limiter.

    The default OpenRouter client has the SDK's own retries off: they would retry 429s
    behind the limiter's back. A given openai_client (tests) is used as-is.
    """
    raw = openai_client or make_openrouter_client(max_retries=0)
    limiter = RateLimiter(eval_config.judge_requests_per_minute, clock)
    raw.chat.completions.create = rate_limited(raw.chat.completions.create, limiter)  # type: ignore[method-assign]
    return make_client(raw)


def error_code(exc: BaseException) -> str:
    """Short code for a failure: an HTTP status ('429'), 'connection', 'timeout',
    'invalid_output', or the exception type. Looks through wrapped causes."""
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, openai.APIStatusError):
            return str(current.status_code)
        if isinstance(current, openai.APITimeoutError):
            return "timeout"
        if isinstance(current, openai.APIConnectionError):
            return "connection"
        current = current.__cause__ or current.__context__
    if isinstance(exc, InstructorError):
        return "invalid_output"
    return type(exc).__name__
