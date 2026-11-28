"""Alerts after a comparison. Optional: with SLACK_WEBHOOK_URL unset, nothing is sent.

Messages carry only run ids, numbers and the report path, never email text.
"""

import json
import os
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from app.evals.compare import Comparison
from app.evals.drift import Drift, DriftStatus, drift_line

SLACK_WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
TIMEOUT_S = 10.0

# (url, body bytes, headers, timeout) -> HTTP status. Swappable in tests.
PostFn = Callable[[str, bytes, dict[str, str], float], int]


class AlertError(RuntimeError):
    pass


class Alerter(Protocol):
    def send(self, comparison: Comparison, report_path: Path | None, drift: Drift | None = None) -> None: ...


def _urllib_post(url: str, body: bytes, headers: dict[str, str], timeout: float) -> int:
    request = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310 (https webhook)
        return response.status


def format_alert_text(cmp: Comparison, report_path: Path | None, drift: Drift | None = None) -> str:
    def pp(change: float | None) -> str:
        return "n/a" if change is None else f"{change * 100:+.1f} pp"

    judge = cmp.judge_score.change
    judge_text = "n/a" if judge is None else f"{judge:+.2f}"
    if not cmp.judge_used_in_status:
        judge_text += " (not used for status)"
    lines = [
        f"*Eval {cmp.headline}*",
        f"{cmp.baseline.prompt_version} ({cmp.baseline.run_id}) -> {cmp.candidate.prompt_version} ({cmp.candidate.run_id})",
        f"Strict {cmp.strict.candidate:.1%} ({pp(cmp.strict.change)}) | "
        f"Acceptable {cmp.acceptable.candidate:.1%} ({pp(cmp.acceptable.change)}) | Judge {judge_text}",
        f"Acceptable flips: {len(cmp.acceptable_regressions)} regressions, {len(cmp.acceptable_improvements)} improvements",
    ]
    lines += [f"Warning: {w}" for w in cmp.warnings]
    if drift is not None and drift.status not in (None, DriftStatus.OK):  # only when there is something to say
        lines.append(drift_line(drift))
    if report_path is not None:
        lines.append(f"Report: {report_path}")
    return "\n".join(lines)


class SlackAlerter:
    def __init__(self, webhook_url: str, post: PostFn | None = None) -> None:
        self.webhook_url = webhook_url
        self._post = post or _urllib_post

    @classmethod
    def from_env(cls, post: PostFn | None = None) -> "SlackAlerter | None":
        url = os.environ.get(SLACK_WEBHOOK_ENV, "").strip()
        return cls(url, post) if url else None

    def send(self, comparison: Comparison, report_path: Path | None, drift: Drift | None = None) -> None:
        body = json.dumps({"text": format_alert_text(comparison, report_path, drift)}).encode("utf-8")
        try:
            status = self._post(self.webhook_url, body, {"Content-Type": "application/json"}, TIMEOUT_S)
        except OSError as exc:  # URLError, HTTPError and timeouts are all OSError
            raise AlertError(f"Slack alert failed: {exc}") from exc
        if not 200 <= status < 300:
            raise AlertError(f"Slack alert failed: HTTP {status}")
