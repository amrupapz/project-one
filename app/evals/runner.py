"""Run every golden case through classify() and the judge, concurrently."""

import asyncio
import secrets
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import instructor

from app.classify import classify
from app.config import PromptConfig
from app.errors import ClassificationError
from app.evals.config import EvalConfig, file_sha256
from app.evals.judge_client import error_code, make_judge_client
from app.evals.models import CaseResult, EvalRun, RunInfo
from app.evals.scoring import acceptable_match, judge_summary, strict_match
from app.golden.io import load_golden
from app.golden.models import GoldenCase, GoldenDataset
from app.llm import make_client

ProgressFn = Callable[[int, int], None]  # (done, total)
FailureFn = Callable[[str, str, str], None]  # (case_id, stage "classify" | "judge", error code)


def new_run_id(now: datetime) -> str:
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


async def _run_case(
    case: GoldenCase,
    prompt_config: PromptConfig,
    eval_config: EvalConfig,
    classifier_client: instructor.AsyncInstructor,
    judge_client: instructor.AsyncInstructor,
    on_failure: FailureFn | None,
) -> CaseResult:
    labels = {
        "case_id": case.id,
        "input": case.input,
        "expected_category": case.expected_category,
        "acceptable_categories": case.acceptable_categories,
        "expected_difficulty": case.expected_difficulty,
        "ideal_summary": case.ideal_summary,
    }
    # Only the email text goes to the classifier, never the labels.
    try:
        result = await classify(case.input, prompt_config, client=classifier_client)
    except ClassificationError as exc:
        if on_failure:
            on_failure(case.id, "classify", error_code(exc))
        return CaseResult(**labels, error=str(exc))

    verdict = await judge_summary(
        case.input,
        case.ideal_summary,
        result.summary,
        rubric=eval_config.judge_rubric,
        judge_model=eval_config.judge_model,
        max_tokens=eval_config.judge_max_tokens,
        client=judge_client,
    )
    if verdict.error_code is not None and on_failure:
        on_failure(case.id, "judge", verdict.error_code)
    return CaseResult(
        **labels,
        predicted_category=result.category,
        predicted_summary=result.summary,
        strict_match=strict_match(result.category, case.expected_category),
        acceptable_match=acceptable_match(result.category, case.expected_category, case.acceptable_categories),
        summary_score=verdict.score,
        judge_reason=verdict.reason,
        latency_ms=result.latency_ms,
        input_tokens=result.input_tokens,
        output_tokens=result.output_tokens,
    )


async def run_eval(
    prompt_config: PromptConfig,
    dataset: GoldenDataset,
    *,
    eval_config: EvalConfig,
    concurrency: int | None = None,
    limit: int | None = None,
    classifier_client: instructor.AsyncInstructor | None = None,
    judge_client: instructor.AsyncInstructor | None = None,
    on_progress: ProgressFn | None = None,
    on_failure: FailureFn | None = None,
) -> list[CaseResult]:
    """Results in dataset order. A failed classify becomes an error row; the run continues.

    Each case holds one semaphore slot for both its classify and judge calls. The judge
    client's requests are additionally rate-limited (see judge_client.py).
    """
    concurrency = concurrency or eval_config.default_concurrency
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    if limit is not None and limit < 1:
        raise ValueError("limit must be >= 1")
    classifier_client = classifier_client or make_client()
    judge_client = judge_client or make_judge_client(eval_config)
    cases = dataset.cases[:limit] if limit else dataset.cases

    semaphore = asyncio.Semaphore(concurrency)
    done = 0

    async def bounded(case: GoldenCase) -> CaseResult:
        nonlocal done
        async with semaphore:
            result = await _run_case(case, prompt_config, eval_config, classifier_client, judge_client, on_failure)
        done += 1
        if on_progress:
            on_progress(done, len(cases))
        return result

    return list(await asyncio.gather(*(bounded(case) for case in cases)))


async def execute_run(
    prompt_path: str | Path,
    dataset_path: str | Path,
    eval_config_path: str | Path,
    *,
    concurrency: int | None = None,
    limit: int | None = None,
    notes: str | None = None,
    classifier_client: instructor.AsyncInstructor | None = None,
    judge_client: instructor.AsyncInstructor | None = None,
    on_progress: ProgressFn | None = None,
    on_failure: FailureFn | None = None,
) -> EvalRun:
    """Load the three input files, run the eval, and return the full run record (not yet saved)."""
    prompt_config = PromptConfig.from_yaml(prompt_path)
    dataset = load_golden(dataset_path)
    eval_config = EvalConfig.from_yaml(eval_config_path)
    concurrency = concurrency or eval_config.default_concurrency
    # A limit at or above the dataset size is a full run.
    case_limit = limit if limit is not None and limit < len(dataset.cases) else None

    started_at = datetime.now(UTC)
    info = RunInfo(
        run_id=new_run_id(started_at),
        started_at=started_at,
        prompt_version=prompt_config.version,
        prompt_file_sha256=file_sha256(prompt_path),
        classifier_model=prompt_config.model_name,
        judge_model=eval_config.judge_model,
        dataset_version=dataset.dataset_version,
        dataset_sha256=file_sha256(dataset_path),
        eval_config_sha256=file_sha256(eval_config_path),
        concurrency=concurrency,
        case_limit=case_limit,
        notes=notes,
    )
    cases = await run_eval(
        prompt_config,
        dataset,
        eval_config=eval_config,
        concurrency=concurrency,
        limit=case_limit,
        classifier_client=classifier_client,
        judge_client=judge_client,
        on_progress=on_progress,
        on_failure=on_failure,
    )
    return EvalRun(info=info, cases=cases)
