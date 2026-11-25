"""Classify a customer support email with a versioned prompt.

CLI: python -m app.classify --prompt prompts/v1.yaml "email text"
"""

import argparse
import asyncio
import sys
import time
from typing import Any

import instructor
import openai
from instructor.core.exceptions import InstructorError

from app.config import PromptConfig
from app.errors import ClassificationError, MissingAPIKeyError
from app.llm import CLASSIFIER_MAX_TOKENS, MAX_RETRIES, make_client
from app.schemas import ClassifyOutput, ClassifyResult


def build_messages(email: str, config: PromptConfig) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": config.system_prompt}]
    for example in config.few_shot_examples:
        messages.append({"role": "user", "content": example.input})
        messages.append({"role": "assistant", "content": example.output.model_dump_json()})
    messages.append({"role": "user", "content": email})
    return messages


def _usage(completion: Any) -> tuple[int | None, int | None]:
    usage = getattr(completion, "usage", None)
    if usage is None:
        return None, None
    return getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)


async def classify(
    email: str,
    config: PromptConfig,
    *,
    client: instructor.AsyncInstructor | None = None,
) -> ClassifyResult:
    """Classify one email. Raises ClassificationError if no valid output after one retry.

    latency_ms and token counts cover the whole call, including the retry if one happened.
    """
    if not email.strip():
        raise ValueError("email must not be empty")
    client = client or make_client()

    start = time.perf_counter()
    try:
        output, completion = await client.chat.completions.create_with_completion(
            model=config.model_name,
            temperature=config.temperature,
            max_tokens=CLASSIFIER_MAX_TOKENS,
            messages=build_messages(email, config),
            response_model=ClassifyOutput,
            max_retries=MAX_RETRIES,
        )
    except InstructorError as exc:
        raise ClassificationError(
            f"model {config.model_name} (prompt {config.version}) returned no valid "
            f"output after {MAX_RETRIES} retry: {exc}"
        ) from exc
    except openai.OpenAIError as exc:
        raise ClassificationError(
            f"API call failed for model {config.model_name}: {exc}"
        ) from exc
    latency_ms = (time.perf_counter() - start) * 1000

    input_tokens, output_tokens = _usage(completion)
    return ClassifyResult(
        **output.model_dump(),
        prompt_version=config.version,
        model_name=config.model_name,
        latency_ms=round(latency_ms, 1),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify a customer support email.")
    parser.add_argument("--prompt", required=True, help="path to a prompt YAML, e.g. prompts/v1.yaml")
    parser.add_argument("email", help="email text to classify")
    args = parser.parse_args(argv)

    config = PromptConfig.from_yaml(args.prompt)
    try:
        result = asyncio.run(classify(args.email, config))
    except (ClassificationError, MissingAPIKeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
