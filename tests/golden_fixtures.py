"""TEST FIXTURES ONLY. Not part of the golden dataset.

These are synthetic, obviously fake cases used to exercise the loader and CLI.
Every input starts with "FIXTURE" so they can never be mistaken for real data.
"""

import copy
import json
from pathlib import Path
from typing import Any

FIXTURE_HEADER: dict[str, Any] = {
    "dataset_version": "v1",
    "created_at": "2026-11-25T00:00:00Z",
    "description": "FIXTURE dataset for unit tests.",
}


def fixture_case(n: int, **overrides: Any) -> dict[str, Any]:
    case = {
        "id": f"case_{n:03d}",
        "input": f"FIXTURE email number {n}.",
        "expected_category": "general",
        "acceptable_categories": [],
        "ideal_summary": f"FIXTURE summary {n}.",
        "expected_difficulty": "easy",
        "tags": [],
        "notes": f"FIXTURE notes {n}.",
    }
    case.update(overrides)
    return case


def fixture_dataset(*cases: dict[str, Any], **header_overrides: Any) -> dict[str, Any]:
    return {**copy.deepcopy(FIXTURE_HEADER), **header_overrides, "cases": list(cases)}


def write_json(path: Path, data: Any) -> Path:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
