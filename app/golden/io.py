"""Load and save golden dataset files, reporting every problem with its case id."""

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.golden.models import GoldenCase, GoldenDataset, GoldenHeader, duplicate_messages


@dataclass(frozen=True)
class GoldenIssue:
    location: str  # a case id, "cases[3]" when the id is missing, or "dataset"
    message: str

    def __str__(self) -> str:
        return f"{self.location}: {self.message}"


class GoldenDatasetError(Exception):
    """The dataset file is invalid. `issues` holds every problem found, not just the first."""

    def __init__(self, path: Path, issues: list[GoldenIssue]) -> None:
        self.path = path
        self.issues = issues
        lines = "\n".join(f"  {issue}" for issue in issues)
        super().__init__(f"{path}: {len(issues)} problem(s)\n{lines}")


def _pydantic_issues(location: str, exc: ValidationError) -> list[GoldenIssue]:
    issues = []
    for err in exc.errors():
        field = ".".join(str(part) for part in err["loc"]) or "(case)"
        issues.append(GoldenIssue(location, f"{field}: {err['msg']}"))
    return issues


def _case_location(index: int, raw: Any) -> str:
    if isinstance(raw, dict) and isinstance(raw.get("id"), str) and raw["id"].strip():
        return raw["id"]
    return f"cases[{index}]"


def _parse_json(text: str, duplicate_keys: list[tuple[str | None, str]]) -> Any:
    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        obj: dict[str, Any] = {}
        for key, value in pairs:
            if key in obj:
                owner = obj.get("id") if isinstance(obj.get("id"), str) else None
                duplicate_keys.append((owner, key))
            obj[key] = value
        return obj

    return json.loads(text, object_pairs_hook=hook)


def find_issues(path: str | Path) -> list[GoldenIssue]:
    """Every problem in the file. An empty list means load_golden() will succeed."""
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return [GoldenIssue("dataset", "file is empty")]

    duplicate_keys: list[tuple[str | None, str]] = []
    try:
        raw = _parse_json(text, duplicate_keys)
    except json.JSONDecodeError as exc:
        return [GoldenIssue("dataset", f"invalid JSON at line {exc.lineno}, column {exc.colno}: {exc.msg}")]

    issues = [GoldenIssue(owner or "dataset", f"key {key!r} appears twice in the same object") for owner, key in duplicate_keys]
    if not isinstance(raw, dict):
        return issues + [GoldenIssue("dataset", "top level must be a JSON object")]

    try:
        header = GoldenHeader.model_validate({k: v for k, v in raw.items() if k != "cases"})
    except ValidationError as exc:
        issues += _pydantic_issues("dataset", exc)
    else:
        if path.stem != header.dataset_version:
            issues.append(GoldenIssue("dataset", f"dataset_version {header.dataset_version!r} does not match file name {path.stem!r}"))

    raw_cases = raw.get("cases")
    if raw_cases is None:
        return issues + [GoldenIssue("dataset", "cases: Field required (use [] for no cases)")]
    if not isinstance(raw_cases, list):
        return issues + [GoldenIssue("dataset", "cases: must be a list")]

    id_input_pairs: list[tuple[str, str | None]] = []
    for i, raw_case in enumerate(raw_cases):
        location = _case_location(i, raw_case)
        try:
            GoldenCase.model_validate(raw_case)
        except ValidationError as exc:
            issues += _pydantic_issues(location, exc)
        # Check duplicates even on otherwise-invalid cases, so one pass shows everything.
        if isinstance(raw_case, dict) and isinstance(raw_case.get("id"), str):
            text = raw_case.get("input")
            id_input_pairs.append((raw_case["id"], text if isinstance(text, str) else None))
    issues += [GoldenIssue(cid, msg) for cid, msg in duplicate_messages(id_input_pairs)]
    return issues


def load_golden(path: str | Path) -> GoldenDataset:
    """Load and validate a golden dataset. Raises GoldenDatasetError listing every problem."""
    path = Path(path)
    issues = find_issues(path)
    if issues:
        raise GoldenDatasetError(path, issues)
    return GoldenDataset.model_validate_json(path.read_text(encoding="utf-8"))


def dump_golden(dataset: GoldenDataset) -> str:
    """Stable on-disk format: indent 2, keys in model field order, non-ASCII kept readable."""
    return json.dumps(dataset.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n"


def save_golden(dataset: GoldenDataset, path: str | Path) -> None:
    """Write atomically, so an interrupted save can't leave a half-written file."""
    path = Path(path)
    GoldenDataset.model_validate(dataset.model_dump())  # re-check invariants before touching disk
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(dump_golden(dataset))
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise
