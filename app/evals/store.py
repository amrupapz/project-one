"""SQLite storage for eval runs. Every run is stored in full."""

import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.evals.models import CaseResult, EvalRun, RunInfo

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              TEXT PRIMARY KEY,
    started_at          TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    prompt_file_sha256  TEXT NOT NULL,
    classifier_model    TEXT NOT NULL,
    judge_model         TEXT NOT NULL,
    dataset_version     TEXT NOT NULL,
    dataset_sha256      TEXT NOT NULL,
    eval_config_sha256  TEXT NOT NULL,
    concurrency         INTEGER NOT NULL,
    case_limit          INTEGER,          -- NULL = full run
    notes               TEXT,
    judge_enabled       INTEGER NOT NULL DEFAULT 1  -- 0 = run with --no-judge
);
CREATE TABLE IF NOT EXISTS case_results (
    run_id                TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
    position              INTEGER NOT NULL,  -- dataset order
    case_id               TEXT NOT NULL,
    input                 TEXT NOT NULL,
    expected_category     TEXT NOT NULL,
    acceptable_categories TEXT NOT NULL,     -- JSON list
    expected_difficulty   TEXT NOT NULL,
    ideal_summary         TEXT NOT NULL,
    predicted_category    TEXT,
    predicted_summary     TEXT,
    strict_match          INTEGER NOT NULL,
    acceptable_match      INTEGER NOT NULL,
    summary_score         INTEGER,
    judge_reason          TEXT,
    latency_ms            REAL,
    input_tokens          INTEGER,
    output_tokens         INTEGER,
    error                 TEXT,
    PRIMARY KEY (run_id, case_id)
);
"""

RUN_COLUMNS = list(RunInfo.model_fields)
CASE_COLUMNS = list(CaseResult.model_fields)


def backup_path(db_path: Path) -> Path:
    """results/eval.db -> results/eval.db.bak; never overwrites an existing backup."""
    candidate = db_path.with_name(db_path.name + ".bak")
    n = 1
    while candidate.exists():
        candidate = db_path.with_name(f"{db_path.name}.bak.{n}")
        n += 1
    return candidate


def _migrate_v1_to_v2(conn: sqlite3.Connection, db_path: Path) -> None:
    """Add runs.judge_enabled. Existing runs were all judged, so they default to 1.
    The database file is copied first, since it holds every stored run."""
    shutil.copy2(db_path, backup_path(db_path))
    with conn:
        conn.execute("ALTER TABLE runs ADD COLUMN judge_enabled INTEGER NOT NULL DEFAULT 1")
        conn.execute("PRAGMA user_version = 2")


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version == 0:
        conn.executescript(SCHEMA)
        conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
    elif version == 1:
        _migrate_v1_to_v2(conn, db_path)
    elif version != SCHEMA_VERSION:
        conn.close()
        raise RuntimeError(f"{db_path}: schema version {version}, expected {SCHEMA_VERSION}")
    return conn


def save_run(db_path: str | Path, run: EvalRun) -> None:
    """Insert the run and all its case rows in one transaction."""
    run_row = run.info.model_dump(mode="json")
    case_rows = []
    for position, case in enumerate(run.cases):
        row = case.model_dump(mode="json")
        row["acceptable_categories"] = json.dumps(row["acceptable_categories"])
        row["position"] = position
        case_rows.append(row)

    with closing(connect(db_path)) as conn, conn:
        conn.execute(
            f"INSERT INTO runs ({', '.join(RUN_COLUMNS)}) VALUES ({', '.join(':' + c for c in RUN_COLUMNS)})",
            run_row,
        )
        cols = ["position", *CASE_COLUMNS]
        conn.executemany(
            f"INSERT INTO case_results (run_id, {', '.join(cols)}) VALUES (:run_id, {', '.join(':' + c for c in cols)})",
            [{**row, "run_id": run.info.run_id} for row in case_rows],
        )


def _info_from_row(row: sqlite3.Row) -> RunInfo:
    data = {c: row[c] for c in RUN_COLUMNS}
    data["judge_enabled"] = bool(data["judge_enabled"])
    return RunInfo.model_validate(data)


def _case_from_row(row: sqlite3.Row) -> CaseResult:
    data = {c: row[c] for c in CASE_COLUMNS}
    data["acceptable_categories"] = json.loads(data["acceptable_categories"])
    data["strict_match"] = bool(data["strict_match"])
    data["acceptable_match"] = bool(data["acceptable_match"])
    return CaseResult.model_validate(data)


def load_run(db_path: str | Path, run_id: str) -> EvalRun:
    with closing(connect(db_path)) as conn:
        info_row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if info_row is None:
            raise KeyError(f"no run with id {run_id!r} in {db_path}")
        case_rows = conn.execute(
            "SELECT * FROM case_results WHERE run_id = ? ORDER BY position", (run_id,)
        ).fetchall()
    info = _info_from_row(info_row)
    return EvalRun(info=info, cases=[_case_from_row(r) for r in case_rows])


@dataclass(frozen=True)
class RunSummary:
    info: RunInfo
    case_count: int
    strict_accuracy: float | None
    acceptable_accuracy: float | None
    errors: int
    judged: int = 0
    mean_judge_score: float | None = None


def list_runs(db_path: str | Path) -> list[RunSummary]:
    """Newest first. Accuracies count errored cases as wrong."""
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT r.*,
                   COUNT(c.case_id)        AS case_count,
                   AVG(c.strict_match)     AS strict_accuracy,
                   AVG(c.acceptable_match) AS acceptable_accuracy,
                   COUNT(c.error)          AS errors,
                   COUNT(c.summary_score)  AS judged,
                   AVG(c.summary_score)    AS mean_judge_score
            FROM runs r LEFT JOIN case_results c ON c.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.started_at DESC, r.run_id DESC
            """
        ).fetchall()
    return [
        RunSummary(
            info=_info_from_row(row),
            case_count=row["case_count"],
            strict_accuracy=row["strict_accuracy"],
            acceptable_accuracy=row["acceptable_accuracy"],
            errors=row["errors"],
            judged=row["judged"],
            mean_judge_score=row["mean_judge_score"],
        )
        for row in rows
    ]


def find_default_baseline(runs: list[RunSummary], candidate: RunInfo) -> RunInfo | None:
    """The most recent full run started before the candidate on the same dataset file.
    A judged candidate never gets a --no-judge baseline by default. `runs` is newest first."""
    for r in runs:
        info = r.info
        if info.run_id == candidate.run_id or info.is_partial:
            continue
        if info.dataset_sha256 != candidate.dataset_sha256:
            continue
        if (info.started_at, info.run_id) >= (candidate.started_at, candidate.run_id):
            continue
        if candidate.judge_enabled and not info.judge_enabled:
            continue
        return info
    return None
