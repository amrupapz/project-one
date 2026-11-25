"""SQLite storage for eval runs. Every run is stored in full."""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from app.evals.models import CaseResult, EvalRun, RunInfo

SCHEMA_VERSION = 1

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
    notes               TEXT
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
    info = RunInfo.model_validate({c: info_row[c] for c in RUN_COLUMNS})
    return EvalRun(info=info, cases=[_case_from_row(r) for r in case_rows])


@dataclass(frozen=True)
class RunSummary:
    info: RunInfo
    case_count: int
    strict_accuracy: float | None
    acceptable_accuracy: float | None
    errors: int


def list_runs(db_path: str | Path) -> list[RunSummary]:
    """Newest first. Accuracies count errored cases as wrong."""
    with closing(connect(db_path)) as conn:
        rows = conn.execute(
            """
            SELECT r.*,
                   COUNT(c.case_id)        AS case_count,
                   AVG(c.strict_match)     AS strict_accuracy,
                   AVG(c.acceptable_match) AS acceptable_accuracy,
                   COUNT(c.error)          AS errors
            FROM runs r LEFT JOIN case_results c ON c.run_id = r.run_id
            GROUP BY r.run_id
            ORDER BY r.started_at DESC, r.run_id DESC
            """
        ).fetchall()
    return [
        RunSummary(
            info=RunInfo.model_validate({c: row[c] for c in RUN_COLUMNS}),
            case_count=row["case_count"],
            strict_accuracy=row["strict_accuracy"],
            acceptable_accuracy=row["acceptable_accuracy"],
            errors=row["errors"],
        )
        for row in rows
    ]
