"""
backend/database.py
SQLite connection manager for the FastAPI backend.
Uses the schema defined in backend/schema.sql and the DB produced by the ML pipeline.
"""
import sqlite3
import logging
from pathlib import Path
from typing import Generator

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────────
# Root of the project (one level up from backend/)
ROOT_DIR = Path(__file__).parent.parent
DB_PATH = ROOT_DIR / "precomputed" / "redrob_candidates.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def get_db_path() -> Path:
    return DB_PATH


def init_db(db_path: Path = DB_PATH, schema_path: Path = SCHEMA_PATH) -> None:
    """
    Initialize the SQLite database with the backend schema if tables don't exist yet.
    Safe to run multiple times (uses CREATE TABLE IF NOT EXISTS).
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    with open(schema_path, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    logger.info(f"Database initialized at: {db_path}")


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a new SQLite connection with row_factory set."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # Allow concurrent reads during writes
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def get_db() -> Generator[sqlite3.Connection, None, None]:
    """FastAPI dependency — yields a connection and closes it after the request."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
