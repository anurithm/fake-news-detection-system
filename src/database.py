"""
database.py
-----------
SQLite persistence layer for storing and retrieving prediction history.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "predictions.db"


def _get_connection() -> sqlite3.Connection:
    """Open (or create) the SQLite database and return a connection."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create the predictions table if it does not already exist."""
    try:
        with _get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp   TEXT    NOT NULL,
                    article     TEXT    NOT NULL,
                    prediction  TEXT    NOT NULL,
                    confidence  REAL,
                    model_name  TEXT    NOT NULL
                )
                """
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.error("Failed to initialise database: %s", exc)
        raise


def save_prediction(
    article: str,
    prediction: str,
    confidence: Optional[float],
    model_name: str,
) -> None:
    """
    Persist a single prediction record.

    Parameters
    ----------
    article    : raw text submitted by the user (truncated to 1000 chars for storage)
    prediction : 'REAL' or 'FAKE'
    confidence : float in [0, 1] or None
    model_name : name of the model used
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    short_article = article[:1000] if len(article) > 1000 else article
    try:
        init_db()
        with _get_connection() as conn:
            conn.execute(
                "INSERT INTO predictions (timestamp, article, prediction, confidence, model_name) "
                "VALUES (?, ?, ?, ?, ?)",
                (timestamp, short_article, prediction, confidence, model_name),
            )
            conn.commit()
    except sqlite3.Error as exc:
        logger.error("Failed to save prediction: %s", exc)
        raise


def get_predictions(limit: int = 50) -> list[dict]:
    """Return the most recent `limit` predictions as a list of dicts."""
    try:
        init_db()
        with _get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
    except sqlite3.Error as exc:
        logger.error("Failed to retrieve predictions: %s", exc)
        raise


def clear_history() -> int:
    """Delete all prediction records. Returns the number of rows deleted."""
    try:
        init_db()
        with _get_connection() as conn:
            cursor = conn.execute("DELETE FROM predictions")
            conn.commit()
            return cursor.rowcount
    except sqlite3.Error as exc:
        logger.error("Failed to clear prediction history: %s", exc)
        raise
