"""SQLite-backed trade log manager."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional


# Use project-relative data path so it works from repo root.
DB_PATH = Path.cwd() / "data" / "trading_logs.db"


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            pair TEXT NOT NULL,
            signal_type TEXT NOT NULL,
            price REAL NOT NULL,
            status TEXT NOT NULL,
            duration REAL
        )
        """
    )
    # Add fundamental_context column if it doesn't exist
    cursor = conn.execute("PRAGMA table_info(trades)")
    columns = [col[1] for col in cursor.fetchall()]
    if "fundamental_context" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN fundamental_context TEXT")

    # Trade details table for periodic snapshots
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_details (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trade_id INTEGER NOT NULL,
            timestamp TEXT NOT NULL,
            rsi REAL,
            price REAL,
            screenshot_filename TEXT,
            FOREIGN KEY(trade_id) REFERENCES trades(id)
        )
        """
    )
    # Table to store sentiment analysis from vision model
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS sentiment_store (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            verdict TEXT NOT NULL,
            confidence REAL
        )
        """
    )


def _get_connection() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    _ensure_schema(conn)
    print("DATABASE SUCCESS: SOL-usdt memory is active")
    return conn


def initialize_db() -> None:
    """
    Initialize the SQLite database by ensuring required tables and columns exist.
    This is safe to call multiple times.
    """
    with _get_connection() as conn:
        _ensure_schema(conn)
        conn.commit()


def log_signal(
    pair: str,
    signal_type: str,
    price: float,
    fundamental_context: Optional[str] = None,
    duration: Optional[float] = None,
) -> int:
    """Insert a new trade signal and return its row id."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        cursor = conn.execute(
            """
            INSERT INTO trades (timestamp, pair, signal_type, price, status, duration, fundamental_context)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, pair, signal_type, price, "SIGNALED", duration, fundamental_context),
        )
        conn.commit()
        return cursor.lastrowid


def update_trade_status(trade_id: int, status: str) -> None:
    """Update the status of an existing trade entry."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE trades SET status = ? WHERE id = ?",
            (status, trade_id),
        )
        conn.commit()


def log_trade_detail(trade_id: int, rsi: Optional[float], price: Optional[float], screenshot_filename: Optional[str]) -> None:
    """Store periodic trade detail snapshot."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO trade_details (trade_id, timestamp, rsi, price, screenshot_filename)
            VALUES (?, ?, ?, ?, ?)
            """,
            (trade_id, timestamp, rsi, price, screenshot_filename),
        )
        conn.commit()


def log_sentiment(verdict: str, confidence: float) -> None:
    """Log the sentiment analysis verdict and confidence."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sentiment_store (timestamp, verdict, confidence)
            VALUES (?, ?, ?)
            """,
            (timestamp, verdict, confidence),
        )
        conn.commit()


def get_sentiment_summary(hours: int = 24) -> dict:
    """Get aggregated sentiment data from the last N hours."""
    start_time = datetime.now(timezone.utc) - timedelta(hours=hours)
    start_time_iso = start_time.isoformat()

    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT verdict, confidence FROM sentiment_store WHERE timestamp >= ?",
            (start_time_iso,),
        )
        rows = cursor.fetchall()

    summary = {
        "BULLISH": {"count": 0, "confidence": 0.0},
        "BEARISH": {"count": 0, "confidence": 0.0},
        "NEUTRAL": {"count": 0, "confidence": 0.0},
        "total": 0,
    }
    confidence_sums = {
        "BULLISH": 0.0,
        "BEARISH": 0.0,
        "NEUTRAL": 0.0,
    }

    for verdict, confidence in rows:
        verdict = verdict.upper()
        if verdict in summary:
            summary[verdict]["count"] += 1
            confidence_sums[verdict] += confidence
            summary["total"] += 1

    for verdict in ["BULLISH", "BEARISH", "NEUTRAL"]:
        if summary[verdict]["count"] > 0:
            summary[verdict]["confidence"] = (
                confidence_sums[verdict] / summary[verdict]["count"]
            )

    return summary

