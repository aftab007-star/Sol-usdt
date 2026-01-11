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
    if "pnl" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN pnl REAL")
    if "pnl_pct" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN pnl_pct REAL")
    if "closed_price" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN closed_price REAL")
    if "closed_at" not in columns:
        conn.execute("ALTER TABLE trades ADD COLUMN closed_at TEXT")

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
    cursor = conn.execute("PRAGMA table_info(sentiment_store)")
    columns = [col[1] for col in cursor.fetchall()]
    if "source" not in columns:
        conn.execute("ALTER TABLE sentiment_store ADD COLUMN source TEXT")
    if "raw_text" not in columns:
        conn.execute("ALTER TABLE sentiment_store ADD COLUMN raw_text TEXT")
    if "channel_id" not in columns:
        conn.execute("ALTER TABLE sentiment_store ADD COLUMN channel_id TEXT")
    if "author" not in columns:
        conn.execute("ALTER TABLE sentiment_store ADD COLUMN author TEXT")


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


def close_trade(trade_id: int, status: str, pnl: float) -> None:
    """Update trade status and persist pnl."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE trades SET status = ?, pnl = ? WHERE id = ?",
            (status, pnl, trade_id),
        )
        conn.commit()


def close_trade_with_details(
    trade_id: int,
    status: str,
    pnl_usdt: float,
    pnl_pct: float,
    closed_price: float,
    closed_at: str,
) -> None:
    """Update trade status and persist pnl/close details."""
    with _get_connection() as conn:
        conn.execute(
            "UPDATE trades SET status = ?, pnl = ?, pnl_pct = ?, closed_price = ?, closed_at = ? WHERE id = ?",
            (status, pnl_usdt, pnl_pct, closed_price, closed_at, trade_id),
        )
        conn.commit()


def get_open_trades() -> list[tuple]:
    """Return trades considered open for paper tracking."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT id, pair, signal_type, price, status FROM trades "
            "WHERE status = 'OPEN'"
        )
        return cursor.fetchall()


def get_trade_metadata(trade_id: int) -> Optional[tuple]:
    """Return (timestamp, pair) for a trade id if available."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT timestamp, pair FROM trades WHERE id = ?",
            (trade_id,),
        )
        row = cursor.fetchone()
    return (row[0], row[1]) if row else None


def get_trades_count_today() -> int:
    """Return count of BUY trades created today (UTC)."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE signal_type = 'BUY' AND substr(timestamp, 1, 10) = date('now')"
        )
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def get_realized_pnl_today_usdt() -> float:
    """Return realized PnL for today from closed trades (UTC)."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades "
            "WHERE status IN ('CLOSED_TP', 'CLOSED_SL') AND "
            "closed_at IS NOT NULL AND substr(closed_at, 1, 10) = date('now')"
        )
        row = cursor.fetchone()
        return float(row[0]) if row else 0.0


def get_last_buy_timestamp() -> Optional[str]:
    """Return ISO timestamp string for the most recent BUY."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT timestamp FROM trades WHERE signal_type = 'BUY' ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None


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
    store_text_sentiment(
        raw_text=None,
        verdict=verdict,
        confidence=confidence,
        source="image",
        channel_id=None,
        author=None,
    )


def store_text_sentiment(
    raw_text: Optional[str],
    verdict: str,
    confidence: Optional[float],
    source: str = "text",
    channel_id: Optional[str] = None,
    author: Optional[str] = None,
) -> None:
    """Store a sentiment record with optional source metadata."""
    timestamp = datetime.now(timezone.utc).isoformat()
    with _get_connection() as conn:
        conn.execute(
            """
            INSERT INTO sentiment_store (timestamp, verdict, confidence, source, raw_text, channel_id, author)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (timestamp, verdict, confidence, source, raw_text, channel_id, author),
        )
        conn.commit()


def get_latest_sentiment() -> Optional[dict]:
    """Return the most recent stored sentiment record, if any."""
    with _get_connection() as conn:
        cursor = conn.execute(
            "SELECT timestamp, verdict, confidence, source FROM sentiment_store ORDER BY id DESC LIMIT 1"
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"timestamp": row[0], "verdict": row[1], "confidence": row[2], "source": row[3]}


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
