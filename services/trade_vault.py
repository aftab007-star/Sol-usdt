from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class TradeVault:
    def __init__(self, base_path: str = "./trade_history") -> None:
        self.base_path = Path(base_path)
        self.unassigned_path = self.base_path / "_unassigned"
        self._logger = logging.getLogger("solana_bot_output")

    def create_trade_folder(self, trade_id: int, symbol: str, ts: Optional[object]) -> str:
        safe_symbol = str(symbol).replace("/", "") or "UNKNOWN"
        ts_label = self._format_ts(ts)
        folder = self.base_path / f"trade_{trade_id}_{ts_label}_{safe_symbol}"
        self._ensure_dirs(folder)
        return str(folder)

    def get_active_trade_folder(self, db_manager) -> Optional[str]:
        try:
            open_trades = db_manager.get_open_trades()
        except Exception:
            self._logger.exception("TradeVault: failed to load open trades")
            return None

        if not open_trades:
            return None

        latest_trade = max(open_trades, key=lambda row: row[0])
        trade_id, pair, *_ = latest_trade
        existing = self._find_trade_folder(trade_id)
        if existing:
            return str(existing)

        ts = None
        pair_value = pair
        try:
            metadata = db_manager.get_trade_metadata(trade_id)
            if metadata:
                ts, pair_value = metadata
        except Exception:
            self._logger.exception("TradeVault: failed to load trade metadata for %s", trade_id)

        return self.create_trade_folder(trade_id, str(pair_value or ""), ts)

    def get_trade_folder(self, trade_id: int) -> Optional[str]:
        existing = self._find_trade_folder(trade_id)
        return str(existing) if existing else None

    def get_unassigned_folder(self) -> str:
        self._ensure_dirs(self.unassigned_path)
        return str(self.unassigned_path)

    def save_screenshot(self, image_bytes: bytes, filename: str, trade_folder: str) -> Optional[str]:
        if self._is_complete(trade_folder):
            return None
        try:
            folder = Path(trade_folder) / "screenshots"
            folder.mkdir(parents=True, exist_ok=True)
            ts_label = self._format_ts(None)
            base_name = os.path.basename(filename) or "image.png"
            path = folder / f"sc_{ts_label}_{base_name}"
            with open(path, "wb") as handle:
                handle.write(image_bytes)
            return str(path)
        except OSError:
            self._logger.exception("TradeVault: failed to save screenshot")
            return None

    def save_fundamental_text(self, text: str, trade_folder: str) -> Optional[str]:
        if self._is_complete(trade_folder):
            return None
        try:
            folder = Path(trade_folder) / "fundamentals"
            folder.mkdir(parents=True, exist_ok=True)
            ts_label = self._format_ts(None)
            path = folder / f"text_{ts_label}.txt"
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return ts_label
        except OSError:
            self._logger.exception("TradeVault: failed to save fundamental text")
            return None

    def save_snapshot(self, name: str, payload: dict, trade_folder: str) -> Optional[str]:
        if self._is_complete(trade_folder):
            return None
        try:
            trade_path = Path(trade_folder)
            ts_label = self._format_ts(payload.get("timestamp"))
            if name == "update_4h":
                folder = trade_path / "updates"
                folder.mkdir(parents=True, exist_ok=True)
                index = self._next_update_index(folder)
                path = folder / f"update_4h_{index}.json"
            elif name in {"fundamental_text", "fundamental_image"}:
                folder = trade_path / "fundamentals"
                folder.mkdir(parents=True, exist_ok=True)
                prefix = "text" if name == "fundamental_text" else "image"
                path = folder / f"{prefix}_{ts_label}.json"
            elif name in {"buy", "sell", "summary", "meta"}:
                path = trade_path / f"{name}.json"
                if path.exists():
                    path = trade_path / f"{name}_{ts_label}.json"
            else:
                path = trade_path / f"{name}_{ts_label}.json"

            self._write_json(path, payload)
            return str(path)
        except OSError:
            self._logger.exception("TradeVault: failed to save snapshot %s", name)
            return None

    def finalize_trade(self, trade_folder: str, final_payload: dict) -> None:
        if self._is_complete(trade_folder):
            return
        try:
            trade_path = Path(trade_folder)
            screenshots = trade_path / "screenshots"
            fundamentals = trade_path / "fundamentals"
            if "screenshot_count" not in final_payload:
                final_payload["screenshot_count"] = self._count_files(screenshots, "sc_")
            if "text_fundamental_count" not in final_payload:
                final_payload["text_fundamental_count"] = self._count_files(fundamentals, "text_", ".txt")
            self.save_snapshot("summary", final_payload, trade_folder)
            complete_path = trade_path / "complete.flag"
            with open(complete_path, "w", encoding="utf-8") as handle:
                handle.write(self._format_ts(None))
        except OSError:
            self._logger.exception("TradeVault: failed to finalize trade")

    def _write_json(self, path: Path, payload: dict) -> None:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _ensure_dirs(self, folder: Path) -> None:
        try:
            folder.mkdir(parents=True, exist_ok=True)
            (folder / "updates").mkdir(parents=True, exist_ok=True)
            (folder / "screenshots").mkdir(parents=True, exist_ok=True)
            (folder / "fundamentals").mkdir(parents=True, exist_ok=True)
        except OSError:
            self._logger.exception("TradeVault: failed to prepare directories")

    def _find_trade_folder(self, trade_id: int) -> Optional[Path]:
        if not self.base_path.exists():
            return None
        prefix = f"trade_{trade_id}_"
        matches = [
            entry for entry in self.base_path.iterdir()
            if entry.is_dir() and entry.name.startswith(prefix)
        ]
        if not matches:
            return None
        return max(matches, key=lambda entry: entry.stat().st_mtime)

    def _next_update_index(self, folder: Path) -> int:
        try:
            existing = [item for item in folder.iterdir() if item.name.startswith("update_4h_")]
            return len(existing) + 1
        except OSError:
            self._logger.exception("TradeVault: failed to count updates")
            return 1

    def _count_files(self, folder: Path, prefix: str, suffix: str = "") -> int:
        try:
            if not folder.exists():
                return 0
            return sum(
                1 for item in folder.iterdir()
                if item.is_file() and item.name.startswith(prefix) and item.name.endswith(suffix)
            )
        except OSError:
            self._logger.exception("TradeVault: failed to count files in %s", folder)
            return 0

    def _is_complete(self, trade_folder: str) -> bool:
        try:
            return Path(trade_folder, "complete.flag").exists()
        except OSError:
            return False

    def _format_ts(self, value: Optional[object]) -> str:
        if isinstance(value, str) and value:
            try:
                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return dt.strftime("%Y%m%d_%H%M%S")
            except ValueError:
                return value.replace(":", "").replace("-", "").replace(" ", "_")[:15]
        if isinstance(value, (int, float)):
            dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
            return dt.strftime("%Y%m%d_%H%M%S")
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
