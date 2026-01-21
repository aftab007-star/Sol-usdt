import datetime
import json
import pathlib
import uuid


class TradeSessionVault:
    def __init__(self, base_folder: str = "trade_sessions") -> None:
        self.base_path = pathlib.Path(base_folder)
        self.active_path = self.base_path / "ACTIVE"
        self.archive_path = self.base_path / "ARCHIVE"
        self.active_path.mkdir(parents=True, exist_ok=True)
        self.archive_path.mkdir(parents=True, exist_ok=True)

    def _now(self) -> datetime.datetime:
        return datetime.datetime.now(tz=datetime.timezone.utc)

    def _session_file(self) -> pathlib.Path:
        return self.active_path / "session.json"

    def _events_file(self) -> pathlib.Path:
        return self.active_path / "events.jsonl"

    def _load_session(self) -> dict:
        session_file = self._session_file()
        if session_file.exists():
            return json.loads(session_file.read_text(encoding="utf-8"))
        return {}

    def _write_session(self, payload: dict) -> None:
        self._session_file().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def start_or_get_active_session(self, pair: str) -> dict:
        session = self._load_session()
        if session:
            return session

        session = {
            "session_id": uuid.uuid4().hex,
            "pair": pair,
            "status": "ACTIVE",
            "started_at": self._now().isoformat(),
        }
        self._write_session(session)
        return session

    def append_event(
        self,
        pair: str,
        event_type: str,
        ta: dict | None,
        fa: dict | None,
        extra: dict | None,
    ) -> None:
        session = self.start_or_get_active_session(pair)
        payload = {
            "timestamp": self._now().isoformat(),
            "session_id": session.get("session_id"),
            "pair": pair,
            "event_type": event_type,
            "ta": ta,
            "fa": fa,
            "extra": extra,
        }
        with self._events_file().open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

    def close_active_session(self, pair: str, reason: str, extra: dict | None) -> str:
        session = self.start_or_get_active_session(pair)
        session["status"] = "CLOSED"
        session["closed_at"] = self._now().isoformat()
        session["close_reason"] = reason
        session["close_extra"] = extra
        self._write_session(session)

        session_id = session.get("session_id", uuid.uuid4().hex)
        safe_pair = pair.replace("/", "-")
        timestamp = self._now().strftime("%Y%m%dT%H%M%SZ")
        archive_folder = f"{timestamp}_{safe_pair}_{session_id}"
        destination = self.archive_path / archive_folder
        self.active_path.rename(destination)
        self.active_path.mkdir(parents=True, exist_ok=True)
        return str(destination)
