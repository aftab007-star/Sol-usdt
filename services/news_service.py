"""News service using a real provider with simple in-memory caching."""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv


load_dotenv()

NEWS_PROVIDER = os.getenv("NEWS_PROVIDER", "cryptopanic").lower()
CRYPTOPANIC_API_KEY = os.getenv("CRYPTOPANIC_API_KEY", "")
NEWS_LIMIT = int(os.getenv("NEWS_LIMIT", "10") or 10)
NEWS_CACHE_SECONDS = int(os.getenv("NEWS_CACHE_SECONDS", "300") or 300)

_CACHE: Dict[str, object] = {"timestamp": 0.0, "data": []}
LAST_STATUS: Dict[str, object] = {"ok": True, "error": None, "source": NEWS_PROVIDER}


def get_latest_news(symbol: Optional[str] = None) -> List[Dict[str, object]]:
    if NEWS_PROVIDER != "cryptopanic":
        return _handle_failure(f"Unsupported NEWS_PROVIDER: {NEWS_PROVIDER}")

    if _is_cache_valid():
        return _cached_items()

    items, status = _fetch_cryptopanic(symbol=symbol)
    _update_status(status)
    if status["ok"]:
        _update_cache(items)
        return items

    if _cached_items():
        return _cached_items()
    return []


def _fetch_cryptopanic(symbol: Optional[str]) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    if not CRYPTOPANIC_API_KEY:
        return [], {"ok": False, "error": "CRYPTOPANIC_API_KEY not set", "source": "cryptopanic"}

    params = {
        "auth_token": CRYPTOPANIC_API_KEY,
        "public": "true",
        "limit": str(NEWS_LIMIT),
    }
    currency = _normalize_symbol(symbol)
    if currency:
        params["currencies"] = currency

    url = "https://cryptopanic.com/api/v1/posts/?" + urllib.parse.urlencode(params)
    request = urllib.request.Request(url, headers={"User-Agent": "sol-usdt-bot/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as exc:
        return [], {"ok": False, "error": str(exc), "source": "cryptopanic"}

    results = payload.get("results", [])
    items = [_normalize_item(item) for item in results if isinstance(item, dict)]
    return items, {"ok": True, "error": None, "source": "cryptopanic"}


def _normalize_item(item: Dict[str, object]) -> Dict[str, object]:
    source = item.get("source")
    source_name = ""
    if isinstance(source, dict):
        source_name = str(source.get("title") or source.get("name") or "")
    elif isinstance(source, str):
        source_name = source

    currencies = item.get("currencies")
    coins = []
    if isinstance(currencies, list):
        for currency in currencies:
            if isinstance(currency, dict) and currency.get("code"):
                coins.append(str(currency.get("code")))

    tags = []
    raw_tags = item.get("tags")
    if isinstance(raw_tags, list):
        for tag in raw_tags:
            if isinstance(tag, dict) and tag.get("tag"):
                tags.append(str(tag.get("tag")))
            elif isinstance(tag, str):
                tags.append(tag)

    normalized: Dict[str, object] = {
        "title": item.get("title") or "",
        "source": source_name,
        "url": item.get("url") or "",
        "published_at": item.get("published_at") or "",
        "coins": coins,
        "tags": tags,
    }

    if "votes" in item:
        normalized["votes"] = item.get("votes")
    return normalized


def _normalize_symbol(symbol: Optional[str]) -> Optional[str]:
    if not symbol:
        return None
    cleaned = symbol.upper()
    for sep in ("/", "-", "_", ":"):
        if sep in cleaned:
            cleaned = cleaned.split(sep, 1)[0]
            break
    if cleaned.endswith("USDT") and len(cleaned) > 4:
        cleaned = cleaned[: -len("USDT")]
    return cleaned or None


def _is_cache_valid() -> bool:
    data = _cached_items()
    if not data:
        return False
    return (time.time() - float(_CACHE["timestamp"])) < NEWS_CACHE_SECONDS


def _cached_items() -> List[Dict[str, object]]:
    cached = _CACHE.get("data")
    return cached if isinstance(cached, list) else []


def _update_cache(items: List[Dict[str, object]]) -> None:
    _CACHE["data"] = items
    _CACHE["timestamp"] = time.time()


def _update_status(status: Dict[str, object]) -> None:
    LAST_STATUS.update(status)


def _handle_failure(message: str) -> List[Dict[str, object]]:
    _update_status({"ok": False, "error": message, "source": NEWS_PROVIDER})
    if _cached_items():
        return _cached_items()
    return []


class NewsService:
    def get_latest_news(self, limit=10, symbol=None):
        news = get_latest_news(symbol=symbol)
        return news[:limit]


if __name__ == "__main__":
    headlines = get_latest_news()
    print("Latest headlines:")
    for item in headlines[:3]:
        title = item.get("title", "")
        source = item.get("source", "")
        print(f"- {title} ({source})")
