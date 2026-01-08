"""Mock news service returning hardcoded headlines."""
from __future__ import annotations

from typing import Dict, List, Optional


class MockNewsService:
    def __init__(self) -> None:
        self._headlines: List[Dict[str, object]] = [
            {"title": "SOL institutional adoption rises", "url": "https://example.com/1", "votes": {}},
            {"title": "Major DeFi protocol integrates Solana", "url": "https://example.com/2", "votes": {}},
            {"title": "Validators report record uptime on Solana", "url": "https://example.com/3", "votes": {}},
            {"title": "Network congestion reported amid NFT mint", "url": "https://example.com/4", "votes": {}},
            {"title": "Concerns over potential exploit in legacy contract", "url": "https://example.com/5", "votes": {}},
        ]

    def get_latest_news(self, symbol: Optional[str] = None) -> List[Dict[str, object]]:
        # Symbol is ignored in mock; returns fixed set.
        return self._headlines


_service = MockNewsService()


def get_latest_news(symbol: Optional[str] = None) -> List[Dict[str, object]]:
    return _service.get_latest_news(symbol)
