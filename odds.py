from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import requests


ODDS_API_BASE = "https://api.the-odds-api.com/v4"
SPORT_KEY = "mma_mixed_martial_arts"


class OddsAPIError(RuntimeError):
    pass


def fetch_mma_moneylines(api_key: str, regions: str = "us,us2", timeout: int = 20) -> list[dict]:
    if not api_key:
        raise OddsAPIError("Missing The Odds API key")
    url = f"{ODDS_API_BASE}/sports/{SPORT_KEY}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": "h2h",
        "oddsFormat": "american",
        "dateFormat": "iso",
    }
    r = requests.get(url, params=params, timeout=timeout)
    if r.status_code != 200:
        raise OddsAPIError(f"Odds API HTTP {r.status_code}: {r.text[:300]}")
    return r.json()


def flatten_moneylines(events: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for event in events:
        event_id = event.get("id")
        commence = event.get("commence_time")
        home = event.get("home_team")
        away = event.get("away_team")
        for book in event.get("bookmakers", []):
            title = book.get("title", book.get("key", "Unknown"))
            book_key = book.get("key", "")
            updated = book.get("last_update")
            for market in book.get("markets", []):
                if market.get("key") != "h2h":
                    continue
                for outcome in market.get("outcomes", []):
                    rows.append({
                        "event_id": event_id,
                        "commence_time": commence,
                        "fighter_a": home,
                        "fighter_b": away,
                        "selection": outcome.get("name"),
                        "odds": outcome.get("price"),
                        "book": title,
                        "book_key": book_key,
                        "last_update": updated or market.get("last_update"),
                    })
    return rows


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").replace("'", "").split())


def find_event_lines(rows: list[dict], fighter_a: str, fighter_b: str) -> list[dict]:
    a, b = _norm(fighter_a), _norm(fighter_b)
    out = []
    for row in rows:
        names = {_norm(row.get("fighter_a", "")), _norm(row.get("fighter_b", ""))}
        if {a, b} == names:
            out.append(row)
    return out


def group_best_moneyline(lines: list[dict], fighter_a: str, fighter_b: str) -> dict[str, Optional[dict]]:
    result: dict[str, Optional[dict]] = {fighter_a: None, fighter_b: None}
    for fighter in (fighter_a, fighter_b):
        candidates = [r for r in lines if _norm(r.get("selection", "")) == _norm(fighter) and r.get("odds") is not None]
        if candidates:
            result[fighter] = max(candidates, key=lambda x: float(x["odds"]))
    return result


def event_pairs(lines: list[dict]) -> list[tuple[str, str, str, str]]:
    """Unique (event_id, commence_time, fighter_a, fighter_b) pairs."""
    seen = set()
    result = []
    for r in lines:
        key = (r.get("event_id"), r.get("commence_time"), r.get("fighter_a"), r.get("fighter_b"))
        if key not in seen:
            seen.add(key)
            result.append(key)
    return result
