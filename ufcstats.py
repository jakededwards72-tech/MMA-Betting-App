from __future__ import annotations

from dataclasses import asdict
from datetime import date, datetime
import re
from typing import Optional
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from .models import FighterProfile

BASE = "http://ufcstats.com"
HEADERS = {"User-Agent": "Mozilla/5.0 MMAQuantEngine/1.0"}


class UFCStatsError(RuntimeError):
    pass


def _get(url: str, timeout: int = 20) -> BeautifulSoup:
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException as e:
        raise UFCStatsError(f"Unable to fetch UFCStats: {e}") from e
    return BeautifulSoup(r.text, "html.parser")


def _num(text: str, default=None):
    m = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(m.group()) if m else default


def _pct(text: str, default=None):
    v = _num(text, None)
    return default if v is None else v / 100.0


def _height_to_inches(text: str) -> Optional[float]:
    m = re.search(r"(\d+)\s*'\s*(\d+)\s*\"?", text or "")
    if not m:
        return None
    return int(m.group(1)) * 12 + int(m.group(2))


def _dob_to_age(text: str) -> Optional[float]:
    text = (text or "").strip()
    if not text or text == "--":
        return None
    for fmt in ("%b %d, %Y", "%B %d, %Y"):
        try:
            d = datetime.strptime(text, fmt).date()
            today = date.today()
            return (today - d).days / 365.2425
        except ValueError:
            pass
    return None


def search_fighter(name: str, timeout: int = 20) -> list[dict]:
    """Search UFCStats index. Returns close substring matches with profile URLs."""
    q = " ".join(name.lower().split())
    if not q:
        return []
    first = q[0]
    url = f"{BASE}/statistics/fighters?char={first}&page=all"
    soup = _get(url, timeout=timeout)
    matches = []
    for row in soup.select("tr.b-statistics__table-row"):
        links = row.select("a.b-link.b-link_style_black")
        if not links:
            continue
        full = " ".join(a.get_text(" ", strip=True) for a in links[:3])
        full_norm = " ".join(full.lower().split())
        if q in full_norm or all(tok in full_norm for tok in q.split()):
            href = links[0].get("href")
            if href:
                matches.append({"name": full.strip(), "url": href})
    return matches[:20]


def parse_fighter_page(url: str, timeout: int = 20) -> FighterProfile:
    soup = _get(url, timeout=timeout)
    name_el = soup.select_one("span.b-content__title-highlight")
    if not name_el:
        raise UFCStatsError("Could not parse fighter name from UFCStats page")
    name = name_el.get_text(" ", strip=True)

    # The UFCStats profile uses label/value text inside list-item nodes.
    values = {}
    for item in soup.select("li.b-list__box-list-item"):
        text = " ".join(item.stripped_strings)
        if ":" not in text:
            continue
        k, v = text.split(":", 1)
        values[k.strip().lower()] = v.strip()

    def v(key: str, default=""):
        return values.get(key.lower(), default)

    profile = FighterProfile(
        name=name,
        age=_dob_to_age(v("dob")),
        height_in=_height_to_inches(v("height")),
        reach_in=_num(v("reach"), None),
        stance=v("stance", "Unknown") or "Unknown",
        slpm=_num(v("slpm"), None),
        sapm=_num(v("sapm"), None),
        str_acc=_pct(v("str. acc."), None),
        str_def=_pct(v("str. def."), None),
        td_avg=_num(v("td avg."), None),
        td_acc=_pct(v("td acc."), None),
        td_def=_pct(v("td def."), None),
        sub_avg=_num(v("sub. avg."), None),
    )

    # Parse UFC fight history to enrich experience and outcome-style priors.
    rows = soup.select("tr.b-fight-details__table-row.b-fight-details__table-row__hover")
    wins = losses = decisions_won = ko_wins = sub_wins = finish_losses = 0
    total_minutes = 0.0
    last_date: Optional[date] = None
    for row in rows:
        cols = row.select("td.b-fight-details__table-col")
        if not cols:
            continue
        cells = [" ".join(c.stripped_strings) for c in cols]
        result = cells[0].lower() if cells else ""
        method = cells[7].upper() if len(cells) > 7 else ""
        rnd = int(_num(cells[8], 1) or 1) if len(cells) > 8 else 1
        t = cells[9] if len(cells) > 9 else "0:00"
        mt = re.match(r"(\d+):(\d+)", t)
        if mt:
            total_minutes += max(0, rnd - 1) * 5 + int(mt.group(1)) + int(mt.group(2)) / 60
        if result.startswith("win"):
            wins += 1
            if "KO/TKO" in method:
                ko_wins += 1
            elif "SUB" in method:
                sub_wins += 1
            elif "DEC" in method:
                decisions_won += 1
        elif result.startswith("loss"):
            losses += 1
            if "KO/TKO" in method or "SUB" in method:
                finish_losses += 1

        # Event date is often contained in the event cell text as 'Mon. DD, YYYY'.
        if len(cells) > 6:
            dm = re.search(r"([A-Z][a-z]{2}\.\s+\d{2},\s+\d{4})", cells[6])
            if dm:
                try:
                    d = datetime.strptime(dm.group(1), "%b. %d, %Y").date()
                    last_date = max(last_date, d) if last_date else d
                except ValueError:
                    pass

    total_wins = max(wins, 1)
    total_losses = max(losses, 1)
    profile.ufc_fights = wins + losses
    profile.minutes = total_minutes if total_minutes > 0 else None
    profile.ko_win_rate = ko_wins / total_wins if wins else None
    profile.sub_win_rate = sub_wins / total_wins if wins else None
    profile.decision_win_rate = decisions_won / total_wins if wins else None
    profile.finish_loss_rate = finish_losses / total_losses if losses else None
    if last_date:
        profile.days_since_last_fight = (date.today() - last_date).days
    return profile


def get_fighter_by_name(name: str, timeout: int = 20) -> tuple[FighterProfile, str]:
    matches = search_fighter(name, timeout=timeout)
    if not matches:
        raise UFCStatsError(f"No UFCStats fighter match found for '{name}'")
    # Prefer exact token-normalized match; otherwise first search result.
    q = " ".join(name.lower().split())
    chosen = matches[0]
    for m in matches:
        if " ".join(m["name"].lower().split()) == q:
            chosen = m
            break
    return parse_fighter_page(chosen["url"], timeout=timeout), chosen["url"]
