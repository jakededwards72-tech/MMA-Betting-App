from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional
import math


def american_to_decimal(odds: float) -> float:
    if odds == 0:
        raise ValueError("American odds cannot be 0")
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def decimal_to_american(decimal_odds: float) -> float:
    if decimal_odds <= 1:
        raise ValueError("Decimal odds must be > 1")
    if decimal_odds >= 2:
        return round((decimal_odds - 1) * 100)
    return round(-100 / (decimal_odds - 1))


def implied_probability_from_american(odds: float) -> float:
    d = american_to_decimal(odds)
    return 1.0 / d


def american_from_probability(p: float) -> float:
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return decimal_to_american(1.0 / p)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float, float]:
    """Return no-vig p(A), p(B), and market hold using proportional normalization."""
    pa = implied_probability_from_american(odds_a)
    pb = implied_probability_from_american(odds_b)
    s = pa + pb
    if s <= 0:
        raise ValueError("Invalid market probabilities")
    return pa / s, pb / s, s - 1.0


def expected_roi(model_p: float, american_odds: float, push_p: float = 0.0) -> float:
    """Expected return per 1 unit staked; pushes return the stake with zero profit/loss."""
    d = american_to_decimal(american_odds)
    push_p = min(max(float(push_p), 0.0), max(0.0, 1.0 - model_p))
    loss_p = max(0.0, 1.0 - model_p - push_p)
    return model_p * (d - 1.0) - loss_p


def kelly_fraction(model_p: float, american_odds: float, push_p: float = 0.0) -> float:
    d = american_to_decimal(american_odds)
    b = d - 1.0
    push_p = min(max(float(push_p), 0.0), max(0.0, 1.0 - model_p))
    q = max(0.0, 1.0 - model_p - push_p)
    decisive = model_p + q
    if b <= 0 or decisive <= 0:
        return 0.0
    return max(0.0, (b * model_p - q) / (b * decisive))


def break_even_probability(american_odds: float) -> float:
    return implied_probability_from_american(american_odds)


def price_at_probability(p: float) -> float:
    return american_from_probability(p)


def conservative_probability(model_p: float, std_error: float = 0.0, z: float = 1.0) -> float:
    """Haircut model probability by z standard errors for safer price thresholds."""
    return min(max(model_p - z * max(std_error, 0.0), 1e-6), 1 - 1e-6)


@dataclass(frozen=True)
class BetEvaluation:
    selection: str
    market: str
    book: str
    odds: float
    model_p: float
    pricing_p: float
    push_p: float
    market_p: float
    no_vig_market_p: Optional[float]
    edge: float
    ev_roi: float
    full_kelly: float
    suggested_fraction: float
    fair_odds: float
    conservative_fair_odds: float
    verdict: str
    reason: str


def classify_bet(ev_roi: float, edge: float, confidence: float) -> tuple[str, str]:
    """
    confidence is 0..1 data/model confidence, not probability of the bet winning.
    Thresholds are deliberately conservative; users can change them in UI.
    """
    if ev_roi <= -0.05:
        return "BAD BET", "Price is materially worse than the model's break-even price."
    if ev_roi < 0:
        return "PASS", "Negative model EV at this price."
    if confidence < 0.35:
        return "PASS", "Positive point estimate, but data/model confidence is too low."
    if ev_roi >= 0.08 and edge >= 0.05 and confidence >= 0.60:
        return "STRONG VALUE", "Meaningful edge survives a conservative confidence filter."
    if ev_roi >= 0.035 and edge >= 0.025:
        return "VALUE", "Positive EV with a useful probability edge."
    return "THIN / SHOP", "Small edge; line movement or estimation error can erase it."


def evaluate_bet(
    selection: str,
    market: str,
    book: str,
    odds: float,
    model_p: float,
    confidence: float,
    fractional_kelly: float = 0.25,
    no_vig_market_p: Optional[float] = None,
    model_std_error: float = 0.0,
    max_bankroll_fraction: float = 0.03,
    push_p: float = 0.0,
) -> BetEvaluation:
    market_p = break_even_probability(odds)
    push_p = min(max(float(push_p), 0.0), max(0.0, 1.0 - model_p))
    pricing_p = model_p / max(1e-9, 1.0 - push_p)
    comparison_p = no_vig_market_p if no_vig_market_p is not None else market_p
    edge = pricing_p - comparison_p
    ev = expected_roi(model_p, odds, push_p=push_p)
    full = kelly_fraction(model_p, odds, push_p=push_p)
    suggested = min(max_bankroll_fraction, full * max(0.0, fractional_kelly) * max(0.0, min(confidence, 1.0)))
    verdict, reason = classify_bet(ev, edge, confidence)
    conservative_p = conservative_probability(pricing_p, model_std_error, z=1.0)
    return BetEvaluation(
        selection=selection,
        market=market,
        book=book,
        odds=odds,
        model_p=model_p,
        pricing_p=pricing_p,
        push_p=push_p,
        market_p=market_p,
        no_vig_market_p=no_vig_market_p,
        edge=edge,
        ev_roi=ev,
        full_kelly=full,
        suggested_fraction=suggested,
        fair_odds=price_at_probability(pricing_p),
        conservative_fair_odds=price_at_probability(conservative_p),
        verdict=verdict,
        reason=reason,
    )


def best_price(lines: Iterable[dict], selection: str) -> Optional[dict]:
    candidates = [x for x in lines if x.get("selection") == selection]
    if not candidates:
        return None
    # Higher American number is always better for bettor: +150 > +130 and -110 > -130.
    return max(candidates, key=lambda x: float(x["odds"]))
