from __future__ import annotations

from dataclasses import asdict
import math
from typing import Iterable

import pandas as pd

from betting import evaluate_bet, devig_two_way
from models import SimulationSummary


def simulation_table(summary: SimulationSummary) -> pd.DataFrame:
    rows = [
        (summary.fighter_a, "Win", summary.p_a_win),
        (summary.fighter_b, "Win", summary.p_b_win),
        (summary.fighter_a, "KO/TKO", summary.p_a_ko),
        (summary.fighter_a, "Submission", summary.p_a_sub),
        (summary.fighter_a, "Decision", summary.p_a_dec),
        (summary.fighter_b, "KO/TKO", summary.p_b_ko),
        (summary.fighter_b, "Submission", summary.p_b_sub),
        (summary.fighter_b, "Decision", summary.p_b_dec),
        ("Fight", "Goes distance", summary.p_distance),
        ("Fight", "Over 1.5", summary.p_over_1_5),
        ("Fight", "Over 2.5", summary.p_over_2_5),
        ("Fight", "Over 4.5", summary.p_over_4_5),
    ]
    df = pd.DataFrame(rows, columns=["Selection", "Outcome", "Probability"])
    df = df[df["Probability"].notna()].copy()
    df["Fair American"] = df["Probability"].apply(_fair_american)
    return df


def _fair_american(p: float) -> int:
    p = min(max(float(p), 1e-8), 1-1e-8)
    if p >= 0.5:
        return int(round(-100 * p/(1-p)))
    return int(round(100 * (1-p)/p))


def evaluate_moneyline_market(summary: SimulationSummary, lines: list[dict], fractional_kelly: float = 0.25, max_bankroll_fraction: float = 0.03) -> pd.DataFrame:
    a = summary.fighter_a
    b = summary.fighter_b
    rows = []
    # Pair each bookmaker's two sides so we can remove vig book-by-book.
    books = sorted({r.get("book", "Unknown") for r in lines})
    for book in books:
        br = [r for r in lines if r.get("book", "Unknown") == book]
        side_a = next((r for r in br if _norm(r.get("selection", "")) == _norm(a)), None)
        side_b = next((r for r in br if _norm(r.get("selection", "")) == _norm(b)), None)
        nv_a = nv_b = None
        hold = None
        if side_a and side_b:
            nv_a, nv_b, hold = devig_two_way(float(side_a["odds"]), float(side_b["odds"]))
        for side, p, nv in ((side_a, summary.p_a_win, nv_a), (side_b, summary.p_b_win, nv_b)):
            if not side:
                continue
            ev = evaluate_bet(
                selection=side["selection"],
                market="Moneyline",
                book=book,
                odds=float(side["odds"]),
                model_p=p,
                confidence=summary.data_confidence,
                fractional_kelly=fractional_kelly,
                no_vig_market_p=nv,
                model_std_error=summary.model_uncertainty,
                max_bankroll_fraction=max_bankroll_fraction,
                push_p=summary.p_draw,
            )
            d = asdict(ev)
            d["hold"] = hold
            d["last_update"] = side.get("last_update")
            rows.append(d)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    order = ["STRONG VALUE", "VALUE", "THIN / SHOP", "PASS", "BAD BET"]
    df["_rank"] = df["verdict"].map({v:i for i,v in enumerate(order)}).fillna(9)
    return df.sort_values(["_rank", "ev_roi", "odds"], ascending=[True, False, False]).drop(columns="_rank")


def evaluate_manual_markets(summary: SimulationSummary, market_rows: Iterable[dict], fractional_kelly: float = 0.25, max_bankroll_fraction: float = 0.03) -> pd.DataFrame:
    keymap = {
        f"{summary.fighter_a} ML": "A ML",
        f"{summary.fighter_b} ML": "B ML",
        "Fight Goes Distance": "Fight Goes Distance",
        "Fight Doesn't Go Distance": "Fight Doesn't Go Distance",
        "Over 1.5": "Over 1.5",
        "Under 1.5": "Under 1.5",
        "Over 2.5": "Over 2.5",
        "Under 2.5": "Under 2.5",
        "Over 4.5": "Over 4.5",
        "Under 4.5": "Under 4.5",
        f"{summary.fighter_a} by KO/TKO": "A by KO/TKO",
        f"{summary.fighter_a} by Submission": "A by Submission",
        f"{summary.fighter_a} by Decision": "A by Decision",
        f"{summary.fighter_b} by KO/TKO": "B by KO/TKO",
        f"{summary.fighter_b} by Submission": "B by Submission",
        f"{summary.fighter_b} by Decision": "B by Decision",
    }
    rows = []
    for r in market_rows:
        selection = str(r.get("selection", "")).strip()
        if selection not in keymap:
            continue
        try:
            odds = float(r.get("odds"))
        except (TypeError, ValueError):
            continue
        p = summary.outcome_probability(keymap[selection])
        if isinstance(p, float) and math.isnan(p):
            continue
        push_p = summary.p_draw if keymap[selection] in ("A ML", "B ML") else 0.0
        ev = evaluate_bet(
            selection=selection,
            market=str(r.get("market", "Manual")),
            book=str(r.get("book", "Manual")),
            odds=odds,
            model_p=p,
            confidence=summary.data_confidence,
            fractional_kelly=fractional_kelly,
            model_std_error=summary.model_uncertainty,
            max_bankroll_fraction=max_bankroll_fraction,
            push_p=push_p,
        )
        rows.append(asdict(ev))
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("ev_roi", ascending=False)


def _norm(s: str) -> str:
    return " ".join((s or "").lower().replace("-", " ").replace("'", "").split())


def expert_breakdown(summary: SimulationSummary) -> dict[str, str]:
    """Generate a model-grounded fight breakdown from simulation outputs/diagnostics only."""
    d = summary.diagnostics
    a, b = summary.fighter_a, summary.fighter_b
    strike = d.get("strike_edge_a", 0.0)
    grapple = d.get("grapple_edge_a", 0.0)
    sub = d.get("submission_edge_a", 0.0)
    latent = d.get("latent_edge_a", 0.0)

    if abs(strike) < 0.25:
        standup = "The striking phase projects close; neither side has a large interaction edge after offense is matched to the opponent's defense."
    elif strike > 0:
        standup = f"{a} projects as the cleaner striking side in the model, with the advantage coming from the offense/defense interaction rather than raw volume alone."
    else:
        standup = f"{b} projects as the cleaner striking side in the model, with the advantage coming from the offense/defense interaction rather than raw volume alone."

    if abs(grapple) < 0.25:
        wrestling = "Takedown access is not a major separator on the current inputs; the fight is less dependent on one fighter consistently forcing a grappling phase."
    elif grapple > 0:
        wrestling = f"{a} has the stronger takedown/control interaction, so {b}'s defensive wrestling is one of the main swing variables."
    else:
        wrestling = f"{b} has the stronger takedown/control interaction, so {a}'s defensive wrestling is one of the main swing variables."

    if abs(sub) >= 0.35:
        fav = a if sub > 0 else b
        submissions = f"Submission equity leans toward {fav}; that edge is modeled separately from takedown success, so it reflects danger after grappling access is established rather than treating every takedown as equivalent."
    else:
        submissions = "The submission differential is not large enough to drive the matchup by itself."

    def path(name: str, ko: float, su: float, dec: float) -> str:
        methods = [(ko, "KO/TKO"), (su, "submission"), (dec, "decision")]
        methods.sort(reverse=True)
        top_p, top = methods[0]
        second_p, second = methods[1]
        return f"{name}'s most common simulated win condition is {top} ({top_p:.1%}), followed by {second} ({second_p:.1%})."

    spread = abs(summary.p_a_win - summary.p_b_win)
    if summary.data_confidence < 0.45:
        volatility = "The largest concern is input uncertainty. Sparse UFC samples can make apparent style edges unstable, so this matchup requires a larger betting margin than the point estimate suggests."
    elif spread < 0.08:
        volatility = "This is a high-sensitivity matchup: the win probabilities are close enough that small changes in assumptions or price can flip the betting conclusion."
    elif summary.p_distance < 0.38:
        volatility = "Finish variance is high. A large share of the distribution ends before the scorecards, which widens single-fight uncertainty even if one side has a measurable model edge."
    else:
        volatility = "The result distribution is comparatively stable for MMA, but single-fight variance remains substantial and should be handled through price discipline rather than confidence language."

    if summary.p_distance >= 0.60:
        pace = "The model expects a decision-heavy fight shape. Round-winning consistency matters more than one isolated finishing sequence."
    elif summary.p_distance <= 0.35:
        pace = "The model expects a finish-heavy fight shape. Method-of-victory and under/distance markets may carry more information than the moneyline alone."
    else:
        pace = "The fight has a mixed finish/decision distribution; neither a pure distance script nor a pure chaos script dominates."

    if latent > 0.20:
        overall = f"Across phases, the composite matchup score leans {a}, but the betting decision still depends on the offered price."
    elif latent < -0.20:
        overall = f"Across phases, the composite matchup score leans {b}, but the betting decision still depends on the offered price."
    else:
        overall = "The composite matchup score is close to neutral, so sportsbook price and uncertainty are especially important."

    return {
        "Overall fight shape": overall + " " + pace,
        "Striking": standup,
        "Wrestling / control": wrestling,
        "Submissions": submissions,
        f"{a} path to win": path(a, summary.p_a_ko, summary.p_a_sub, summary.p_a_dec),
        f"{b} path to win": path(b, summary.p_b_ko, summary.p_b_sub, summary.p_b_dec),
        "Variance / confidence": volatility,
    }
