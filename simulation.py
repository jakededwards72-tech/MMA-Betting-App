from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import math
import numpy as np

from .models import FighterProfile, FightContext, SimulationSummary


PRIORS = {
    "slpm": 3.8,
    "sapm": 3.8,
    "str_acc": 0.46,
    "str_def": 0.54,
    "td_avg": 1.45,
    "td_acc": 0.38,
    "td_def": 0.69,
    "sub_avg": 0.55,
    "ko_win_rate": 0.34,
    "sub_win_rate": 0.18,
    "decision_win_rate": 0.48,
    "finish_loss_rate": 0.43,
    "knockdowns_per_15": 0.42,
    "elo": 1500.0,
}

WEIGHT_CLASS_FINISH_MULTIPLIER = {
    "Women's Strawweight": 0.78,
    "Women's Flyweight": 0.82,
    "Women's Bantamweight": 0.88,
    "Flyweight": 0.84,
    "Bantamweight": 0.90,
    "Featherweight": 0.96,
    "Lightweight": 1.00,
    "Welterweight": 1.04,
    "Middleweight": 1.08,
    "Light Heavyweight": 1.15,
    "Heavyweight": 1.28,
    "Unknown": 1.0,
}


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def _v(v: Any, key: str) -> float:
    return float(PRIORS[key] if v is None else v)


def _experience_weight(f: FighterProfile) -> float:
    fights = f.ufc_fights if f.ufc_fights is not None else (f.pro_fights or 0)
    minutes = f.minutes or 0.0
    # 0.15 floor prevents sparse fighters from being treated as completely unknowable.
    return float(min(1.0, 0.15 + 0.055 * math.sqrt(max(fights, 0)) + 0.0045 * math.sqrt(max(minutes, 0))))


def _shrunk(v: Any, key: str, f: FighterProfile) -> float:
    w = _experience_weight(f)
    return w * _v(v, key) + (1 - w) * PRIORS[key]


def _age_curve(age: float | None) -> float:
    if age is None:
        return 0.0
    # Broad MMA prime centered around 30, gentle penalty before 24 and increasingly steep after 34.
    if age < 24:
        return -0.035 * (24 - age)
    if age <= 31:
        return 0.018 * (age - 24) / 7
    if age <= 34:
        return 0.018 - 0.015 * (age - 31)
    return -0.027 - 0.050 * (age - 34)


def _layoff_penalty(days: int | None) -> float:
    if days is None or days <= 365:
        return 0.0
    # Modest until two years, steeper after. Caps to avoid dominating matchup.
    return -min(0.28, 0.00018 * (days - 365))


def data_confidence(a: FighterProfile, b: FighterProfile) -> float:
    fields = ["slpm", "sapm", "str_acc", "str_def", "td_avg", "td_acc", "td_def", "sub_avg"]
    completeness = np.mean([getattr(a, x) is not None for x in fields] + [getattr(b, x) is not None for x in fields])
    exp = (_experience_weight(a) + _experience_weight(b)) / 2
    # Penalize UFC newcomers harder because regional competition translation is uncertain.
    novelty = 1.0
    for f in (a, b):
        if (f.ufc_fights or 0) < 2:
            novelty *= 0.88
    return float(np.clip((0.52 * completeness + 0.48 * exp) * novelty, 0.08, 0.97))


def _style_scores(f: FighterProfile) -> dict[str, float]:
    slpm = _shrunk(f.slpm, "slpm", f)
    sapm = _shrunk(f.sapm, "sapm", f)
    sacc = _shrunk(f.str_acc, "str_acc", f)
    sdef = _shrunk(f.str_def, "str_def", f)
    tdavg = _shrunk(f.td_avg, "td_avg", f)
    tdacc = _shrunk(f.td_acc, "td_acc", f)
    tddef = _shrunk(f.td_def, "td_def", f)
    sub = _shrunk(f.sub_avg, "sub_avg", f)
    kd = _shrunk(f.knockdowns_per_15, "knockdowns_per_15", f)

    striking_off = 0.34 * ((slpm - 3.8) / 1.7) + 0.24 * ((sacc - 0.46) / 0.12) + 0.25 * ((kd - 0.42) / 0.48) + 0.17 * f.recent_form
    striking_def = 0.55 * ((3.8 - sapm) / 1.7) + 0.45 * ((sdef - 0.54) / 0.12)
    wrestling_off = 0.52 * ((tdavg - 1.45) / 1.55) + 0.28 * ((tdacc - 0.38) / 0.16) + 0.20 * f.wrestling_chain
    wrestling_def = 0.78 * ((tddef - 0.69) / 0.18) + 0.22 * f.wrestling_chain
    submission_off = 0.76 * ((sub - 0.55) / 0.78) + 0.24 * f.wrestling_chain
    return {
        "striking_off": float(striking_off),
        "striking_def": float(striking_def),
        "wrestling_off": float(wrestling_off),
        "wrestling_def": float(wrestling_def),
        "submission_off": float(submission_off),
    }


def matchup_components(a: FighterProfile, b: FighterProfile, ctx: FightContext) -> dict[str, float]:
    sa, sb = _style_scores(a), _style_scores(b)

    reach_a, reach_b = a.reach_in, b.reach_in
    reach_edge = 0.0 if reach_a is None or reach_b is None else np.clip((reach_a - reach_b) / 8.0, -0.6, 0.6)
    height_edge = 0.0 if a.height_in is None or b.height_in is None else np.clip((a.height_in - b.height_in) / 8.0, -0.45, 0.45)

    strike_edge = (sa["striking_off"] - sb["striking_def"]) - (sb["striking_off"] - sa["striking_def"])
    grapple_edge = (sa["wrestling_off"] - sb["wrestling_def"]) - (sb["wrestling_off"] - sa["wrestling_def"])
    sub_edge = (sa["submission_off"] - b.submission_defense) - (sb["submission_off"] - a.submission_defense)

    elo_a = _shrunk(a.elo, "elo", a)
    elo_b = _shrunk(b.elo, "elo", b)
    elo_edge = (elo_a - elo_b) / 300.0

    physical_edge = 0.65 * reach_edge + 0.20 * height_edge
    form_edge = 0.34 * (a.recent_form - b.recent_form) + 0.16 * (a.camp_change - b.camp_change)
    age_edge = _age_curve(a.age) - _age_curve(b.age)
    layoff_edge = _layoff_penalty(a.days_since_last_fight) - _layoff_penalty(b.days_since_last_fight)
    risk_edge = -0.34 * (a.weight_cut_risk - b.weight_cut_risk) - 0.26 * (a.injury_risk - b.injury_risk)
    cardio_edge = 0.30 * (a.cardio - b.cardio)
    durability_edge = 0.27 * (a.durability - b.durability)

    # Coefficients deliberately avoid letting any one public counting stat dominate.
    latent = (
        0.47 * strike_edge
        + 0.37 * grapple_edge
        + 0.18 * sub_edge
        + 0.32 * elo_edge
        + 0.12 * physical_edge
        + 0.18 * form_edge
        + 0.16 * age_edge
        + 0.13 * layoff_edge
        + 0.13 * risk_edge
        + 0.12 * cardio_edge
        + 0.10 * durability_edge
    )

    return {
        "latent_edge_a": float(latent),
        "strike_edge_a": float(strike_edge),
        "grapple_edge_a": float(grapple_edge),
        "submission_edge_a": float(sub_edge),
        "elo_edge_a": float(elo_edge),
        "physical_edge_a": float(physical_edge),
        "age_edge_a": float(age_edge),
        "cardio_edge_a": float(cardio_edge),
        "durability_edge_a": float(durability_edge),
        "risk_edge_a": float(risk_edge),
    }


def _finish_hazards(a: FighterProfile, b: FighterProfile, comp: dict[str, float], ctx: FightContext) -> tuple[float, float, float, float]:
    wc_mult = WEIGHT_CLASS_FINISH_MULTIPLIER.get(ctx.weight_class, 1.0)
    cage_mult = 1.05 if ctx.small_cage else 1.0
    ref_mult = 1.0 + 0.06 * np.clip(ctx.referee_finish_bias, -1, 1)

    a_ko = _shrunk(a.ko_win_rate, "ko_win_rate", a)
    b_ko = _shrunk(b.ko_win_rate, "ko_win_rate", b)
    a_sub = _shrunk(a.sub_win_rate, "sub_win_rate", a)
    b_sub = _shrunk(b.sub_win_rate, "sub_win_rate", b)
    a_vuln = _shrunk(a.finish_loss_rate, "finish_loss_rate", a)
    b_vuln = _shrunk(b.finish_loss_rate, "finish_loss_rate", b)

    # Per-minute competing-risk hazards. These are matchup-informed, not raw historical finish %.
    a_ko_h = 0.0165 * wc_mult * cage_mult * ref_mult * math.exp(0.46 * comp["strike_edge_a"] + 0.58 * (a_ko - 0.34) + 0.33 * (b_vuln - 0.43) + 0.18 * a.durability - 0.18 * b.durability)
    b_ko_h = 0.0165 * wc_mult * cage_mult * ref_mult * math.exp(-0.46 * comp["strike_edge_a"] + 0.58 * (b_ko - 0.34) + 0.33 * (a_vuln - 0.43) + 0.18 * b.durability - 0.18 * a.durability)

    a_sub_h = 0.0076 * math.exp(0.42 * comp["grapple_edge_a"] + 0.47 * comp["submission_edge_a"] + 0.75 * (a_sub - 0.18) - 0.30 * b.submission_defense)
    b_sub_h = 0.0076 * math.exp(-0.42 * comp["grapple_edge_a"] - 0.47 * comp["submission_edge_a"] + 0.75 * (b_sub - 0.18) - 0.30 * a.submission_defense)

    # Risk controls to avoid absurd hazards from tiny samples/extreme manual modifiers.
    vals = [a_ko_h, a_sub_h, b_ko_h, b_sub_h]
    vals = [float(np.clip(x, 0.0004, 0.075)) for x in vals]
    return vals[0], vals[1], vals[2], vals[3]


def _notes(a: FighterProfile, b: FighterProfile, comp: dict[str, float], conf: float) -> list[str]:
    notes: list[str] = []
    if abs(comp["strike_edge_a"]) >= 0.45:
        fav = a.name if comp["strike_edge_a"] > 0 else b.name
        notes.append(f"{fav} owns the clearer modeled striking interaction edge.")
    if abs(comp["grapple_edge_a"]) >= 0.45:
        fav = a.name if comp["grapple_edge_a"] > 0 else b.name
        notes.append(f"{fav} projects better in takedown access/control exchanges.")
    if abs(comp["submission_edge_a"]) >= 0.50:
        fav = a.name if comp["submission_edge_a"] > 0 else b.name
        notes.append(f"{fav} carries the stronger submission-threat profile in this matchup.")
    if abs(comp["physical_edge_a"]) >= 0.25:
        fav = a.name if comp["physical_edge_a"] > 0 else b.name
        notes.append(f"{fav} has a meaningful modeled length/size advantage.")
    if conf < 0.45:
        notes.append("Low data confidence: widen error bars and demand a larger betting edge.")
    if not notes:
        notes.append("No single phase dominates the matchup model; price and uncertainty matter heavily.")
    return notes


def simulate_fight(
    a: FighterProfile,
    b: FighterProfile,
    ctx: FightContext,
    n_sims: int = 100_000,
    seed: int = 42,
) -> SimulationSummary:
    if n_sims < 1_000:
        raise ValueError("Use at least 1,000 simulations for stable estimates")
    a.clamp(); b.clamp()
    rng = np.random.default_rng(seed)
    comp = matchup_components(a, b, ctx)
    conf = data_confidence(a, b)

    a_ko_h, a_sub_h, b_ko_h, b_sub_h = _finish_hazards(a, b, comp, ctx)
    base_hazards = np.array([a_ko_h, a_sub_h, b_ko_h, b_sub_h], dtype=float)

    # Latent uncertainty increases for sparse fighter data and close/volatile style interactions.
    latent_sd = 0.34 + 0.78 * (1 - conf)
    latent_draws = rng.normal(comp["latent_edge_a"], latent_sd, size=n_sims)

    total_minutes = ctx.total_minutes
    # Fight-to-fight finish intensity variability captures pace/style volatility.
    intensity = rng.lognormal(mean=-0.5 * 0.22**2, sigma=0.22, size=n_sims)
    hazard_scale_a = np.exp(0.14 * latent_draws)
    hazard_scale_b = np.exp(-0.14 * latent_draws)

    h = np.empty((n_sims, 4), dtype=float)
    h[:, 0] = base_hazards[0] * hazard_scale_a * intensity
    h[:, 1] = base_hazards[1] * np.exp(0.10 * latent_draws) * intensity
    h[:, 2] = base_hazards[2] * hazard_scale_b * intensity
    h[:, 3] = base_hazards[3] * np.exp(-0.10 * latent_draws) * intensity
    total_h = h.sum(axis=1)

    finish_time = rng.exponential(1.0 / total_h)
    finished = finish_time < total_minutes

    # Pick finish type conditional on a finish occurring.
    u = rng.random(n_sims) * total_h
    c1 = h[:, 0]
    c2 = c1 + h[:, 1]
    c3 = c2 + h[:, 2]
    finish_type = np.full(n_sims, -1, dtype=np.int8)
    finish_type[(u < c1) & finished] = 0
    finish_type[(u >= c1) & (u < c2) & finished] = 1
    finish_type[(u >= c2) & (u < c3) & finished] = 2
    finish_type[(u >= c3) & finished] = 3

    # For decisions, simulate round effectiveness then three judges. Judging centers on effective
    # striking/grappling; aggressiveness/control appear only as small tie-break/noise terms.
    decision_idx = np.where(~finished)[0]
    dec_a = np.zeros(n_sims, dtype=bool)
    dec_b = np.zeros(n_sims, dtype=bool)
    dec_draw = np.zeros(n_sims, dtype=bool)

    if len(decision_idx):
        m = len(decision_idx)
        judge_totals = np.zeros((m, 3), dtype=np.int16)  # point differential A - B per judge
        for r in range(ctx.scheduled_rounds):
            # Cardio and altitude become more relevant in later rounds.
            late = r / max(ctx.scheduled_rounds - 1, 1)
            cardio_term = 0.16 * (a.cardio - b.cardio) * late
            altitude_term = 0.09 * (a.cardio - b.cardio) * late if ctx.altitude else 0.0
            round_mean = latent_draws[decision_idx] + cardio_term + altitude_term
            round_perf = rng.normal(round_mean, 0.86, size=m)
            for j in range(3):
                observed = round_perf + rng.normal(0, 0.25, size=m)
                # 10-8 threshold requires a much larger effectiveness gap; 10-10 is rare.
                diff = np.where(observed > 2.25, 2, np.where(observed > 0, 1, np.where(observed < -2.25, -2, -1)))
                judge_totals[:, j] += diff.astype(np.int16)
        votes_a = (judge_totals > 0).sum(axis=1)
        votes_b = (judge_totals < 0).sum(axis=1)
        dec_a_local = votes_a >= 2
        dec_b_local = votes_b >= 2
        natural_draw = ~(dec_a_local | dec_b_local)
        # Rare extra majority/split draws or point-deduction-like outcomes, concentrated in close fights.
        closeness = np.exp(-np.abs(latent_draws[decision_idx]) / 0.90)
        extra_draw_prob = np.clip(0.010 * closeness, 0.001, 0.010)
        extra_draw = rng.random(m) < extra_draw_prob
        dec_draw_local = natural_draw | extra_draw
        dec_a_local = dec_a_local & ~dec_draw_local
        dec_b_local = dec_b_local & ~dec_draw_local
        dec_a[decision_idx] = dec_a_local
        dec_b[decision_idx] = dec_b_local
        dec_draw[decision_idx] = dec_draw_local

    a_ko = finish_type == 0
    a_sub = finish_type == 1
    b_ko = finish_type == 2
    b_sub = finish_type == 3
    a_win = a_ko | a_sub | dec_a
    b_win = b_ko | b_sub | dec_b

    # Time thresholds: over 1.5 rounds = after 7:30, etc. Decisions always count as over threshold.
    effective_time = np.where(finished, finish_time, total_minutes)
    p_over_1_5 = float(np.mean(effective_time > 7.5))
    p_over_2_5 = float(np.mean(effective_time > 12.5)) if total_minutes >= 15 else float('nan')
    p_over_4_5 = float(np.mean(effective_time > 22.5)) if total_minutes >= 25 else float('nan')

    p_a = float(np.mean(a_win))
    p_b = float(np.mean(b_win))
    p_draw = float(np.mean(dec_draw))
    # Monte Carlo SE + structural confidence penalty. Used for conservative pricing, not as a literal CI.
    mc_se = math.sqrt(max(p_a * (1 - p_a), 1e-9) / n_sims)
    structural = 0.015 + 0.070 * (1 - conf)
    model_uncertainty = float(math.sqrt(mc_se**2 + structural**2))

    finished_times = finish_time[finished]
    return SimulationSummary(
        fighter_a=a.name,
        fighter_b=b.name,
        n_sims=n_sims,
        seed=seed,
        p_a_win=p_a,
        p_b_win=p_b,
        p_draw=p_draw,
        p_distance=float(np.mean(~finished)),
        p_over_1_5=p_over_1_5,
        p_over_2_5=p_over_2_5,
        p_over_4_5=p_over_4_5,
        p_a_ko=float(np.mean(a_ko)),
        p_a_sub=float(np.mean(a_sub)),
        p_a_dec=float(np.mean(dec_a)),
        p_b_ko=float(np.mean(b_ko)),
        p_b_sub=float(np.mean(b_sub)),
        p_b_dec=float(np.mean(dec_b)),
        avg_finish_minute=float(np.mean(finished_times)) if len(finished_times) else None,
        data_confidence=conf,
        model_uncertainty=model_uncertainty,
        matchup_notes=_notes(a, b, comp, conf),
        diagnostics={
            **comp,
            "a_ko_hazard_per_min": a_ko_h,
            "a_sub_hazard_per_min": a_sub_h,
            "b_ko_hazard_per_min": b_ko_h,
            "b_sub_hazard_per_min": b_sub_h,
            "experience_weight_a": _experience_weight(a),
            "experience_weight_b": _experience_weight(b),
        },
    )
