from __future__ import annotations

from dataclasses import dataclass, replace
import math
import numpy as np
import pandas as pd

from .betting import expected_roi
from .models import SimulationSummary


def _clip(p):
    return np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)


def brier_score(p, y) -> float:
    p = _clip(p); y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def log_loss(p, y) -> float:
    p = _clip(p); y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1-y) * np.log(1-p)))


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    p = _clip(p); y = np.asarray(y, dtype=float)
    frame = pd.DataFrame({"p": p, "y": y})
    # Fixed probability bins make interpretation stable over time.
    edges = np.linspace(0, 1, bins + 1)
    frame["bin"] = pd.cut(frame["p"], bins=edges, include_lowest=True, duplicates="drop")
    out = frame.groupby("bin", observed=False).agg(n=("y","size"), mean_pred=("p","mean"), win_rate=("y","mean")).reset_index()
    out["calibration_error"] = out["win_rate"] - out["mean_pred"]
    return out[out["n"] > 0].reset_index(drop=True)


@dataclass
class PlattCalibrator:
    a: float = 1.0
    b: float = 0.0
    n: int = 0

    def predict(self, p):
        p = _clip(p)
        x = np.log(p / (1-p))
        z = np.clip(self.a * x + self.b, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    @classmethod
    def fit(cls, p, y, max_iter: int = 100, l2: float = 0.02):
        p = _clip(p); y = np.asarray(y, dtype=float)
        if len(p) < 20:
            raise ValueError("Use at least 20 historical predictions for calibration; 100+ is preferable.")
        x = np.log(p / (1-p))
        X = np.column_stack([x, np.ones_like(x)])
        beta = np.array([1.0, 0.0], dtype=float)
        for _ in range(max_iter):
            z = np.clip(X @ beta, -30, 30)
            mu = 1.0 / (1.0 + np.exp(-z))
            w = np.maximum(mu * (1-mu), 1e-6)
            grad = X.T @ (mu - y) + l2 * np.array([beta[0] - 1.0, beta[1]])
            hess = X.T @ (X * w[:, None]) + l2 * np.eye(2)
            step = np.linalg.solve(hess, grad)
            beta_new = beta - step
            if np.max(np.abs(beta_new - beta)) < 1e-8:
                beta = beta_new
                break
            beta = beta_new
        return cls(a=float(beta[0]), b=float(beta[1]), n=len(p))


def backtest_report(df: pd.DataFrame) -> dict:
    required = {"model_p", "result"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    d = df.copy()
    d["model_p"] = pd.to_numeric(d["model_p"], errors="coerce")
    d["result"] = pd.to_numeric(d["result"], errors="coerce")
    d = d.dropna(subset=["model_p", "result"])
    d = d[d["result"].isin([0,1])]
    if len(d) < 5:
        raise ValueError("Not enough valid rows.")
    report = {
        "n": len(d),
        "brier": brier_score(d.model_p, d.result),
        "log_loss": log_loss(d.model_p, d.result),
        "calibration": calibration_table(d.model_p, d.result),
    }
    if "odds" in d.columns:
        d["odds"] = pd.to_numeric(d["odds"], errors="coerce")
        q = d.dropna(subset=["odds"]).copy()
        if len(q):
            q["model_ev"] = [expected_roi(p, o) for p, o in zip(q.model_p, q.odds)]
            # Realized return for a 1u stake using the recorded offered price.
            def realized(row):
                if row.result == 0:
                    return -1.0
                o = row.odds
                return o / 100 if o > 0 else 100 / abs(o)
            q["realized_roi"] = q.apply(realized, axis=1)
            report["avg_model_ev"] = float(q.model_ev.mean())
            report["flat_bet_roi"] = float(q.realized_roi.mean())
    return report


def recalibrate_summary(summary: SimulationSummary, calibrator: PlattCalibrator) -> SimulationSummary:
    decisive_mass = summary.p_a_win + summary.p_b_win
    if decisive_mass <= 0:
        return summary
    raw_conditional = summary.p_a_win / decisive_mass
    new_conditional = float(calibrator.predict([raw_conditional])[0])
    new_a = decisive_mass * new_conditional
    new_b = decisive_mass * (1-new_conditional)
    scale_a = new_a / summary.p_a_win if summary.p_a_win > 0 else 1.0
    scale_b = new_b / summary.p_b_win if summary.p_b_win > 0 else 1.0
    return replace(
        summary,
        p_a_win=new_a,
        p_b_win=new_b,
        p_a_ko=summary.p_a_ko*scale_a,
        p_a_sub=summary.p_a_sub*scale_a,
        p_a_dec=summary.p_a_dec*scale_a,
        p_b_ko=summary.p_b_ko*scale_b,
        p_b_sub=summary.p_b_sub*scale_b,
        p_b_dec=summary.p_b_dec*scale_b,
        matchup_notes=summary.matchup_notes + [f"Win probabilities calibrated with {calibrator.n} historical predictions (Platt scaling)."],
    )
