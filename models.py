from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional
import math


@dataclass
class FighterProfile:
    name: str
    age: Optional[float] = None
    height_in: Optional[float] = None
    reach_in: Optional[float] = None
    stance: str = "Unknown"

    # UFCStats-style rate stats
    slpm: Optional[float] = None
    sapm: Optional[float] = None
    str_acc: Optional[float] = None  # 0..1
    str_def: Optional[float] = None  # 0..1
    td_avg: Optional[float] = None   # per 15 min
    td_acc: Optional[float] = None   # 0..1
    td_def: Optional[float] = None   # 0..1
    sub_avg: Optional[float] = None  # per 15 min

    # Additional finish/experience features
    ko_win_rate: Optional[float] = None
    sub_win_rate: Optional[float] = None
    decision_win_rate: Optional[float] = None
    finish_loss_rate: Optional[float] = None
    knockdowns_per_15: Optional[float] = None
    ufc_fights: Optional[int] = None
    pro_fights: Optional[int] = None
    minutes: Optional[float] = None
    days_since_last_fight: Optional[int] = None

    # Ratings / user adjustments
    elo: Optional[float] = None
    cardio: float = 0.0       # -1..1
    durability: float = 0.0   # -1..1
    wrestling_chain: float = 0.0
    submission_defense: float = 0.0
    recent_form: float = 0.0
    camp_change: float = 0.0
    weight_cut_risk: float = 0.0  # 0..1, higher worse
    injury_risk: float = 0.0      # 0..1, higher worse

    def as_dict(self) -> dict:
        return asdict(self)

    def clamp(self) -> "FighterProfile":
        for field in ("str_acc", "str_def", "td_acc", "td_def", "ko_win_rate", "sub_win_rate", "decision_win_rate", "finish_loss_rate"):
            v = getattr(self, field)
            if v is not None:
                setattr(self, field, min(max(float(v), 0.0), 1.0))
        for field in ("cardio", "durability", "wrestling_chain", "submission_defense", "recent_form", "camp_change"):
            setattr(self, field, min(max(float(getattr(self, field)), -1.0), 1.0))
        for field in ("weight_cut_risk", "injury_risk"):
            setattr(self, field, min(max(float(getattr(self, field)), 0.0), 1.0))
        return self


@dataclass
class FightContext:
    scheduled_rounds: int = 3
    weight_class: str = "Unknown"
    title_fight: bool = False
    altitude: bool = False
    small_cage: bool = False
    referee_finish_bias: float = 0.0  # -1..1, optional manual qualitative modifier

    @property
    def total_minutes(self) -> int:
        return int(self.scheduled_rounds) * 5


@dataclass
class SimulationSummary:
    fighter_a: str
    fighter_b: str
    n_sims: int
    seed: int
    p_a_win: float
    p_b_win: float
    p_draw: float
    p_distance: float
    p_over_1_5: float
    p_over_2_5: float
    p_over_4_5: float
    p_a_ko: float
    p_a_sub: float
    p_a_dec: float
    p_b_ko: float
    p_b_sub: float
    p_b_dec: float
    avg_finish_minute: Optional[float]
    data_confidence: float
    model_uncertainty: float
    matchup_notes: list[str]
    diagnostics: dict

    def outcome_probability(self, key: str) -> float:
        mapping = {
            "A ML": self.p_a_win,
            "B ML": self.p_b_win,
            "Fight Goes Distance": self.p_distance,
            "Fight Doesn't Go Distance": 1.0 - self.p_distance,
            "Over 1.5": self.p_over_1_5,
            "Under 1.5": 1.0 - self.p_over_1_5,
            "Over 2.5": self.p_over_2_5,
            "Under 2.5": 1.0 - self.p_over_2_5,
            "Over 4.5": self.p_over_4_5,
            "Under 4.5": 1.0 - self.p_over_4_5,
            "A by KO/TKO": self.p_a_ko,
            "A by Submission": self.p_a_sub,
            "A by Decision": self.p_a_dec,
            "B by KO/TKO": self.p_b_ko,
            "B by Submission": self.p_b_sub,
            "B by Decision": self.p_b_dec,
        }
        if key not in mapping:
            raise KeyError(f"Unknown market outcome: {key}")
        return mapping[key]
