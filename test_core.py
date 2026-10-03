from mma_engine.betting import american_to_decimal, implied_probability_from_american, devig_two_way, expected_roi
from mma_engine.models import FighterProfile, FightContext
from mma_engine.simulation import simulate_fight


def test_odds_math():
    assert round(american_to_decimal(-110), 6) == round(1 + 100/110, 6)
    assert abs(implied_probability_from_american(100) - 0.5) < 1e-9
    a, b, hold = devig_two_way(-110, -110)
    assert abs(a - 0.5) < 1e-9
    assert abs(b - 0.5) < 1e-9
    assert hold > 0
    assert abs(expected_roi(0.5, 100)) < 1e-9


def test_simulation_probabilities_sum():
    a = FighterProfile(name="A", ufc_fights=8, minutes=90, slpm=4.2, sapm=3.1, str_acc=.49, str_def=.58, td_avg=1.8, td_acc=.42, td_def=.75, sub_avg=.5)
    b = FighterProfile(name="B", ufc_fights=8, minutes=90, slpm=3.6, sapm=4.0, str_acc=.45, str_def=.52, td_avg=1.0, td_acc=.35, td_def=.67, sub_avg=.3)
    s = simulate_fight(a, b, FightContext(scheduled_rounds=3, weight_class="Lightweight"), n_sims=5000, seed=123)
    assert abs((s.p_a_win + s.p_b_win + s.p_draw) - 1.0) < 0.01
    assert 0 <= s.p_distance <= 1
    assert 0 <= s.p_a_ko + s.p_a_sub + s.p_a_dec <= 1
