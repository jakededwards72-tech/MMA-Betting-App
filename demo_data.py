from models import FighterProfile

DEMO_A = FighterProfile(
    name="Alex Mercer", age=29, height_in=71, reach_in=74, stance="Orthodox",
    slpm=5.05, sapm=3.25, str_acc=.51, str_def=.58, td_avg=1.15, td_acc=.42,
    td_def=.76, sub_avg=.35, ko_win_rate=.52, sub_win_rate=.10,
    decision_win_rate=.38, finish_loss_rate=.28, knockdowns_per_15=.72,
    ufc_fights=10, pro_fights=17, minutes=112, days_since_last_fight=168, elo=1580,
    cardio=.18, durability=.15, wrestling_chain=.05, submission_defense=.12,
)
DEMO_B = FighterProfile(
    name="Darius Cole", age=32, height_in=70, reach_in=71, stance="Southpaw",
    slpm=3.72, sapm=3.88, str_acc=.45, str_def=.52, td_avg=3.10, td_acc=.47,
    td_def=.67, sub_avg=1.20, ko_win_rate=.22, sub_win_rate=.39,
    decision_win_rate=.39, finish_loss_rate=.35, knockdowns_per_15=.28,
    ufc_fights=12, pro_fights=21, minutes=139, days_since_last_fight=224, elo=1545,
    cardio=.12, durability=.05, wrestling_chain=.38, submission_defense=.20,
)
