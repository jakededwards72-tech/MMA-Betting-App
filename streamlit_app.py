from __future__ import annotations

import os
from dataclasses import asdict

import pandas as pd
import streamlit as st

from models import FighterProfile, FightContext
from simulation import simulate_fight
from ufcstats import get_fighter_by_name, UFCStatsError
from odds import fetch_mma_moneylines, flatten_moneylines, find_event_lines
from analysis import simulation_table, evaluate_moneyline_market, evaluate_manual_markets, expert_breakdown
from backtest import backtest_report, PlattCalibrator, recalibrate_summary
from demo_data import DEMO_A, DEMO_B
from storage import add_bet, list_bets, settle_bet, delete_bet

st.set_page_config(page_title="MMA Quant Engine", page_icon="🥊", layout="wide")

st.markdown("# 🥊 MMA Quant Engine")
st.caption("Fight simulation • fair prices • line shopping • EV/Kelly • bet tracking • calibration")
st.markdown("""<style>
.block-container{padding-top:1.4rem;max-width:1500px}.stMetric{background:rgba(120,120,120,.08);padding:12px;border-radius:12px}
div[data-testid="stTabs"] button{font-weight:650}.value-card{border:1px solid rgba(120,120,120,.25);border-radius:14px;padding:14px}
</style>""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Engine settings")
    view = st.radio("View", ["Simple", "Pro"], horizontal=True)
    n_sims = st.select_slider("Monte Carlo simulations", options=[10_000, 25_000, 50_000, 100_000, 250_000, 500_000], value=100_000)
    seed = st.number_input("Simulation seed", min_value=1, max_value=2_000_000_000, value=42, step=1)
    fractional_kelly = st.slider("Kelly fraction", 0.0, 0.5, 0.25, 0.05, help="0.25 = quarter Kelly. Model confidence is applied on top of this.")
    max_bet_pct = st.slider("Hard cap per bet (% bankroll)", 0.25, 10.0, 3.0, 0.25) / 100
    st.divider()
    st.caption("Bet labels are model/price classifications, not guarantees. A line can be a bad bet even when that fighter is more likely to win.")


def empty_profile(name: str) -> FighterProfile:
    return FighterProfile(name=name or "Fighter")


def profile_editor(prefix: str, f: FighterProfile) -> FighterProfile:
    st.markdown(f"### {f.name}")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18.0, 55.0, float(f.age or 30.0), 0.1, key=f"{prefix}_age")
    height = c2.number_input("Height (in)", 55.0, 84.0, float(f.height_in or 70.0), 0.5, key=f"{prefix}_height")
    reach = c3.number_input("Reach (in)", 55.0, 90.0, float(f.reach_in or 72.0), 0.5, key=f"{prefix}_reach")

    c1, c2, c3, c4 = st.columns(4)
    slpm = c1.number_input("Sig. strikes landed/min", 0.0, 12.0, float(f.slpm or 3.8), 0.05, key=f"{prefix}_slpm")
    sapm = c2.number_input("Sig. strikes absorbed/min", 0.0, 12.0, float(f.sapm or 3.8), 0.05, key=f"{prefix}_sapm")
    str_acc = c3.number_input("Strike accuracy %", 0.0, 100.0, float((f.str_acc if f.str_acc is not None else 0.46) * 100), 1.0, key=f"{prefix}_sacc") / 100
    str_def = c4.number_input("Strike defense %", 0.0, 100.0, float((f.str_def if f.str_def is not None else 0.54) * 100), 1.0, key=f"{prefix}_sdef") / 100

    c1, c2, c3, c4 = st.columns(4)
    td_avg = c1.number_input("TD avg / 15", 0.0, 12.0, float(f.td_avg or 1.45), 0.05, key=f"{prefix}_tdavg")
    td_acc = c2.number_input("TD accuracy %", 0.0, 100.0, float((f.td_acc if f.td_acc is not None else 0.38) * 100), 1.0, key=f"{prefix}_tdacc") / 100
    td_def = c3.number_input("TD defense %", 0.0, 100.0, float((f.td_def if f.td_def is not None else 0.69) * 100), 1.0, key=f"{prefix}_tddef") / 100
    sub_avg = c4.number_input("Sub attempts / 15", 0.0, 8.0, float(f.sub_avg or 0.55), 0.05, key=f"{prefix}_subavg")

    with st.expander("Advanced fighter inputs", expanded=(view == "Pro")):
        c1, c2, c3, c4 = st.columns(4)
        ufc_fights = c1.number_input("UFC fights", 0, 60, int(f.ufc_fights or 0), 1, key=f"{prefix}_ufcf")
        minutes = c2.number_input("UFC minutes", 0.0, 800.0, float(f.minutes or 0.0), 1.0, key=f"{prefix}_mins")
        elo = c3.number_input("Optional Elo", 1000.0, 2200.0, float(f.elo or 1500.0), 5.0, key=f"{prefix}_elo")
        layoff = c4.number_input("Days since last fight", 0, 3000, int(f.days_since_last_fight or 180), 10, key=f"{prefix}_layoff")

        c1, c2, c3, c4 = st.columns(4)
        ko_wr = c1.slider("KO share of wins", 0.0, 1.0, float(f.ko_win_rate if f.ko_win_rate is not None else 0.34), 0.01, key=f"{prefix}_kowr")
        sub_wr = c2.slider("SUB share of wins", 0.0, 1.0, float(f.sub_win_rate if f.sub_win_rate is not None else 0.18), 0.01, key=f"{prefix}_subwr")
        fin_loss = c3.slider("Finish share of losses", 0.0, 1.0, float(f.finish_loss_rate if f.finish_loss_rate is not None else 0.43), 0.01, key=f"{prefix}_flr")
        kd15 = c4.number_input("Knockdowns / 15", 0.0, 5.0, float(f.knockdowns_per_15 or 0.42), 0.05, key=f"{prefix}_kd15")

        st.caption("Manual scouting modifiers are intentionally bounded. 0 = neutral; ±1 = extreme. Use them only for information not captured by public stats.")
        c1, c2, c3 = st.columns(3)
        cardio = c1.slider("Cardio", -1.0, 1.0, float(f.cardio), 0.05, key=f"{prefix}_cardio")
        durability = c2.slider("Durability", -1.0, 1.0, float(f.durability), 0.05, key=f"{prefix}_dur")
        wrestling_chain = c3.slider("Chain wrestling / control", -1.0, 1.0, float(f.wrestling_chain), 0.05, key=f"{prefix}_chain")
        c1, c2, c3 = st.columns(3)
        sub_defense = c1.slider("Submission defense", -1.0, 1.0, float(f.submission_defense), 0.05, key=f"{prefix}_subdef")
        recent_form = c2.slider("Recent-form adjustment", -1.0, 1.0, float(f.recent_form), 0.05, key=f"{prefix}_form")
        camp_change = c3.slider("Camp/change adjustment", -1.0, 1.0, float(f.camp_change), 0.05, key=f"{prefix}_camp")
        c1, c2 = st.columns(2)
        cut_risk = c1.slider("Weight-cut risk", 0.0, 1.0, float(f.weight_cut_risk), 0.05, key=f"{prefix}_cut")
        injury_risk = c2.slider("Known injury risk", 0.0, 1.0, float(f.injury_risk), 0.05, key=f"{prefix}_inj")

    return FighterProfile(
        name=f.name, age=age, height_in=height, reach_in=reach, stance=f.stance,
        slpm=slpm, sapm=sapm, str_acc=str_acc, str_def=str_def,
        td_avg=td_avg, td_acc=td_acc, td_def=td_def, sub_avg=sub_avg,
        ko_win_rate=ko_wr if view == "Pro" else f.ko_win_rate,
        sub_win_rate=sub_wr if view == "Pro" else f.sub_win_rate,
        decision_win_rate=f.decision_win_rate,
        finish_loss_rate=fin_loss if view == "Pro" else f.finish_loss_rate,
        knockdowns_per_15=kd15 if view == "Pro" else f.knockdowns_per_15,
        ufc_fights=ufc_fights if view == "Pro" else f.ufc_fights,
        pro_fights=f.pro_fights,
        minutes=minutes if view == "Pro" else f.minutes,
        days_since_last_fight=layoff if view == "Pro" else f.days_since_last_fight,
        elo=elo if view == "Pro" else f.elo,
        cardio=cardio if view == "Pro" else f.cardio,
        durability=durability if view == "Pro" else f.durability,
        wrestling_chain=wrestling_chain if view == "Pro" else f.wrestling_chain,
        submission_defense=sub_defense if view == "Pro" else f.submission_defense,
        recent_form=recent_form if view == "Pro" else f.recent_form,
        camp_change=camp_change if view == "Pro" else f.camp_change,
        weight_cut_risk=cut_risk if view == "Pro" else f.weight_cut_risk,
        injury_risk=injury_risk if view == "Pro" else f.injury_risk,
    )


if "fighter_a" not in st.session_state:
    st.session_state.fighter_a = empty_profile("Fighter A")
if "fighter_b" not in st.session_state:
    st.session_state.fighter_b = empty_profile("Fighter B")

matchup_tab, odds_tab, tracker_tab, pro_tab, backtest_tab, about_tab = st.tabs(["🥊 Matchup", "💵 Odds & Bets", "🧾 Bet Tracker", "🔬 Pro Diagnostics", "📈 Backtest", "📘 Methodology"])

with matchup_tab:
    st.subheader("1. Load fighters")
    demo_col, _ = st.columns([1,4])
    if demo_col.button("Load demo fight", use_container_width=True):
        st.session_state.fighter_a = DEMO_A
        st.session_state.fighter_b = DEMO_B
        st.rerun()
    c1, c2, c3 = st.columns([2, 2, 1])
    name_a = c1.text_input("Fighter A", value=st.session_state.fighter_a.name if st.session_state.fighter_a.name != "Fighter A" else "")
    name_b = c2.text_input("Fighter B", value=st.session_state.fighter_b.name if st.session_state.fighter_b.name != "Fighter B" else "")
    if c3.button("Load UFCStats", use_container_width=True):
        if not name_a or not name_b:
            st.error("Enter both fighter names first.")
        else:
            try:
                with st.spinner("Loading fighter statistics…"):
                    pa, url_a = get_fighter_by_name(name_a)
                    pb, url_b = get_fighter_by_name(name_b)
                st.session_state.fighter_a = pa
                st.session_state.fighter_b = pb
                st.session_state.ufcstats_urls = (url_a, url_b)
                st.success("Loaded UFCStats profiles. Review/edit anything you want before simulating.")
                st.rerun()
            except UFCStatsError as e:
                st.error(str(e))

    if "ufcstats_urls" in st.session_state:
        ua, ub = st.session_state.ufcstats_urls
        st.caption(f"Source profiles: {ua}  |  {ub}")

    st.divider()
    st.subheader("2. Fight setup")
    c1, c2, c3, c4 = st.columns(4)
    rounds = c1.selectbox("Scheduled rounds", [3, 5], index=0)
    weight_class = c2.selectbox("Weight class", ["Women's Strawweight", "Women's Flyweight", "Women's Bantamweight", "Flyweight", "Bantamweight", "Featherweight", "Lightweight", "Welterweight", "Middleweight", "Light Heavyweight", "Heavyweight", "Unknown"], index=6)
    altitude = c3.checkbox("Meaningful altitude", False)
    small_cage = c4.checkbox("Small-cage effect", False)

    st.divider()
    st.subheader("3. Review fighter inputs")
    left, right = st.columns(2)
    with left:
        pa = profile_editor("a", st.session_state.fighter_a)
    with right:
        pb = profile_editor("b", st.session_state.fighter_b)

    run = st.button("Run simulation", type="primary", use_container_width=True)
    if run:
        ctx = FightContext(scheduled_rounds=rounds, weight_class=weight_class, title_fight=(rounds == 5), altitude=altitude, small_cage=small_cage)
        with st.spinner(f"Running {n_sims:,} fight simulations…"):
            summary = simulate_fight(pa, pb, ctx, n_sims=n_sims, seed=int(seed))
        st.session_state.summary = summary
        st.session_state.current_a = pa
        st.session_state.current_b = pb
        st.session_state.context = ctx

    if "summary" in st.session_state:
        s = st.session_state.summary
        st.divider()
        st.subheader("Simulation result")
        a1, a2, a3, a4 = st.columns(4)
        a1.metric(f"{s.fighter_a} win", f"{s.p_a_win:.1%}")
        a2.metric(f"{s.fighter_b} win", f"{s.p_b_win:.1%}")
        a3.metric("Goes distance", f"{s.p_distance:.1%}")
        a4.metric("Data confidence", f"{s.data_confidence:.0%}", help="Confidence in input coverage/sample size; not a pick confidence score.")

        for note in s.matchup_notes:
            st.write("•", note)

        table = simulation_table(s)
        st.dataframe(table.style.format({"Probability": "{:.1%}"}), use_container_width=True, hide_index=True)

        st.markdown("### Model-grounded expert breakdown")
        for heading, body in expert_breakdown(s).items():
            st.markdown(f"**{heading}:** {body}")

        if view == "Simple":
            st.info("Fair odds are the model's break-even prices. A sportsbook price is attractive only when it is better than the model's fair price by enough to survive uncertainty and vig.")

with odds_tab:
    st.subheader("Live moneyline shopping")
    if "summary" not in st.session_state:
        st.info("Run a matchup simulation first.")
    else:
        s = st.session_state.summary
        default_key = os.getenv("ODDS_API_KEY", "")
        api_key = st.text_input("The Odds API key", value=default_key, type="password", help="Current MMA moneylines are available from The Odds API. Your key stays in this Streamlit session unless you put it in an environment variable.")
        regions = st.text_input("Book regions", value="us,us2")
        if st.button("Fetch live MMA lines"):
            try:
                events = fetch_mma_moneylines(api_key, regions=regions)
                rows = flatten_moneylines(events)
                matchup_rows = find_event_lines(rows, s.fighter_a, s.fighter_b)
                st.session_state.live_lines = matchup_rows
                if matchup_rows:
                    st.success(f"Found {len(matchup_rows)} sportsbook sides for this matchup.")
                else:
                    st.warning("The feed returned MMA events, but this exact fighter-name pair was not found. Use the manual line table below if needed.")
            except Exception as e:
                st.error(str(e))

        if st.session_state.get("live_lines"):
            live = st.session_state.live_lines
            raw_df = pd.DataFrame(live)
            show_cols = [c for c in ["book", "selection", "odds", "last_update"] if c in raw_df.columns]
            st.dataframe(raw_df[show_cols].sort_values(["selection", "odds"], ascending=[True, False]), use_container_width=True, hide_index=True)
            evdf = evaluate_moneyline_market(s, live, fractional_kelly=fractional_kelly, max_bankroll_fraction=max_bet_pct)
            if not evdf.empty:
                st.markdown("### Price evaluation")
                display = evdf.rename(columns={
                    "selection":"Selection", "book":"Book", "odds":"Odds", "model_p":"Raw win P", "pricing_p":"Model fair P", "push_p":"Draw/push P",
                    "market_p":"Break-even P", "no_vig_market_p":"No-vig market P", "edge":"Edge",
                    "ev_roi":"Expected ROI", "suggested_fraction":"Suggested bankroll", "fair_odds":"Fair odds",
                    "conservative_fair_odds":"Conservative fair odds", "verdict":"Verdict"
                })
                cols = ["Verdict","Selection","Book","Odds","Model fair P","No-vig market P","Edge","Expected ROI","Fair odds","Conservative fair odds","Suggested bankroll"]
                st.dataframe(display[cols].style.format({
                    "Model fair P":"{:.1%}", "No-vig market P":"{:.1%}", "Edge":"{:+.1%}",
                    "Expected ROI":"{:+.1%}", "Suggested bankroll":"{:.2%}", "Fair odds":"{:.0f}", "Conservative fair odds":"{:.0f}"
                }), use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("Manual props / alternate books")
        choices = [
            f"{s.fighter_a} ML", f"{s.fighter_b} ML", "Fight Goes Distance", "Fight Doesn't Go Distance",
            "Over 1.5", "Under 1.5", "Over 2.5", "Under 2.5",
        ]
        if st.session_state.context.scheduled_rounds == 5:
            choices += ["Over 4.5", "Under 4.5"]
        choices += [
            f"{s.fighter_a} by KO/TKO", f"{s.fighter_a} by Submission", f"{s.fighter_a} by Decision",
            f"{s.fighter_b} by KO/TKO", f"{s.fighter_b} by Submission", f"{s.fighter_b} by Decision",
        ]
        starter = pd.DataFrame([
            {"market":"Prop", "selection": choices[0], "book":"Book", "odds": -110},
            {"market":"Prop", "selection": choices[1], "book":"Book", "odds": +100},
        ])
        manual = st.data_editor(
            starter,
            num_rows="dynamic",
            use_container_width=True,
            column_config={
                "selection": st.column_config.SelectboxColumn("Selection", options=choices, required=True),
                "odds": st.column_config.NumberColumn("American odds", step=5, required=True),
            },
            key="manual_lines_editor",
        )
        if st.button("Evaluate manual lines"):
            mdf = evaluate_manual_markets(s, manual.to_dict("records"), fractional_kelly=fractional_kelly, max_bankroll_fraction=max_bet_pct)
            if mdf.empty:
                st.warning("No valid lines to evaluate.")
            else:
                st.session_state.manual_eval = mdf
        if "manual_eval" in st.session_state:
            mdf = st.session_state.manual_eval
            display = mdf.rename(columns={
                "selection":"Selection", "market":"Market", "book":"Book", "odds":"Odds", "model_p":"Raw win P", "pricing_p":"Model fair P", "push_p":"Push P",
                "market_p":"Break-even P", "edge":"Raw edge", "ev_roi":"Expected ROI", "suggested_fraction":"Suggested bankroll",
                "fair_odds":"Fair odds", "conservative_fair_odds":"Conservative fair odds", "verdict":"Verdict", "reason":"Why"
            })
            cols = ["Verdict","Selection","Book","Odds","Model fair P","Break-even P","Expected ROI","Fair odds","Conservative fair odds","Suggested bankroll","Why"]
            st.dataframe(display[cols].style.format({
                "Model fair P":"{:.1%}", "Break-even P":"{:.1%}", "Expected ROI":"{:+.1%}",
                "Suggested bankroll":"{:.2%}", "Fair odds":"{:.0f}", "Conservative fair odds":"{:.0f}"
            }), use_container_width=True, hide_index=True)

with tracker_tab:
    st.subheader("Bet tracker")
    st.caption("Saved locally on this device in mma_quant.db. Track actual bets separately from model recommendations.")
    bets = list_bets()
    if bets.empty:
        st.info("No tracked bets yet. Add one below or evaluate a market in Odds & Bets first.")
    else:
        closed = bets[bets["result"] != "OPEN"]
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Tracked bets", len(bets))
        c2.metric("Open risk", f"${bets.loc[bets.result=='OPEN','stake'].sum():,.2f}")
        c3.metric("Realized P/L", f"${closed.profit.sum():+,.2f}")
        denom = closed.stake.sum()
        c4.metric("Realized ROI", f"{(closed.profit.sum()/denom if denom else 0):+.1%}")
        st.dataframe(bets, use_container_width=True, hide_index=True)
        ids = bets["id"].astype(int).tolist()
        c1,c2,c3 = st.columns(3)
        bid = c1.selectbox("Bet ID", ids)
        result = c2.selectbox("Set result", ["WIN","LOSS","PUSH"])
        if c3.button("Settle selected bet", use_container_width=True):
            settle_bet(bid, result); st.rerun()
        if st.button("Delete selected bet"):
            delete_bet(bid); st.rerun()
    st.markdown("### Add bet")
    with st.form("add_bet_form", clear_on_submit=True):
        c1,c2 = st.columns(2)
        event = c1.text_input("Event", "UFC")
        fight = c2.text_input("Fight")
        c1,c2,c3 = st.columns(3)
        market = c1.text_input("Market", "Moneyline")
        selection = c2.text_input("Selection")
        book = c3.text_input("Sportsbook", "Manual")
        c1,c2,c3,c4 = st.columns(4)
        odds = c1.number_input("American odds", value=-110, step=5)
        stake = c2.number_input("Stake ($)", min_value=0.01, value=10.0, step=1.0)
        model_p = c3.number_input("Model probability", min_value=0.0, max_value=1.0, value=0.50, step=.01)
        model_ev = c4.number_input("Model EV", value=0.0, step=.01)
        submitted = st.form_submit_button("Save bet", use_container_width=True)
        if submitted:
            add_bet(event,fight,market,selection,book,odds,stake,model_p,model_ev)
            st.success("Bet saved."); st.rerun()

with pro_tab:
    if "summary" not in st.session_state:
        st.info("Run a matchup simulation first.")
    else:
        s = st.session_state.summary
        st.subheader("Matchup decomposition")
        diag = pd.DataFrame([{"metric": k, "value": v} for k, v in s.diagnostics.items()])
        st.dataframe(diag, use_container_width=True, hide_index=True)
        st.write(f"Structural/model uncertainty allowance: **±{s.model_uncertainty:.1%}** probability points (approximate, intentionally conservative).")
        st.markdown("#### Interpretation")
        st.write("Positive edge metrics favor Fighter A; negative values favor Fighter B. These components feed the latent decision-performance model and the KO/submission competing-risk hazards separately, so one fighter can have a higher decision probability while the opponent owns more finish equity.")
        st.markdown("#### Sensitivity checklist")
        st.write("Re-run the fight after changing any uncertain input that could plausibly move the result: injury status, layoff, 5-round cardio, bad weight cut, last-minute opponent, major camp change, or a UFC newcomer whose regional stats are difficult to translate. A bet that disappears under reasonable assumptions is fragile and should be treated as a pass or reduced stake.")


with backtest_tab:
    st.subheader("Backtest & probability calibration")
    st.write("Upload your historical prediction log. This measures whether the engine is actually calibrated and can fit a simple out-of-sample-style probability correction without changing the matchup model itself.")
    st.code("model_p,result,odds\n0.62,1,-140\n0.41,0,+125", language="text")
    st.caption("Required: model_p (0–1) and result (1 if the selected side won, 0 if it lost). Optional: odds, the price you recorded when the prediction was made. Use pre-fight predictions only; do not backfill probabilities after results are known.")
    up = st.file_uploader("Prediction log CSV", type=["csv"], key="backtest_csv")
    if up is not None:
        try:
            bt = pd.read_csv(up)
            rep = backtest_report(bt)
            c1, c2, c3 = st.columns(3)
            c1.metric("Predictions", f"{rep['n']:,}")
            c2.metric("Brier score", f"{rep['brier']:.4f}", help="Lower is better. 0 is perfect; a constant 50% forecast has Brier 0.25 in a balanced sample.")
            c3.metric("Log loss", f"{rep['log_loss']:.4f}", help="Lower is better and heavily penalizes confident wrong predictions.")
            if "flat_bet_roi" in rep:
                c1, c2 = st.columns(2)
                c1.metric("Average model EV at logged prices", f"{rep['avg_model_ev']:+.1%}")
                c2.metric("Realized flat-bet ROI", f"{rep['flat_bet_roi']:+.1%}")
            cal = rep["calibration"].copy()
            st.markdown("### Calibration buckets")
            st.dataframe(cal.style.format({"mean_pred":"{:.1%}", "win_rate":"{:.1%}", "calibration_error":"{:+.1%}"}), use_container_width=True, hide_index=True)
            if rep["n"] >= 20:
                calibrator = PlattCalibrator.fit(bt["model_p"], bt["result"])
                st.write(f"Platt calibration fitted on **{calibrator.n}** predictions: slope **{calibrator.a:.3f}**, intercept **{calibrator.b:.3f}**.")
                grid = pd.DataFrame({"Raw model P": [0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90]})
                grid["Calibrated P"] = calibrator.predict(grid["Raw model P"])
                st.dataframe(grid.style.format({"Raw model P":"{:.0%}","Calibrated P":"{:.1%}"}), use_container_width=True, hide_index=True)
                if "summary" in st.session_state and st.button("Apply calibration to current fight"):
                    st.session_state.summary = recalibrate_summary(st.session_state.summary, calibrator)
                    st.session_state.active_calibrator = calibrator
                    st.success("Applied to the current fight's win and method probabilities. Re-open Matchup/Odds to see recalibrated prices.")
            else:
                st.info("At least 20 rows are required to fit calibration; 100+ is much more useful. Metrics above are still shown.")
        except Exception as e:
            st.error(str(e))

with about_tab:
    st.subheader("What the engine is doing")
    st.markdown(
        """
**1) Shrink noisy stats.** UFC newcomers and small samples are pulled toward broad divisional-neutral priors instead of taking extreme rate stats literally.

**2) Model interactions.** Striking offense is evaluated against the opponent's striking defense; wrestling offense is evaluated against takedown defense; submission threat depends partly on grappling access. Length, age, layoffs, cardio, durability and optional scouting modifiers are bounded so they cannot overwhelm the statistical base.

**3) Simulate finishes as competing risks.** KO/TKO and submission hazards are separate for each fighter and vary fight-to-fight. Weight class, cage size and matchup edges influence finish rates.

**4) Simulate decisions round by round.** Each round gets an underlying effectiveness score plus judge-level noise. Large effectiveness gaps can produce 10-8s. Aggression/control are not treated as equal to effective striking/grappling.

**5) Price bets, not picks.** The same fighter can be the model favorite and still be a bad bet. Moneylines are compared against de-vigged market probability when both sides from the same book are available. Props are compared to break-even probability.

**6) Apply uncertainty to staking.** Quarter-Kelly is the default, then scaled by data confidence and capped per bet. The "conservative fair odds" price subtracts a model-uncertainty allowance before converting probability to odds.
        """
    )
    st.warning("This is a probabilistic research tool. It cannot know private injuries, judging mistakes, last-minute illness, undisclosed camp issues, or future randomness. Backtest and recalibrate before trusting real money at scale.")
