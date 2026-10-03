# MMA Quant Engine

A local Streamlit app for MMA fight simulation, matchup analysis, live moneyline shopping, prop pricing, expected-value analysis, and conservative bankroll sizing.

## What makes it different

This is not a pick aggregator and it does not ingest pundit/tout opinions. It uses fighter statistics and explicit scouting inputs to create its own probability distribution, then compares those probabilities to sportsbook prices.

The engine includes:

- Bayesian-style shrinkage for sparse fighter samples
- Striking offense vs. opponent striking defense interactions
- Takedown/wrestling offense vs. takedown defense interactions
- Submission threat tied to grappling access
- Bounded physical, age, layoff, cardio, durability, camp, injury and weight-cut adjustments
- Separate KO/TKO and submission competing-risk hazards
- Weight-class finish-rate adjustment
- Round-by-round decision simulation with judge noise and 10-8 logic
- 3-round and 5-round fights
- 10k to 500k Monte Carlo runs
- Moneyline, distance, totals and method-of-victory probabilities
- American-odds conversion, de-vigging, fair odds, EV and Kelly staking
- "Bad Bet / Pass / Thin / Value / Strong Value" price classifications
- Conservative fair price that incorporates model uncertainty
- Live MMA moneylines through The Odds API
- UFCStats fighter-profile loading, with manual override
- Beginner and Pro views

## Quick start

**Windows:** double-click `RUN_WINDOWS.bat`.

**macOS/Linux:** open a terminal in the folder and run `./run_mac_linux.sh`.

The launchers create a virtual environment, install requirements, and start the app.

### Manual installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the localhost address Streamlit prints in your terminal.

## Optional live odds

Create an account/API key at The Odds API and either paste the key in the app or set:

```bash
export ODDS_API_KEY="your_key"
```

The app requests the `mma_mixed_martial_arts` sport with `h2h` moneylines. The provider documents current MMA/UFC winner odds and some limited totals coverage. The engine supports manual entry for props/markets that are not available from the feed.

Provider docs: https://the-odds-api.com/sports/mma-ufc-odds.html

## UFC stats

The "Load UFCStats" button looks fighters up on UFCStats and imports the public profile-rate fields used by the model (SLpM, SApM, striking accuracy/defense, takedown rates/accuracy/defense, submission attempts, height, reach, age and UFC fight history when parsable).

UFCStats: http://ufcstats.com/

If UFCStats changes its HTML, manual inputs still work and the scraper can be updated independently in `mma_engine/ufcstats.py`.

## Betting logic

### Fair price

If the model gives a fighter 60% win probability, the fair decimal price is `1 / 0.60 = 1.667`, or approximately `-150` American odds.

A model favorite is **not automatically a bet**. If the sportsbook offers -190, that fighter can be the likeliest winner while still being a negative-EV wager.

### De-vigging

When both sides from the same sportsbook are available, the app converts each price to implied probability and proportionally normalizes the probabilities so they sum to 100%. This is the comparison baseline for moneyline edge.

### EV

For a 1-unit stake:

`EV = P(win) × net_profit_if_win − P(loss)`

### Kelly

The app defaults to quarter-Kelly, multiplies the result by data confidence, and imposes a hard bankroll cap. These are deliberately conservative defaults and can be changed in the sidebar.

## Decision simulation and judging

The decision engine prioritizes effective striking/grappling, then uses only small tie-breaking/noise effects rather than treating cage control as equal to damaging/effective offense. This is aligned with the ordering in the Unified Rules judging criteria.

ABC Unified Rules / scoring resources:
- https://www.abcboxing.com/unified-rules/
- https://www.abcboxing.com/wp-content/uploads/2025/02/ABC-MMA-Officials-Handbook.pdf

## Important accuracy notes

No model is "accurate" merely because it runs many simulations. Monte Carlo count reduces simulation noise; it does **not** fix bad inputs or model misspecification. The biggest accuracy gains come from:

1. using only pre-fight information,
2. tracking closing lines and calibration,
3. measuring out-of-sample Brier/log loss,
4. validating newcomers separately from established UFC fighters,
5. checking performance by division, gender, 3-round/5-round format and market type,
6. recording line movement and closing-line value,
7. recalibrating probability outputs over time.

Before risking meaningful money, keep a prediction log of at least several hundred fights and verify calibration. A 70% bucket should win about 70% of the time over a large enough sample.

## Project layout

```text
mma_quant_engine/
├── app.py
├── requirements.txt
├── README.md
├── .env.example
├── RUN_WINDOWS.bat
├── run_mac_linux.sh
├── sample_prediction_log.csv
├── mma_engine/
│   ├── __init__.py
│   ├── models.py
│   ├── simulation.py
│   ├── betting.py
│   ├── analysis.py
│   ├── odds.py
│   └── ufcstats.py
└── tests/
    └── test_core.py
```

## Recommended next upgrades

For a production betting model, the next layer should be a true historical training/backtesting pipeline using timestamped pre-fight features. Recommended additions are opponent-adjusted Elo/Glicko, rolling/decayed round-level stats, control time and position data, knockdown rates, short-notice flags, travel/altitude, southpaw/orthodox matchup interactions, weigh-in misses, line history and separate calibrated models by market. The current code is structured so those can be added without rewriting the UI or bet-pricing layer.
