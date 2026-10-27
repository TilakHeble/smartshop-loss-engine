# SmartShop Loss Engine

[![ci](https://github.com/TilakHeble/smartshop-loss-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/TilakHeble/smartshop-loss-engine/actions/workflows/ci.yml)

**Which SmartShop sessions should be rescanned, and which items checked, so that each minute of colleague time recovers the most loss while honest shoppers are stopped as rarely as possible?**

SmartShop (scan-as-you-shop) loses money when items leave unscanned: some on purpose, usually high-value lines, and some by genuine mistake. Today partial and full rescans are mostly random. This project replaces them with **risk-based rescans that keep a small random floor**, so the policy can still be measured, can't be gamed, and gives colleagues a neutral explanation.

It is the companion to the [Nectar Offer Engine](https://github.com/TilakHeble/nectar-offer-engine): the same members, the same data and shared code. Nectar decides each member's personalised prices; SmartShop is where the member sees them, Nectar price next to normal price, while scanning.

> Status: **weeks 1–7 of 10 complete** (ingestion, EDA, dbt warehouse, calibrated simulator, basket model, risk model, rescan policy, trust and targeted checks). 47 dbt data tests and 52 Python tests, including point-in-time leakage tests, run in CI on every push.

## What makes it different

| Idea | What it does |
|---|---|
| ⭐ **Game-theory rescan policy** | The store commits to rescan *probabilities*; shoppers learn them and choose whether, and how much, to steal (a Stackelberg game). Every policy is judged against their best responses, within today's colleague time, with a random floor and a per-member cap |
| **Targeted checks: "check these items"** | A basket model, trained without any theft label, ranks which product types are probably in the bags but unscanned. Tested honestly: it finds honest mistakes with ~25% fewer units counted, but full rescans recover more per colleague-minute, so it serves as a risk feature and a "look here first" hint, not a separate check |
| **Nectar-linked trust** | Clean full rescans earn a member fewer rescans (up to 75% off, per member across stores); a small honest slip does not reset it; the random floor always applies. A modest fairness refinement that thieves cannot easily earn |
| **True loss from biased rescans** | Known rescan probabilities give an unbiased (inverse-probability-weighted) estimate of total loss, including sessions never checked |

**Target headline:** with the same colleague time, recover more loss than random rescans *and* stop fewer honest shoppers, while holding up against shoppers who adapt.

## Week 1 findings

[dunnhumby — *The Complete Journey*](https://www.dunnhumby.com/source-files/), weeks 17–102: 2.35m scanned item lines, 230k trips, 2,492 members, 532 stores. Amounts in US dollars, as recorded.

![Targeted vs random partial rescan](reports/figures/02_targeted_vs_random_check.png)

| # | Finding | Decision |
|---|---|---|
| 1 | Median trip has 5 item lines (7 units); 15% of trips have 20+ lines | Cost a rescan in minutes per item, not per trip |
| 2 | The 3 most valuable items hold 46% of basket value in 10–19-line trips vs 23% for 3 random items, and 27% vs 10% at 20+ lines | Targeted partial rescans are worth building, before any model |
| 3 | High-risk categories (fresh meat, alcohol, tobacco, health & beauty, baby, coffee) are 9% of lines but 20% of value; 72% of trips with 10+ lines contain one | They anchor the simulator's deliberate-skip behaviour and the risk features |
| 4 | Half of lines carry a Nectar price; valuing loss at the normal price overstates it by 18% overall, 32% for fresh meat, and almost 0% for alcohol and tobacco | Value every lost item at the shopper's own price, line by line |
| 5 | 20% of lines are multi-buys, holding 39% of units | Simulate quantity under-declaring; check quantities in targeted rescans |
| 6 | 4pm–7pm carries 30% of daytime trips; the busiest hour is 1.7x the average | Rescan capacity moves during the day (v2: queue-aware capacity) |
| 7 | 89% of trips come with 10+ earlier trips by the same member; 91% of members use 2+ stores and 56% make over 20% of trips away from their main store | Trust needs member-level history via Nectar, not per-store history |

![Nectar price overstatement](reports/figures/04_nectar_price_overstatement.png)

Full analysis: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb) (HTML export in `reports/`).

## Week 2: the warehouse

A dbt project on DuckDB turns raw transactions into the tables every later week reads from.

```mermaid
flowchart LR
    R[raw.transactions<br/>raw.products] --> S[stg_transactions<br/>normal + Nectar price]
    SD[seeds<br/>high-risk list<br/>non-scan departments] --> P[stg_products<br/>risk group]
    P --> S
    S --> L[fct_scan_line<br/>item line]
    L --> T[fct_trip<br/>session]
    T --> H[fct_trip_history<br/>point-in-time shopper history]
    T --> M[dim_member · dim_store<br/>mart_store_hour]
    L --> D[dim_product<br/>unit prices]
    R --> X[stg_member_demographics<br/>fairness only]
```

| Model | Grain | What it is for |
|---|---|---|
| `fct_scan_line` | Item line | What was scanned, at normal and Nectar price, with risk group and value rank in the basket. Week 3 expands it into timed scans |
| `fct_trip` | Session | Basket size, value, high-risk share, multi-buys, top-3 value share, peak-hour flag, member's trip number |
| `fct_trip_history` | Session | What the system knows about the shopper **when the session starts**: earlier trips, stores, days since last trip, usual basket. Built only from earlier trips |
| `dim_product` | Product | Normal and Nectar unit prices, used to value an unscanned item at the shopper's price |
| `dim_member`, `dim_store` | Member / store | Reporting: main store, stores used, demand profile |
| `mart_store_hour` | Store × hour | Sessions and items arriving per open day: the input to rescan capacity (week 6) |
| `stg_member_demographics` | Member | **Fairness checks only.** Kept apart, and a test fails if any feature table carries a demographic column |

**Tests that protect the later weeks**
- **No leakage.** A pytest rebuilds the warehouse with the data cut off at day 400 and checks every earlier session gets *exactly* the same history. Checked by planting a leak (one history column using the member's whole period): the test fails and names the column.
- **Money reconciles.** Line totals equal session totals equal staging, to the cent; the Nectar price is never above the normal price.
- **Business rules are data.** The high-risk list and non-scan departments are dbt seeds, with a test that every seed entry exists in the product table, so a typo can't silently drop a category.
- **CI runs the real pipeline.** Raw data is never committed, so CI generates a small synthetic warehouse with the same schema (every high-risk product, fuel lines, returns, multi-store members) and runs the full dbt build, all 47 data tests and the leakage test on it.
- The rebuilt marts reproduce the week 1 analysis exactly, figure for figure.

## Week 3: a simulator calibrated to published research

SmartShop data isn't public, and the truth about loss (who skipped what, and why) is never fully known even inside a retailer. `smartshop.simulate` adds SmartShop behaviour on top of the **real** baskets and checks it against published figures.

| Real (Complete Journey) | Simulated (`simulate/config.py`) |
|---|---|
| Members, stores, days, checkout times | Who adopted SmartShop and when; app or handset |
| Every basket's items, quantities, normal and Nectar prices | Hidden shopper type: honest, careless, opportunist, deliberate, adaptive |
| Repeat visits per member | Genuine mistakes (missed and double scans) and deliberate non-scans |
| | Timed scan-event log with deletions |
| | Today's random partial and full rescans, and what they found |

Every SmartShop session is Nectar-linked: [Sainsbury's requires a Nectar card](https://help.sainsburys.co.uk/help/products/smartshop-faq) for both the app and the handset. The cold-start case is a **new SmartShop user**.

**Calibrated to the ECR Retail Loss Group's studies** ([Beck 2018](https://ecrloss.com/research-paper/self-checkout-loss-measurement/): 13 retailers, 140m scan-and-go transactions, 17m audits; [Beck 2022](https://www.rgis.co.uk/wp-content/uploads/2023/04/ECR_Global_Study_on_Self-checkout_in_Retail.pdf): 93 retailers):

| Check | Simulated | Published |
|---|---|---|
| Sessions with an error: 20 / 50 / 100 units | 31% / 65% / 88% | 30% / 60% / 90% |
| Non-scans as share of item errors (rest are double scans) | 76% | 77% |
| Deliberate share of non-scanned value | 55% | 46–56% |
| Net loss, share of SmartShop sales | 2.0% | 0.96–4.68% |
| Sessions rescanned today | 12% | ~12% |
| Partial rescans that find an error | 3.0% | 2.88% |

![Error curve calibration](reports/figures/07_error_curve_calibration.png)

**Design choices that matter later**
- **Truth and observations are separated.** Models only ever read `sim_obs` (sessions, scan events, rescan results). The hidden truth sits in `sim_truth`, and a test fails if any `gt_` column reaches `sim_obs`.
- **Signals come from mechanisms, not labels.** Shopping time follows the *true* basket, so a session with unscanned items looks slow for what was scanned; some deliberate skips are scan-then-delete. Each signal alone is weak (AUC 0.56–0.66), as in a real store.
- **Shoppers respond to the policy.** Opportunists are deterred as the rescan probability rises; adaptive shoppers steal only when the expected haul beats the expected cost of being caught. This is what the game-theory policy (week 6) has to beat.
- **Common random numbers.** Every random draw is made once per unit and session, so any two policies are compared on identical shoppers and baskets.
- Every session's scan log accounts exactly for its true basket: scanned = true − missed + double-scanned (tested).

| Finding | Implication |
|---|---|
| 14% of members cause 75% of the loss; genuine mistakes by everyone else are the other 25% | Two problems: catch deliberate skips, and fix honest mistakes without treating people as suspects |
| High-risk items are 19% of value but 47% of deliberately skipped value | Targeted rescans should look there first |
| Random partial rescans put loss at 0.23% of sales, about a ninth of the truth (2.04%); random full rescans estimate 1.97% (95% CI 1.51–2.45%) | Measure loss with a random floor of *full* rescans, never from partial checks (as [ECR] found in practice) |
| Today's random rescans: 221 colleague hours recover \$10.94 an hour and 3.1% of loss; 96% of rescans stop someone who skipped nothing on purpose | **The baseline the policy must beat** |

![What rescans measure](reports/figures/09_what_rescans_measure.png)

Full validation: [`notebooks/02_simulator.ipynb`](notebooks/02_simulator.ipynb). Results are claims that *the method recovers the known truth*, not that it would cut Sainsbury's loss by a given amount; week 8 reruns the comparison under lower and higher loss settings.

## Week 4: the basket model

**What is probably in the trolley but not on the scan list?** Half of missing value is a product type never scanned, half is a short count of a type that was (2 of 3 yogurts), so the model scores *every* product type in a session.

- **Masked-basket training.** Staffed-till baskets are complete: hide 10% of their units at random and train LightGBM to say where they were, from the visible rest and the member's Nectar history (earlier trips only; for SmartShop trips, only what was scanned). No theft label is ever used. Held-out AUC 0.93, calibrated (mean prediction 0.26% vs 0.25% actual).
- **A fair test.** Every method gets the same budget of units a colleague may count per session, and the simulator's detection rules: mistakes are found when their type is counted; items skipped on purpose are usually concealed outside the bags, so *any* partial check rarely finds them.

![Basket model check lists](reports/figures/10_basket_model_recall.png)

| Finding | Implication |
|---|---|
| The model's check list finds as many honest mistakes as today's random units while counting about 25% fewer units (1.3–1.5x found per unit) | Faster, friendlier partial rescans for honest shoppers |
| "Check the most valuable scanned items" finds about a third as much per unit as random | Intuition is not enough; every rule is measured against random |
| No partial check, random or targeted, finds much deliberate loss; "this basket looks incomplete" is a weak risk signal on its own (AUC 0.55) | **Theft is caught by choosing *who* gets a full rescan** (weeks 5–6), not by item lists |

The gain is deliberately reported as modest: genuine mistakes fall roughly at random across units, so random picks already sample where they are. If real mistakes cluster on particular products (bulky items, loose produce, multi-buys), targeted lists gain more, and the random floor of full rescans is what would reveal it. Full analysis: [`notebooks/03_basket_model.ipynb`](notebooks/03_basket_model.ipynb).

## Week 5: the session risk model

Week 4 showed partial checks rarely find items skipped on purpose (they are concealed), so catching theft is about **who gets a full rescan**. The risk model ranks sessions for it.

- **Only what a store can observe.** Features: the session's scan log (pauses, pace, deletions, what was scanned) and the member's history from earlier trips (Nectar purchases, earlier sessions, earlier rescan outcomes, the basket model's "looks incomplete" gap). Labels: what *random full* rescans found: unbiased, but only 926 in the training period.
- **Honest selection.** Label, features and model chosen by repeated cross-validation of *observable* value capture on those rescans; the hidden truth is used once, at the end. Chosen: logistic regression on 27 features, so every flag comes with plain-English reasons ("long pause between scans; usually deletes a lot").

![Risk capture curve](reports/figures/11_risk_capture_curve.png)

| Flag the top 12% (today's rescan rate) | Deliberate loss caught | Net loss caught |
|---|---|---|
| Random (today) | 13% | 12% |
| Biggest baskets | 22% | 28% |
| Rule of thumb (red-flag points) | 36% | 37% |
| Best single signal (deletion rate) | 50% | 40% |
| **Risk model** | **53%** | **54%** |
| Ceiling: same features, trained on hidden truth | 82% | 67% |

| Finding | Implication |
|---|---|
| About 4x today's random selection at the same rescan rate | Who gets rescanned is where the gain is |
| A 5–20% random full floor adds only 2–6 points; the ceiling is trained on *why* items were missing | The bottleneck is label quality (a rescan records *that* items were missing, not why), more than label count |
| **The first fairness check failed:** handset users, evening shoppers and new SmartShop users were flagged far more, though none of these changes behaviour. With so few labels the model learned spurious channel and time effects, and tenure features made newcomers look unusual | Channel, time of day and tenure removed from scoring; missing history filled with typical values. Channel and evening shoppers now pass comfortably (0.97–0.98); new users pass narrowly (0.82, interval 0.69–0.97) |
| Honest households with bigger baskets are still stopped more: honest-stop ratios 0.51 (age), 0.53 (household size), 0.39 (income), with wide intervals (some groups have very few households) | A review item, not proof. Carried into the policy (a per-member cap on rescans, week 6) and the full check (week 10) |

Full analysis: [`notebooks/04_risk_model.ipynb`](notebooks/04_risk_model.ipynb).

## Week 6: the rescan game

The policy decides **who gets a full rescan, with what probability**, using exactly today's colleague time. It must work against shoppers who learn it.

- **The game.** The store leads: it commits to a function from risk score to rescan probability. Shoppers follow: opportunists steal less when a rescan is likelier; adaptive shoppers choose to take 0, 25, 50 or 100% of their usual haul, since taking less keeps their scan log calmer and their score lower. Every version of reality is simulated once from the same random draws, so any policy is evaluated exactly against shoppers' best responses.
- **Fair comparison.** Policy shapes are fitted on the tuning period (days 331–500); the test period only evaluates, with volume set to today's 113 colleague hours. Every policy keeps the **2% random floor** of full rescans: unbiased measurement and training labels.

![Policy frontier](reports/figures/13_policy_frontier.png)

| Same colleague time (test period) | Loss remaining | Recovered | Per colleague hour | Honest shoppers stopped |
|---|---|---|---|---|
| Today: random partial + full | \$41,639 | \$1,196 | \$11 | 5,428 |
| Random full rescans | \$40,851 | \$3,023 | \$27 | 3,182 |
| Threshold + random floor | \$35,605 | \$7,302 | \$64 | 1,579 |
| **Randomised, cap 0.5 (recommended)** | **\$36,767** | **\$6,483** | **\$57** | **1,665** |

| Finding | Implication |
|---|---|
| The recommended policy leaves 12% less loss, recovers 5.4x as much and stops 69% fewer honest shoppers than today | Rescans go where they find things |
| Judged against shoppers who keep today's behaviour, a threshold policy's recovery is overstated by 37% | Policies must be evaluated against shoppers who learn them |
| Adaptive shoppers do exploit a threshold (about a fifth hold back to stay under it), but randomising gains little: deterring them needs about a one-in-three rescan chance, while the budget allows about 7% | Here the store wins by **detection, not deterrence**; the best policy is close to threshold + random floor |
| Without a cap, some honest members would be rescanned on nearly every trip; a cap of 0.5 stops that for about 3% more loss, but moves group gaps only a little | Cap 0.5 recommended; age, household and income gaps stay a monitored review item (week 10) |
| Today's many cheap partial rescans do deter opportunists (18% steal vs 22% under fewer, full rescans) | A deliberate trade: a little less deterrence for about 5x the recovery |

![Static vs strategic](reports/figures/14_static_vs_strategic.png)

Full analysis: [`notebooks/05_rescan_policy.ipynb`](notebooks/05_rescan_policy.ipynb). The game is one session at a time (member history held at today's); week 8 stress-tests the shoppers' assumed penalty and sensitivity.

## Week 7: trust for honest regulars, and targeted checks

Two ideas tested on the rescan game, with colleague time held at today's level.

**Nectar-linked trust**, played over four rescan cycles: each cycle's rescans (drawn at the policy's probabilities) update trust for the next, and shoppers respond to the probabilities they face.

![Trust and repeat stops](reports/figures/15_trust_repeat_stops.png)

| Finding | Implication |
|---|---|
| With colleague time fixed, trust leaves total loss and honest stops about unchanged, but 12% fewer honest regulars are stopped 3+ times and recovery edges up | Added to the recommended policy as a fairness refinement, not a loss lever |
| "Any find resets trust" hurts honest families: big baskets often contain a slip. Letting one small slip (one item under \$10) pass fixes that | The rule: each clean full rescan takes 25% off, up to 75%; bigger finds reset it |
| Thieves are 14% of members but get about 10% of the trust discount | Trust built on full rescans is hard to game |
| Counting clean *partial* checks spreads trust to far more members, thieves included, and is worse on loss and honest stops | Never build trust on checks that cannot see concealed items |

**Targeted checks** (week 4's lists) against full rescans, per colleague-minute, at three risk bands under the recommended policy:

![Targeted vs full](reports/figures/16_targeted_vs_full.png)

| Finding | Implication |
|---|---|
| Full rescans recover several times more per colleague-minute than a 3-unit targeted check at every band (\$1.20 vs \$0.32 on flagged sessions) | Targeted checks do not earn colleague time; the basket model's value is as a risk feature and a "look here first" hint inside a full rescan |
| An in-app "did you forget...?" prompt about the model's top guess would be right only 10–24% of the time and point at at most 5–9% of the value missed by mistake | Not worth interrupting shoppers |

Full analysis: [`notebooks/06_trust_and_targeting.ipynb`](notebooks/06_trust_and_targeting.ipynb). The recommended policy (week 6 randomised policy plus this trust rule) is stored in `models/policy.json`.

## Ground rules

- **No demographics or protected characteristics** in any model; they are used only for the fairness check. Channel, time of day and tenure are not scored either (they caused unfair flags in week 5), and proxies are checked.
- **Stopping an honest shopper is a cost** in the objective, not only £ recovered.
- **No history is not a reason to rescan more:** new SmartShop users get the standard rate.
- **A colleague makes the final call.** Every decision comes with a plain-English reason and the items to check.
- **Rescans never touch Nectar prices or points.** The Nectar link is only used positively.

## Roadmap

**v1: the core system**

| Week | Deliverable |
|---|---|
| 1 ✅ | Repo, ingestion (shared with Nectar), SmartShop EDA |
| 2 ✅ | dbt warehouse: session, scan-line (normal and Nectar price) and point-in-time shopper-history models, with tests |
| 3 ✅ | Session simulator: honest, careless, opportunist, deliberate and adaptive shoppers, calibrated to published ECR figures |
| 4 ✅ | Basket model: which items are probably in the basket but unscanned |
| 5 ✅ | Session risk model against three baselines: random, simple rules, top-k |
| 6 ✅ | Game-theory rescan policy |
| 7 ✅ | Targeted rescans and Nectar trust score |
| 8 | True loss from biased rescans (IPW) and the headline comparison |
| 9 | FastAPI rescan-decision API in Docker; gated weekly pipeline |
| 10 | Streamlit dashboard, fairness check, model card with a data-protection section, business case |

**v2: after v1**: deliberate vs accidental misses, deterrence value of a rescan, queue-aware capacity, a conformal cap on honest shoppers stopped, data-driven product hotspots, a nudge experiment sized with CUPED, in-app "you might have forgotten…" prompts, scan-time camera verification.

## Quick start

Needs the 7 Complete Journey CSVs used here (the in-store promotions file is not needed).

```bash
pip install -e ".[dev]"     # also installs the Nectar engine, pinned to an exact commit
# put the CSVs in data/raw/
make ingest                 # CSV -> typed Parquet + DuckDB warehouse (~3 s)
make dbt                    # build and test the staging and mart models (~15 s)
make simulate               # SmartShop sessions, scan events and today's rescans (~10 s)
make basket                 # train the masked-basket model (~1.5 min)
make risk                   # session risk model (first run scores the basket gap, ~5 min)
make policy                 # solve the rescan game, compare policies (~1 min)
make eda                    # run the EDA notebook and export HTML
make test                   # unit, leakage and data-quality tests
make dbt-docs               # browse model documentation and lineage
make sample                 # no data? build a synthetic warehouse with the same schema
```

On Windows without `make`: `python -m smartshop.ingest.load_raw`, then `cd dbt; dbt build --profiles-dir .; cd ..`, then `python -m smartshop.simulate.simulator`, then `python -m smartshop.basket.model`, then `python -m smartshop.risk.model`, then `python -m smartshop.policy.search`, then `pytest -q`. Use Python 3.12, as for Nectar.

## Repository layout

```
├── src/smartshop/
│   ├── config.py            # paths; reads business rules from dbt seeds
│   ├── ingest/load_raw.py   # reuses Nectar's typed table specs -> Parquet + DuckDB
│   ├── ingest/sample.py     # synthetic warehouse with the real schema (CI, demos)
│   ├── simulate/            # SimConfig (sources), simulator, rescan model, calibration
│   ├── basket/              # observed history, masked-basket features, LightGBM model, fair evaluation
│   ├── risk/                # observable risk features, honest selection, evaluation, fairness
│   ├── policy/              # the rescan game, policy search, trust over cycles, targeted checks
│   ├── warehouse.py         # connect to DuckDB, run dbt from Python
│   └── viz.py               # chart style shared with Nectar
├── dbt/                     # seeds, staging, marts, 47 data tests
├── notebooks/               # 01 EDA · 02 simulator validation · 03 basket model · 04 risk model · 05 rescan policy · 06 trust and targeting
├── models/                  # basket and risk metrics (+ model files, generated by make basket / make risk)
├── reports/                 # HTML export + figures
├── tests/                   # marts on synthetic data, point-in-time leakage, data-quality contract
└── .github/workflows/ci.yml # ruff + dbt parse + pytest (full dbt build on synthetic data)
```

## Tech stack

Python · SQL · DuckDB · Parquet · dbt · LightGBM · scikit-learn · pytest · GitHub Actions. To come: MLflow · FastAPI · Docker · Prefect · Streamlit.
