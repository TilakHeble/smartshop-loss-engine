# Model card: SmartShop rescan decision

| | |
|---|---|
| **System** | A session risk model plus a randomised rescan policy and a Nectar-linked trust rule |
| **Decides** | At the SmartShop exit: full rescan or not. It returns the probability used, the random draw, plain-English reasons and the shopper's usual high-risk items that are missing ("look here first") |
| **Risk model** | Logistic regression on 25 observable features: the session's scan log (pauses, pace, deletions, what was scanned) and the member's history from earlier trips (usual basket, pace, deletions, earlier rescan outcomes). No demographics, channel, time of day or tenure |
| **Labels** | What random full rescans found ("found anything"): the only unbiased label a store has. 1,934 at the latest cutoff (390 found something); 926 in the training period used for evaluation |
| **Policy (v1.1)** | The risk score is first made partly **size-neutral**: compared with sessions of a similar number of units scanned (strength 0.5, chosen on the tuning period). Full-rescan probability then rises smoothly from a **2% random floor** to a **50% cap**. Volume is recalibrated every week so expected colleague time equals today's |
| **Trust** | Each clean full rescan takes 25% off a member's probability, up to 75%. A small slip (one item under \$10) doesn't reset it. Never below the floor |
| **Selected by** | Repeated cross-validation of observable value capture on random full rescans. The hidden truth was used once, for final evaluation |
| **Code** | `risk/`, `policy/`, `serving/`, `pipeline/weekly.py`, `reporting/` |
| **Status** | Portfolio project on simulated SmartShop behaviour over real baskets. Not deployed on real customers |

## Intended use

- **For:** choosing which SmartShop sessions get a full rescan at a store's existing colleague time, so rescans find more and honest shoppers are stopped less. Also for measuring total loss without bias from the logged probabilities.
- **Users:** store colleagues at the exit (who make the final call), loss-prevention and store-operations analysts.
- **Not for:**
  - accusing anyone. A rescan request is an operational check, and the reasons are written to be neutral;
  - any decision beyond the rescan: no bans, no withdrawal of SmartShop, no change to Nectar prices or points;
  - staffed tills or self-checkout without retraining (different signals);
  - stores or populations unlike the training ones (the drift gate blocks this).

## How it was built and why

- **Calibrated simulator.** SmartShop data isn't public, and true loss is never fully known even inside a retailer. Simulated shoppers (honest, careless, opportunist, deliberate, adaptive) act on the real dunnhumby baskets. They are calibrated to the ECR Retail Loss Group's studies of scan-and-go loss (error rates by basket size, the deliberate share of loss, net loss at 2.0% of sales, today's 12% rescan rate).
- **Truth and observations kept apart.** Models read only what a store can observe. A test fails if a hidden-truth column reaches an observed table.
- **Shoppers respond.** Policies are judged against shoppers who learn them (a Stackelberg game). Judging them against static shoppers overstated recovery by 20–52%.

## Performance

**Ranking (week 5, test period, flag the top 12%):** the model catches 53% of deliberately skipped value and 54% of net loss, against 12–13% at random and 36–37% with a red-flag points rule. A model trained on the hidden truth reaches 82%; the gap is label quality (a rescan records *that* items were missing, not why).

**Policy (test period, today's colleague time, shoppers respond, trust over four cycles, mean of 10 runs):**

| | Today | v1.0 (week 8) | **v1.1 (served)** |
|---|---|---|---|
| Loss remaining | \$41,639 (2.07% of sales) | \$36,767 (1.83%) | **\$33,430 (1.66%)** |
| Value recovered by rescans | \$1,196 | \$6,640 (5.6x) | **\$8,513** (7.1x) |
| Recovered per colleague hour | \$11 | \$59 | **\$75** |
| Honest shoppers stopped | 5,428 | 1,683 (−69%) | **1,918** (−65%) |

The served model leaves out the basket-gap feature, which would need the whole basket model at the till; the stress tests use the same configuration. In fourteen stress tests, each changing one simulator assumption and retraining everything, v1.1 leaves 6–22% less loss than today, recovers 3.2–8.2x as much and stops 63–66% fewer honest shoppers, and beats v1.0 on loss in every one.

**An unseen theft method.** Barcode switching (scanning a cheap item's barcode for an expensive one) leaves the count, pace and deletions normal, so the strongest scan-log signals vanish. With every thief switching and the model never retrained, v1.1 still leaves 5.7% less loss than today (8.5% if retrained) and 36% of deliberate loss sits in the top 12% of scores (random: 12%), because member history still points at it.

## Fairness

**Harm definition.** Being asked for a full rescan **when honest**. The check uses the policy as it would run (shoppers respond, trust updates) and the anonymised demographics available for 32% of households. Demographics are joined only in this check.

| Attribute | Four-fifths ratio | Same basket sizes | Random groups would give | Most-stopped group vs today | Verdict |
|---|---|---|---|---|---|
| Channel (app / handset) | 0.98 | 0.99 | 0.98 | 36% | pass |
| New SmartShop user | 0.88 | 0.84 | 0.97 | 40% | pass |
| Evening peak | 0.99 | 0.98 | 0.98 | 36% | pass |
| Age group | 0.69 | 0.78 | 0.82 (p = 0.05) | 42% | on the edge of chance: monitor |
| Household size | 0.72 | 0.81 | 0.85 | 46% | explained by basket size |
| Income level | 0.72 | 0.80 | 0.64 (p = 0.78) | 45% | explained by basket size |

Policy v1.1, as served. Under v1.0 (week 10) the ratios were 0.96 / 0.98 / 1.00 / **0.61 (age, p = 0.01: review)** / 0.64 / 0.60.

- **No group is worse off than today:** the most-stopped group in every attribute is stopped at 36–46% of today's rate.
- **The permutation test matters.** With 11 income groups of as few as 11 households, random groups give a ratio of 0.61, so the raw 0.60 is not evidence of unfairness.
- **The age gap, and the fix.** Under v1.0, younger family households (age groups 2–3) were stopped about 1.6x as often as the oldest when honest, because they buy the largest baskets and big baskets score high. v1.1 compares each session with sessions of a similar size, which narrows the gap (0.61 → 0.69) and also cuts loss further, because big baskets are slow to rescan and mostly hold honest slips. Full strength closes it (0.90) but stops 35% more honest shoppers than v1.1 and takes new users to 0.80. Age is a protected characteristic under the Equality Act 2010, so a remaining gap needs to be a proportionate means of a legitimate aim. That is a business and legal judgement, to be made explicitly and recorded:
  1. choose the strength knowingly (0.5 now; 1.0 if relative parity matters most);
  2. measure the gap in the store trial with Nectar's consented demographic data (aggregated);
  3. re-run this check on every weekly bundle.
- **Earlier failures, fixed.** The first model (week 5) flagged handset users, evening shoppers and new users far more, from spurious channel and time effects learned from few labels. Those features and tenure were removed from scoring, and missing history is filled with typical values. The weekly job blocks any bundle that rescans app or handset shoppers, or new users, more than 1.25x everyone else.

## Data protection (UK GDPR / DPA 2018, as amended by the Data (Use and Access) Act 2025)

This is a design assessment for a portfolio project, not legal advice. A real deployment needs a full Data Protection Impact Assessment with the retailer's DPO.

| Question | Position in this design |
|---|---|
| **Is a DPIA needed?** | Yes. The design profiles every SmartShop session at scale and uses an innovative technology, both ICO triggers for a DPIA (Article 35) |
| **Lawful basis** | Detecting deliberate non-scanning is *prevention or detection of crime*, a recognised legitimate interest since 5 February 2026 (Article 6(1)(ea), Annex 1), so no balancing test is needed for it. The same profiling also targets honest mistakes and measures loss, which are ordinary legitimate interests (Article 6(1)(f)) and need a documented legitimate interests assessment. Necessity is supported by the result: fewer honest shoppers stopped at the same colleague time |
| **Automated decision-making** | A colleague makes the final call and sees the reasons, so the request should not be *solely* automated. Even if it were judged solely automated and significant, Article 22C safeguards are designed in: shoppers are told how rescans are chosen, every decision is explained and logged, and a shopper can ask a colleague, make representations and contest a decision through the complaints route |
| **Special category data** | None used. Demographics (anonymised, not special category) are used only for the fairness check, kept apart in the warehouse, and a test fails if any feature table carries one |
| **Data minimisation** | Features come from the scan log the app already records and the member's own Nectar history. No camera, location, device or payment data. Channel, time of day and tenure are deliberately not scored |
| **Transparency** | The privacy notice and SmartShop terms should say that rescans are chosen partly at random and partly by risk from scanning patterns and history, and that clean rescans reduce future checks |
| **Accuracy and fairness** | Reasons are neutral ("long pause between scans"). The 2% random floor means anyone can be rescanned, so a rescan implies nothing. Fairness is checked weekly (channel, new users) and per review (demographics) |
| **Retention** | Decision log (probability, draw, outcome) for loss measurement and audits, e.g. 13 months for year-on-year loss. Member profiles are rebuilt weekly from source, not kept as history |
| **Rights** | Access: a member's trust status is available from the API (`/members/{id}/trust`), and their logged decisions through a subject access request. Objection under Article 21 to the non-crime part of the profiling has to be handled (for example, the standard rate for that member). Complaints: an electronic complaints route with acknowledgement within 30 days (section 164A DPA 2018, from 19 June 2026) |
| **Security** | The serving image holds no raw data. Member profiles are pseudonymous Nectar IDs with derived numbers only, mounted read-only. The dashboard shows aggregates only |

## Limitations

- **Simulated behaviour.** The results show the method recovers a known truth when shoppers behave like the calibrated simulator, not that Sainsbury's loss would fall by 12%. A store-level randomised trial is the test (business case).
- **US baskets, small population.** 2,500 dunnhumby households in US dollars; simulated stores are much smaller than real ones.
- **Label quality.** Rescans record that something was missing, not why. The model ranks "likely to have missing items", which mixes honest mistakes with theft (the 82% ceiling shows the cost).
- **Static store capacity.** Colleague time is held constant per period; queue-aware capacity by hour is v2.
- **Adaptive thieves.** Detection drives the result. If thieves adapt faster or harder than assumed, the random floor and cap are the levers (stress tests cover several such cases).
- **New methods.** A method that leaves no trace in the scan log (tested: barcode switching) weakens the gain to a few per cent until retraining catches up; the random floor still measures it.

## Monitoring and maintenance

- **Weekly job gates:** at least 300 labels; expected colleague time within 2% of today's; every probability between floor and cap; no PSI alert (≥ 0.25) against the live bundle's last cycle; channel and new-user parity (no group above 1.25x); score rank correlation with the live model ≥ 0.7. A failed gate keeps the live bundle, and every run is recorded.
- **Loss reporting:** inverse-probability weighted from the logged probabilities (unbiased in simulation, about 9% error per period), with the random floor as the cross-check.
- **Review triggers:** any gate failure; drift warnings in consecutive weeks; the age gap widening in the trial; a change to store layout, the app, or the rescan procedure.
