# Rescanning the right SmartShop sessions

[← Back to the project page](../README.md) · [All documents](README.md)

*A two-page summary of the SmartShop Loss Engine for a commercial and store-operations audience. Technical detail is in the [README](../README.md) and the [model card](model_card.md).*

## The problem

Scan-as-you-shop is convenient and cheap to run, but items leave unscanned: some on purpose, usually high-value lines, and many by genuine mistake. Published studies of scan-and-go put net loss at roughly 1–5% of its sales (ECR Retail Loss Group, 93 retailers). Today stores protect it with random rescans, about 12% of sessions, mostly partial checks of about a fifth of the basket (anything found and the check device suggests a full check). Random rescans have three costs:

- **They find little.** Partial checks rarely find items skipped on purpose, because those items are hidden outside the bags. In the simulation, today's checks recover about \$13 per colleague hour.
- **They stop honest shoppers.** 96% of today's rescans stop someone who skipped nothing on purpose.
- **They can't measure loss properly.** Partial checks put loss at about a ninth of its true level.

## What was built

1. **A calibrated simulator** of SmartShop behaviour on 2.35 million real basket lines (dunnhumby *The Complete Journey*). It matches the published error rates, the deliberate share of loss and today's rescan rate, and it keeps the hidden truth apart from what a store can see.
2. **A session risk model** that reads only what a store already has: the scan log (pauses, pace, deletions, what was scanned) and the member's Nectar history. It learns from what random full rescans found and gives plain-English reasons.
3. **A rescan policy designed against shoppers who learn it.** Rescan probability rises with risk, from a 2% random floor (everyone can be checked) to a 50% cap (no one is checked every time). Each session's risk is judged against baskets of a similar size, so big family shops full of honest slips don't soak up colleague time. Regulars with clean rescans earn up to 75% off. Colleague time stays exactly at today's level.
4. **A decision API and weekly job.** The exit decision takes about 25 ms. Every week the model retrains and is published only if it passes budget, drift and fairness gates.
5. **Unbiased loss measurement.** Every decision logs its probability, so total loss can be estimated without bias even though rescans are targeted.

## Results (simulation, test period, same colleague time as today)

| | Today | Recommended | |
|---|---|---|---|
| Loss remaining | \$41,130 | **\$32,216** | **−22%** (2.04% → 1.60% of SmartShop sales) |
| Value recovered by checks | \$1,705 | **\$9,320** | **5.5x** |
| Recovered per colleague hour | \$13 | **\$70** | |
| Honest shoppers stopped | 5,428 | **2,270** | **−58%** |

- **Two wins at once, at no extra cost:** less loss and a much better experience for honest customers. The highest-risk 6.5% of sessions receive 53% of checks and hold 86% of the value found.
- **It holds up.** In fourteen stress tests that each change one assumption (more or less theft, weaker signals, bolder or more cautious thieves, and a theft method the model was never designed for) and retrain everything, loss is 5–24% lower, recovery 2.1–5.8x higher and honest stops 56–60% lower.
- **It got better by being fairer.** The first version (v1.0: −13% loss, 1,944 honest stops) stopped younger family households more, because big baskets score high. Judging each session against baskets of its size narrowed that gap and caught more, because big baskets are slow to rescan and mostly hold honest slips. The cost: about 17% more honest stops than v1.0, still 58% fewer than today.
- **It covers sessions that never reach the till.** A shoplifter who scans everything, cancels SmartShop and walks out behind a paying customer costs the whole basket, about an eighth of all loss, and no check sees it. Colleagues carry no device, so the response uses what the store has. Anyone who cancels with items is asked to leave the basket at the SmartShop checkout, where a colleague scans a cancel barcode on the check device: genuine shoppers are helped, and a cancel never finished there becomes countable (about 3 in 10 are walk-outs). Two unfinished cancels in 90 days bring a full check on the member's next shop, reaching about 2% of honest members and recovering \$154 per colleague hour. Whether that deters walk-outs is unmeasured (a quarter put off would cut them by 26%). Dead handsets are the store's failure: replacing batteries that keep dying cuts them by about 93%, and a dead device is resumed, never rescanned.
- **As an illustration only:** in the simulation's terms, loss falls by about 0.44% of SmartShop sales, roughly 4,400 per million of SmartShop sales each period. The figure that counts is the one a store trial measures.
- **Both kinds of check are modelled as they work in store (v1.4).** A partial check covers about a fifth of the basket and escalates to a full check if it finds anything; modelled this way today's checks catch about 43% more than the first version assumed, and the results above are measured against that stronger baseline. When the till targets a shopper, a full check beats a partial one at every mix tested, even on a half-staffed shift, because partial checks rarely reach items hidden on purpose. For the random partial checks, a sample of **25% of the basket** (at least 2 items) is the safe size: the best size depends on how much a quick check deters thieves, which nobody has measured, and 25% stays within 1.4% of the best whichever way that goes, while stopping about 6% fewer honest shoppers than 20% for the same colleague time.
- **What didn't earn its place** was dropped: targeted partial checks (full rescans recover several times more per colleague-minute) and in-app "did you forget...?" prompts (usually wrong).

## The trade-offs to decide

| Trade-off | What the evidence says | Lever |
|---|---|---|
| Deterrence vs detection | Today's many cheap partial checks deter cautious opportunists slightly more; the gain from catching more is far larger. The policy gains least where theft is low (14%) or where thieves switch to a method the model wasn't built for (5–7% when every thief swaps barcodes) | Raise the random floor; keep retraining on random full rescans |
| Repeat stops of honest regulars | Without a cap some honest members would be checked on nearly every trip | Cap 50%, plus the trust discount |
| Fairness vs honest stops | Every group is stopped at under half today's rate. The recommended policy narrows the age gap (four-fifths ratio 0.60 → 0.70, on the edge of chance); going fully size-neutral closes it (0.88) and cuts loss slightly more, but stops more honest shoppers and pushes new users to the edge of the four-fifths rule | The size-neutral strength (0.5 now); age is a protected characteristic, so record the choice |

## How to prove it in real stores

The simulator shows the method works; a trial shows what it is worth.

1. **Randomise stores, not shoppers.** Rescan rules are visible in store, so shoppers in one store must all face the same policy. Match stores in pairs on their baseline loss rate and randomise within each pair.
2. **Same colleague time in both arms.** The recommended policy is recalibrated each week to the arm's own staffing.
3. **Measure loss the same way in both arms**, from random full rescans: today's 2% random full rescans in the control arm, the policy's 2% floor in the trial arm. Primary outcome: loss rate (unscanned value / SmartShop sales). Secondary: honest-shopper stops, recovery per colleague hour, rescan queue time, customer complaints, and the age-group stop rates.
4. **Adjust for each store's baseline period** (CUPED / ANCOVA) and analyse at store level.
5. **Size it.** Each store's loss estimate from a 2% audit has about 23% error over 20,000 SmartShop sessions. To detect a 12% reduction with 80% power (deliberately below the 22% expected, and below every stress test except barcode switching), with store loss rates fairly stable over time (baseline correlation 0.7), the trial needs **roughly 40–100 stores per arm**, depending on how much real stores differ. A longer trial or a larger random audit share cuts that. The interactive sizing is on the dashboard.
6. **Start small.** A one-region shadow mode first: the API scores every session and logs what it would have done, while today's random rescans continue. That checks latency, reasons and fairness on real data before any shopper sees a change.

## Risks and how they are handled

| Risk | Control |
|---|---|
| Shoppers learn the rules | Randomised probabilities, a random floor and a cap; policies judged against shoppers who adapt |
| A bad retrain or data change | Weekly gates (labels, budget, floor and cap, drift, parity, score sanity); atomic publishing; the last good bundle keeps serving |
| Unfair treatment | No demographics, channel, time or tenure in the model; risk judged against similar-sized baskets; parity gates every week; a full demographic review per release |
| Cancel-and-walk through the gates | Finish every cancel at the SmartShop point; count unfinished cancels next to the loss estimate; a full check after two in 90 days; gates that allow one person per receipt scan (a quarter fewer walk-outs) |
| Dead handsets | Replace batteries that keep failing (from the handset log); a recently returned handset asks to be left on charge; resume the session, never make the customer rescan |
| A new way to steal | Random full rescans keep finding whatever is missed; the weekly retrain learns from them; finds on low-score sessions are the early warning (tested with barcode switching) |
| Losing sight of true loss | Every decision logs its probability; unbiased weighted estimates, with the random floor as the cross-check |
| Customer trust | The till only asks for a check, as it does today; neutral reasons; a colleague handles anything found; clean rescans reduce future checks; and every shopper can still be checked at random |

## What I would do next (v2)

- **Queue-aware capacity:** move rescan probability with exit queues and staffing by hour.
- **Deliberate vs accidental:** use rescan outcomes (concealed vs in-bag) as richer labels; the ceiling suggests large gains.
- **Product hotspots:** learn which products are missed most from the random floor's finds.
- **Scan-time camera checks** on the shopper's phone, where the app already uses the camera.

---
*Data: dunnhumby "The Complete Journey" (US grocery, amounts in USD). SmartShop behaviour is simulated and calibrated to published ECR Retail Loss Group figures; see the model card for limitations.*
