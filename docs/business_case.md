# Rescanning the right SmartShop sessions

*A two-page summary of the SmartShop Loss Engine for a commercial and store-operations audience. Technical detail is in the [README](../README.md) and the [model card](model_card.md).*

## The problem

Scan-as-you-shop is convenient and cheap to run, but items leave unscanned: some on purpose, usually high-value lines, and many by genuine mistake. Published studies of scan-and-go put net loss at roughly 1–5% of its sales (ECR Retail Loss Group, 93 retailers). Today stores protect it with random rescans, about 12% of sessions, mostly partial checks of a few items. Random rescans have three costs:

- **They find little.** Partial checks rarely find items skipped on purpose, because those items are hidden outside the bags. In the simulation, today's rescans recover about \$11 per colleague hour.
- **They stop honest shoppers.** 96% of today's rescans stop someone who skipped nothing on purpose.
- **They can't measure loss properly.** Partial checks put loss at about a ninth of its true level.

## What was built

1. **A calibrated simulator** of SmartShop behaviour on 2.35 million real basket lines (dunnhumby *The Complete Journey*). It matches the published error rates, the deliberate share of loss and today's rescan rate, and it keeps the hidden truth apart from what a store can see.
2. **A session risk model** that reads only what a store already has: the scan log (pauses, pace, deletions, what was scanned) and the member's Nectar history. It learns from what random full rescans found and gives plain-English reasons.
3. **A rescan policy designed against shoppers who learn it.** Rescan probability rises with risk, from a 2% random floor (everyone can be checked) to a 50% cap (no one is checked every time). Regulars with clean rescans earn up to 75% off. Colleague time stays exactly at today's level.
4. **A decision API and weekly job.** The exit decision takes about 25 ms. Every week the model retrains and is published only if it passes budget, drift and fairness gates.
5. **Unbiased loss measurement.** Every decision logs its probability, so total loss can be estimated without bias even though rescans are targeted.

## Results (simulation, test period, same colleague time as today)

| | Today | Recommended | |
|---|---|---|---|
| Loss remaining | \$41,639 | **\$36,767** | **−12%** (2.07% → 1.83% of SmartShop sales) |
| Value recovered by rescans | \$1,196 | **\$6,640** | **5.6x** |
| Recovered per colleague hour | \$11 | **\$59** | |
| Honest shoppers stopped | 5,428 | **1,683** | **−69%** |

- **Two wins at once, at no extra cost:** less loss and a much better experience for honest customers. The top 3.8% of sessions by risk receive 43% of rescans and hold 83% of the value found.
- **It holds up.** In ten stress tests that each change one assumption (more or less theft, weaker signals, bolder or more cautious thieves) and retrain everything, loss is 6–13% lower, recovery 3.9–6.4x higher and honest stops 69–70% lower.
- **As an illustration only:** in the simulation's terms, loss falls by about 0.24% of SmartShop sales, roughly 2,400 per million of SmartShop sales each period. The figure that counts is the one a store trial measures.
- **What didn't earn its place** was dropped: targeted partial checks (full rescans recover several times more per colleague-minute) and in-app "did you forget...?" prompts (usually wrong).

## The trade-offs to decide

| Trade-off | What the evidence says | Lever |
|---|---|---|
| Deterrence vs detection | Today's many cheap partial checks deter cautious opportunists slightly more; the gain from catching more is far larger. Where theft is low or thieves are cautious, the policy gains least (6–8%) | Raise the random floor |
| Repeat stops of honest regulars | Without a cap some honest members would be checked on nearly every trip | Cap 50%, plus the trust discount |
| Fairness | Channel, new users and evening shoppers are treated alike. Every demographic group is stopped far less than today, but younger family households are stopped about 1.6x as often as the oldest when honest, because their baskets are bigger | Decide explicitly whether that is acceptable (age is a protected characteristic); options in the model card |

## How to prove it in real stores

The simulator shows the method works; a trial shows what it is worth.

1. **Randomise stores, not shoppers.** Rescan rules are visible in store, so shoppers in one store must all face the same policy. Match stores in pairs on their baseline loss rate and randomise within each pair.
2. **Same colleague time in both arms.** The recommended policy is recalibrated each week to the arm's own staffing.
3. **Measure loss the same way in both arms**, from random full rescans: today's 2% random full rescans in the control arm, the policy's 2% floor in the trial arm. Primary outcome: loss rate (unscanned value / SmartShop sales). Secondary: honest-shopper stops, recovery per colleague hour, rescan queue time, customer complaints, and the age-group stop rates.
4. **Adjust for each store's baseline period** (CUPED / ANCOVA) and analyse at store level.
5. **Size it.** Each store's loss estimate from a 2% audit has about 23% error over 20,000 SmartShop sessions. To detect a 12% reduction with 80% power, with store loss rates fairly stable over time (baseline correlation 0.7), the trial needs **roughly 40–100 stores per arm**, depending on how much real stores differ. A longer trial or a larger random audit share cuts that. The interactive sizing is on the dashboard.
6. **Start small.** A one-region shadow mode first: the API scores every session and logs what it would have done, while today's random rescans continue. That checks latency, reasons and fairness on real data before any shopper sees a change.

## Risks and how they are handled

| Risk | Control |
|---|---|
| Shoppers learn the rules | Randomised probabilities, a random floor and a cap; policies judged against shoppers who adapt |
| A bad retrain or data change | Weekly gates (labels, budget, floor and cap, drift, parity, score sanity); atomic publishing; the last good bundle keeps serving |
| Unfair treatment | No demographics, channel, time or tenure in the model; parity gates every week; a full demographic review per release |
| Losing sight of true loss | Every decision logs its probability; unbiased weighted estimates, with the random floor as the cross-check |
| Customer trust | Neutral reasons, a colleague makes the final call, clean rescans reduce future checks, and every shopper can still be checked at random |

## What I would do next (v2)

- **Queue-aware capacity:** move rescan probability with exit queues and staffing by hour.
- **Deliberate vs accidental:** use rescan outcomes (concealed vs in-bag) as richer labels; the ceiling suggests large gains.
- **Product hotspots:** learn which products are missed most from the random floor's finds.
- **Scan-time camera checks** on the shopper's phone, where the app already uses the camera.

---
*Data: dunnhumby "The Complete Journey" (US grocery, amounts in USD). SmartShop behaviour is simulated and calibrated to published ECR Retail Loss Group figures; see the model card for limitations.*
