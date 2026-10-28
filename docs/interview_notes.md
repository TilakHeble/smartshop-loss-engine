# Interview notes: SmartShop Loss Engine

*How to explain the project in 30 seconds, in 2 minutes, and under pressure. Numbers are from the simulation (test period, same colleague time as today).*

## The 30-second version

> SmartShop loses money when items leave unscanned, and today about 12% of sessions are rescanned at random. I built a system that decides who to rescan from the scan log and the shopper's Nectar history, while keeping a small random floor. In a simulation calibrated to published loss research it cuts loss by about 20%, recovers seven times as much, and stops 65% fewer honest shoppers, with the same colleague time.

## The 2-minute version

1. **Problem.** Random rescans mostly stop honest people (96% of them) and find little. And once rescans are targeted, you can no longer measure loss by averaging what they find.
2. **Data.** SmartShop data isn't public, so I simulated SmartShop behaviour on 2.35 million real basket lines from dunnhumby, calibrated to the ECR Retail Loss Group studies: error rates by basket size, the share of loss that's deliberate, and loss at 2% of sales. Truth and observations are kept apart, so models only see what a store sees.
3. **Model.** A logistic regression on 25 features: pauses, pace, deletions, and how this shop compares with the member's usual Nectar basket. It's trained only on what *random* full rescans found, because those are the only unbiased labels.
4. **Policy.** Rescan probability rises with risk, from a 2% random floor to a 50% cap, with a trust discount for regulars with clean rescans. I evaluated it against shoppers who learn the policy and respond, not static ones: that matters, because static evaluation overstated results by 20–50%.
5. **Proof it's robust.** Fourteen stress tests, including a theft method the model was never designed for (barcode switching). It beats today in every one.
6. **Fairness.** The fairness check found younger families were stopped more because their baskets are bigger. Scoring each session against baskets of a similar size narrowed the gap *and* caught more.
7. **Production.** A 25 ms FastAPI service in Docker, a weekly retrain that only publishes through drift and fairness gates, and a live Streamlit dashboard.

## Numbers to remember

| | |
|---|---|
| Rescanned today | ~12% of sessions, at random; \$11 recovered per colleague hour |
| Recommended policy (v1.1) | **−20% loss**, **7.1x recovered** (\$75 per colleague hour), **−65% honest shoppers stopped** |
| Stress tests | 14 scenarios, loss 6–22% lower in every one |
| Unseen method (barcode switching, every thief, no retraining) | still 5.7% less loss than today |
| Measuring loss | averaging targeted rescans overstates loss ~4x; weighting by 1/probability is unbiased (~1%) |
| Fairness | every group stopped at under half today's rate; age ratio 0.61 → 0.69 with v1.1 |
| Store trial | roughly 40–100 stores per arm |

## Hard questions

**"It's all simulated. How do you know it works?"**
I don't claim it would cut Sainsbury's loss by 20%. I claim the *method* recovers a known truth in a world calibrated to published figures, and that the conclusion survives when the assumptions are wrong (14 stress tests). The real test is a store-level randomised trial, which I sized: about 40–100 stores per arm, with a shadow-mode pilot first where the system decides but nothing changes for shoppers.

**"Your model just finds the signals you put in the simulator."**
That's the right worry, so I tested it with barcode switching. The count, pace and deletions all look normal, so the scan-log signals vanish. With every thief switching and no retraining, it still beats today (5.7% less loss), because member history still points at it, and the random floor keeps finding it so the weekly retrain can learn.

**"Why not just rescan the top 10% riskiest sessions?"**
Two reasons. A fixed threshold can be learned and gamed; a random draw with a floor can't, and anyone can be checked. And the floor gives unbiased labels and loss measurement. Interestingly, I found randomisation itself adds little (at most 1.4%). The gain is detection, not deterrence, and I report that honestly.

**"How do you get training labels when rescans are targeted?"**
Only from the random full rescans: today's 2% random full rescans, and the policy's 2% floor. Targeted rescans are a biased sample. For total loss I weight each rescan by one over the probability it had. That's inverse-probability weighting, and it's why the API logs the probability with every decision.

**"Why logistic regression, not XGBoost?"**
There were fewer than 1,000 labels in the training period. I compared candidates by repeated cross-validation and the leaders were close, so I chose the one that gives plain-English reasons a colleague can use, like "long pause between scans".

**"Why did v1.1 catch more when it was meant to be fairer?"**
The model learned "will a rescan find *anything*", and big baskets nearly always contain an honest slip, so they scored high. But they're also the slowest to rescan. Comparing like with like moved rescans to sessions that look unusual for their size: quicker rescans, more of them, more of them on theft. I chose the strength on the tuning period, not the test period.

**"Is this legal? It's profiling customers."**
It needs a DPIA. Crime prevention is now a recognised legitimate interest under the Data (Use and Access) Act 2025, but catching honest mistakes still needs a balancing test. A colleague makes the final call, every decision is explained and logged, and no demographics, channel or time of day go into the model. Age is a protected characteristic, so the remaining age gap is a decision the business should make and record.

**"What would you do differently?"**
I'd get richer labels: whether a missing item was concealed or in the bag. A model trained on *why* items were missing reached 82% of deliberate loss against 53%, so label quality is the bottleneck, not the algorithm. I'd also make capacity depend on queues by hour.

**"The data is American, in dollars."**
Yes, and it's labelled everywhere. The method doesn't depend on currency. What would change with Sainsbury's data is the calibration, which is exactly what the shadow-mode pilot is for.

## Techniques in one line each

- **Stackelberg game:** the store commits to a policy first; shoppers see it and choose their best response.
- **Inverse-probability weighting:** a rescan that had a 10% chance of happening stands for 10 sessions, which makes loss estimates unbiased.
- **Four-fifths rule:** the lowest group's rate divided by the highest; below 0.8 needs review.
- **Permutation test:** shuffle group labels to see how big a gap chance alone makes. With 11 small income groups, chance alone gives 0.61.
- **PSI (population stability index):** how far this week's data has drifted from what the model was calibrated on; 0.25 or more blocks publication.
- **CUPED / ANCOVA:** adjust each store's trial result for its own baseline, which cuts the noise and the stores needed.

## If you can show the dashboard (2 minutes)

1. **Overview:** the four numbers, and v1.1 against v1.0.
2. **Policy:** move the risk slider and the "clean rescans" selector; the probability never drops below 2%.
3. **Fairness:** pick *age group*; scroll to "the fix" table.
4. **Robustness:** the barcode-switching bars at the top: weaker, never negative.
5. **Measuring loss & trial:** change the effect size and watch the number of stores.
