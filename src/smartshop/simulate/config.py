"""Every simulator assumption in one place, with its source.

Calibration targets come from published research on scan-and-go loss. Where no figure is
published, the value is a stated assumption: notebooks/02_simulator.ipynb shows the result and
week 8 tests how sensitive conclusions are to it.

Sources
  [ECR18] Beck, A. (2018) Self-Checkout in Retail: Measuring the Loss. ECR Retail Loss Group.
          13 retailers, 140m scan-and-go transactions, 17m audits. Random full rescans at one
          grocer (~20k audits): 43.4% found at least one error; error likelihood ~30% at 20
          items, ~60% at 50, ~90% at 100; 77% of item errors were non-scans, 23% over-scans;
          net loss up to 4.68% of scan-and-go sales. Partial rescans found an error in only
          2.88% of audits. ~12% of users audited on average (range 2-17%).
  [ECR22] Beck, A. (2022) Global Study on Self-checkout in Retail. ECR Retail Loss Group,
          93 retailers. Share of scan-and-go loss judged malicious: 46% (56% among retailers
          with validation research).
  [SBY]   Sainsbury's SmartShop FAQ: a Nectar card is needed for the app and the handset;
          every item is scanned individually; partial or full rescans happen at checkout.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SimConfig:
    seed: int = 42

    # --- Who uses SmartShop ---------------------------------------------------------------
    # [SBY] Every session is Nectar-linked. Members adopt SmartShop at some point in the window
    # (cold start = new SmartShop users), then use it on a share of trips that grows with basket
    # size (scan-and-go suits larger grocery shops [ECR22]). Assumptions.
    adoption_day_range: tuple[int, int] = (113, 500)
    usage_logit_intercept: float = -0.8
    usage_logit_per_unit: float = 0.12
    app_share: float = 0.55  # app vs handset, per member (assumption)

    # --- Shopper types (member level, hidden) ---------------------------------------------
    # Mix is an assumption, tuned so loss and its malicious share match [ECR18]/[ECR22].
    type_mix: dict = field(
        default_factory=lambda: {
            "honest": 0.72,
            "careless": 0.14,
            "opportunist": 0.08,
            "deliberate": 0.04,
            "adaptive": 0.02,
        }
    )

    # --- Genuine mistakes (every type) ----------------------------------------------------
    # Per-unit rates. [ECR18] P(any error) ~ 1-(1-p)^n with p ~ 1.8% fits 30%/60%/90% at
    # 20/50/100 items; non-scans 77% / over-scans 23% of item errors.
    accidental_nonscan: dict = field(
        default_factory=lambda: {"honest": 0.0068, "careless": 0.017}
    )  # opportunist / deliberate / adaptive use the honest rate
    accidental_overscan: dict = field(
        default_factory=lambda: {"honest": 0.0040, "careless": 0.0085}
    )
    extra_unit_multiplier: float = 2.0  # 2nd+ unit of a multi-buy line is easier to forget
    # Mistakes per unit rise with basket size (a rushed big shop): x (units / 20) ** elasticity.
    # [ECR18]'s curve is steeper than a constant per-unit rate allows (30% -> 90% from 20 to 100).
    size_elasticity: float = 0.35
    member_error_sd: float = 0.25  # lognormal spread of each member's error rate (unobserved)

    # --- Deliberate non-scanning ----------------------------------------------------------
    # Probability a session is a "stealing" session, before deterrence. Assumptions, tuned.
    intent_prob: dict = field(
        default_factory=lambda: {"opportunist": 0.30, "deliberate": 0.55, "adaptive": 1.0}
    )
    # Given intent: per-unit skip probability = base x price factor x risk factor (capped).
    skip_base: dict = field(
        default_factory=lambda: {"opportunist": 0.09, "deliberate": 0.16, "adaptive": 0.16}
    )
    high_risk_skip_factor: float = 2.5
    price_ref: float = 2.0  # price factor = min(price_cap, sqrt(unit price / price_ref))
    price_cap: float = 3.0
    skip_cap: float = 0.9
    scan_then_delete_share: float = 0.1  # deliberate skips done by scanning then deleting

    # Deterrence: an opportunist's intent falls as the rescan probability q rises,
    # x (1 - q) ** opportunist_deterrence. Adaptive shoppers steal only when the expected gain
    # beats the expected cost: V * (1 - q*d) > q * d * penalty.
    opportunist_deterrence: float = 4.0
    adaptive_perceived_detection: float = 0.6
    adaptive_penalty: float = 30.0  # $ equivalent of being caught (embarrassment, ban risk)

    # --- Shopping time and scan behaviour (observable signals; assumptions) ---------------
    minutes_base: float = 3.0
    minutes_per_unit: float = 0.75  # time in store depends on the TRUE basket
    minutes_sd_log: float = 0.35
    honest_delete_rate: float = 0.015  # scan, delete, rescan (changed mind / mistake)
    # Shoppers differ in habits that have nothing to do with honesty: some delete and rescan a
    # lot, some shop slowly. Per-member multipliers (lognormal, mean 1, hidden) stop these habits
    # from becoming a perfect fingerprint once averaged over a member's history.
    member_delete_sd: float = 0.8
    member_pace_sd: float = 0.25
    intent_delete_rate: float = 0.025
    late_departments: tuple = ("MEAT", "SEAFOOD", "DELI", "MEAT-PCKGD")  # picked near the end

    # --- Today's rescans (the historical policy) ------------------------------------------
    # [ECR18] ~12% of users audited (2-17%); partial rescans dominate [ECR22]. Split assumed.
    status_quo_partial_rate: float = 0.10
    status_quo_full_rate: float = 0.02
    partial_items: int = 3  # assumption: colleague checks 3 units picked from the bags

    # --- What a rescan finds and costs (assumptions) --------------------------------------
    full_detect_nonscan: float = 0.97
    full_detect_overscan: float = 0.95
    partial_detect_nonscan: float = 0.97  # if a picked unit was missed by mistake
    # Items skipped on purpose are usually concealed (pocket, bottom of a bag), so a quick
    # partial check rarely picks them; a full rescan empties the bags. Assumption, chosen so
    # partial rescans find errors about as rarely as [ECR18] reports (2.88% of audits).
    partial_detect_deliberate: float = 0.15
    rescan_fixed_seconds: float = 45.0
    full_seconds_per_unit: float = 4.0
    partial_seconds_per_unit: float = 6.0


# Published figures the simulator is checked against in notebooks/02_simulator.ipynb.
TARGETS = {
    "error_prob_20_items": 0.30,  # [ECR18]
    "error_prob_50_items": 0.60,  # [ECR18]
    "error_prob_100_items": 0.90,  # [ECR18]
    "nonscan_share_of_item_errors": 0.77,  # [ECR18]
    "malicious_share_of_loss": (0.46, 0.56),  # [ECR22]
    "net_loss_share_of_sales": (0.0096, 0.0468),  # [ECR18] overall estimate .. full audits
    "partial_rescan_hit_rate": 0.0288,  # [ECR18]
    "audit_rate": 0.12,  # [ECR18]
}
