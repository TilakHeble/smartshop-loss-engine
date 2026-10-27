"""Sizing the store-level randomised trial that would settle the headline in real stores.

Design: stores are randomised (matched in pairs on their baseline loss rate) to keep today's
random rescans or switch to the recommended policy, at the same colleague time. Outcome: each
store's loss rate (unscanned value / SmartShop sales) over the trial period, measured the same
way in both arms from the random audit sample, adjusted for the store's own baseline period
(CUPED / ANCOVA).

Stores needed per arm, for a relative effect `effect` at significance `alpha` and power `power`:

    n = 2 (z_{1-alpha/2} + z_power)^2 * cv^2 * (1 - rho^2) / effect^2

    cv^2  = between-store variation in the loss rate (as a coefficient of variation)
          + measurement noise of each store's estimate
    rho   = correlation of a store's loss rate between the baseline and trial periods

The simulator's stores are too small to estimate the between-store spread (a handful of
households drive each store's loss), so it is an explicit assumption, shown as a range. The
measurement noise is not assumed: it comes from week 8 (inverse-probability weighting had a 9%
error over the whole test period) and scales with one over the square root of sessions.
"""

from __future__ import annotations

import math
from statistics import NormalDist

import pandas as pd


def stores_per_arm(
    effect: float, cv: float, rho: float, alpha: float = 0.05, power: float = 0.8
) -> int:
    z = NormalDist().inv_cdf(1 - alpha / 2) + NormalDist().inv_cdf(power)
    return math.ceil(2 * z**2 * cv**2 * (1 - rho**2) / effect**2)


def measurement_cv(sessions_per_store: float, chain_error: float, chain_sessions: float) -> float:
    """Error of one store's weighted loss estimate, scaled from the whole-chain error."""
    return chain_error * math.sqrt(chain_sessions / sessions_per_store)


def sizing_table(
    effect: float,
    chain_error: float,
    chain_sessions: float,
    sessions_per_store: float = 20_000,
    between_cvs=(0.15, 0.25, 0.35),
    rhos=(0.0, 0.5, 0.7, 0.85),
) -> pd.DataFrame:
    meas = measurement_cv(sessions_per_store, chain_error, chain_sessions)
    rows = []
    for b in between_cvs:
        cv = math.sqrt(b**2 + meas**2)
        rows.append(
            {
                "between-store CV": b,
                "total CV": cv,
                **{f"rho={r}": stores_per_arm(effect, cv, r) for r in rhos},
            }
        )
    return pd.DataFrame(rows).set_index("between-store CV")
