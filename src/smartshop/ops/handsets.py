"""Dead SmartShop handsets: a store's problem, not the customer's (v1.3).

A SmartShop handset is the store's equipment. If it dies part-way through a shop, today the
customer has to scan everything again on another handset, at self-scan or at a checkout: their
time and a colleague's, for the store's failure.

The handsets do not show a battery level, so a store cannot pick handsets by charge. What it
does have (assumed here, as for any loan device) is a log of which handset each session used,
when it left the rack and when it came back. From that log alone a store can:

- hand out the handset that has been charging longest ("longest on the rack first"), instead of
  whichever one the customer grabs, and
- spot handsets that keep dying early and send them for a new battery.

This module simulates a handset fleet in one busy store, using the real start times and shop
lengths of the test period's handset sessions (all simulated stores stacked onto one store's
timeline, about 100 handset sessions a day), and compares failures per 1,000 sessions.

Battery assumptions (no published figures; stated so they can be checked in a pilot): a healthy
handset runs 8 hours, 15% of the fleet has a worn battery (1-4 hours), every battery loses a
little capacity each day, a handset fully recharges in 3 hours on the rack, and the fleet is
fully charged overnight.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

RANDOM, LONGEST_FIRST, LONGEST_FIRST_RETIRE = "random", "longest on rack first", "+ retire"


@dataclass(frozen=True)
class FleetConfig:
    handsets: int = 40
    healthy_minutes: float = 480.0
    worn_share: float = 0.15
    worn_minutes: tuple = (60.0, 240.0)
    full_charge_minutes: float = 180.0
    fade_per_day: float = 0.001  # share of capacity lost per day of service
    idle_minutes: float = 5.0  # handset on but not yet scanning, per session
    retire_failures: int = 2  # failures within the window that send a handset for a battery
    retire_window_days: int = 30
    seed: int = 7


def store_timeline(sessions: pd.DataFrame) -> pd.DataFrame:
    """Handset sessions as (day, start, duration) in one store, sorted by start."""
    t = sessions[["day", "start_minute", "duration_min"]].copy()
    t["day"] = t.day.astype("int64")  # stored as a small integer: day * 1440 would overflow
    t["start"] = t.day * 1440.0 + t.start_minute.clip(lower=0)
    return t.sort_values("start").reset_index(drop=True)


def simulate(timeline: pd.DataFrame, policy: str, cfg: FleetConfig | None = None) -> dict:
    """Run the fleet through the timeline; returns failures and where in the shop they happened."""
    cfg = cfg or FleetConfig()
    rng = np.random.default_rng(cfg.seed)
    h = cfg.handsets
    worn = rng.random(h) < cfg.worn_share
    capacity = np.where(worn, rng.uniform(*cfg.worn_minutes, h), cfg.healthy_minutes)
    age_days = rng.uniform(0, 365, h)
    charge = capacity.copy()
    back_at = np.zeros(h)  # minute the handset was returned to the rack
    busy_until = np.zeros(h)
    fail_log: list[list[float]] = [[] for _ in range(h)]
    current_day = None
    failures, stop_fraction, sessions_served, no_handset = 0, [], 0, 0

    for day, start, dur in timeline[["day", "start", "duration_min"]].itertuples(index=False):
        if day != current_day:  # overnight: everything back on the rack and fully charged
            current_day = day
            age_days += 1
            if policy == LONGEST_FIRST_RETIRE:
                for i in range(h):
                    recent = [d for d in fail_log[i] if d > day - cfg.retire_window_days]
                    if len(recent) >= cfg.retire_failures:  # send it for a new battery
                        capacity[i], age_days[i], fail_log[i] = cfg.healthy_minutes, 0.0, []
            cap_now = capacity * np.maximum(0.3, 1 - cfg.fade_per_day * age_days)
            charge = cap_now.copy()
            back_at[:] = day * 1440
            busy_until[:] = 0
        cap_now = capacity * np.maximum(0.3, 1 - cfg.fade_per_day * age_days)
        free = np.flatnonzero(busy_until <= start)
        if len(free) == 0:
            no_handset += 1
            continue
        # charge gained on the rack since it came back (the store cannot see this number)
        rack = start - back_at[free]
        level = np.minimum(
            cap_now[free], charge[free] + cap_now[free] * rack / cfg.full_charge_minutes
        )
        if policy == RANDOM:
            pick = rng.integers(len(free))
        else:
            pick = int(np.argmax(rack))  # the only thing the log shows: time on the rack
        i = free[pick]
        need = dur + cfg.idle_minutes
        sessions_served += 1
        if level[pick] < need:
            failures += 1
            stop_fraction.append(max(0.0, (level[pick] - cfg.idle_minutes) / max(dur, 1e-9)))
            fail_log[i].append(day)
            used = level[pick]
        else:
            used = need
        charge[i] = level[pick] - used
        busy_until[i] = start + min(need, level[pick])
        back_at[i] = busy_until[i]
    return {
        "policy": policy,
        "sessions": sessions_served,
        "failures": failures,
        "failures per 1,000 sessions": 1000 * failures / max(sessions_served, 1),
        "mean point of failure (share of shop)": float(np.mean(stop_fraction))
        if stop_fraction
        else np.nan,
        "no handset free": no_handset,
    }


def compare(timeline: pd.DataFrame, cfg: FleetConfig | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        [simulate(timeline, p, cfg) for p in (RANDOM, LONGEST_FIRST, LONGEST_FIRST_RETIRE)]
    )
