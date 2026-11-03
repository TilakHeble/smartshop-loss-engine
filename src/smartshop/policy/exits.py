"""Sessions that never reach the till: genuine cancellations, cancel-and-walk, dead devices (v1.3).

How it happens in store
-----------------------
- **Genuine cancellation.** A shopper scans a few items (or most of them), gets an urgent call or
  can't find what they came for, cancels SmartShop and leaves the items. Nothing is lost. It can
  happen early or late in the shop.
- **Cancel-and-walk.** A shoplifter scans items so the session looks normal, cancels, pays
  nothing, and leaves through the self-scan gates behind a customer who has scanned their receipt
  barcode to open them (easier when it is busy). The whole basket is lost, and no rescan ever
  sees it. It too can happen early or late, so timing does not tell the two apart.
- **Dead device.** The store's handset (or the shopper's phone) dies part-way. Today the customer
  must scan everything again on another handset, at self-scan or at a checkout. A handset is the
  store's equipment, so that is the store's failure, not the customer's.

What v1.3 changes
-----------------
- **Help on cancellation.** Any session cancelled with items scanned sends a colleague near the
  self-scan exit a prompt to help (take the basket back, find the item, show the way to a till).
  It is help, not an accusation: genuine shoppers get help, and a would-be cancel-and-walk meets a
  colleague before the gates. When colleagues are stretched, the prompt can go only to the
  higher-risk cancellations (by the session's risk score), which is where the score earns its keep.
- **Resume after a dead device.** The scanned basket is kept by the system; the shopper carries
  on with another handset, or a colleague or till brings it up from their Nectar card. Nothing is
  scanned twice, and the normal exit decision still applies at the end. A device failure is never
  a risk signal.
- **Fewer dead handsets** (ops/handsets.py): hand out the handset longest on the rack and send
  repeat failures for a new battery, from the handset log alone (the handsets show no battery
  level).

All rates are stated assumptions (no published figures exist), varied in the sensitivity worlds.
Each run draws which sessions cancel, walk, or lose their device, and whether a colleague is
free; outcomes are averaged over runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

TODAY, HELP_ALL, HELP_RISK = "Today", "v1.3: help every cancellation", "v1.3: help higher-risk"
PEAK_HOURS = (11, 12, 13, 16, 17, 18)


@dataclass(frozen=True)
class ExitConfig:
    # cancel-and-walk attempts per session, by shopper type (others never)
    walk_rate: dict = field(
        default_factory=lambda: {"opportunist": 0.01, "deliberate": 0.05, "adaptive": 0.03}
    )
    genuine_cancel_rate: float = 0.03  # every kind of shopper
    phone_failure_rate: float = 0.005  # the shopper's own phone dies mid-shop
    handset_failure_today: float = 0.0064  # from ops/handsets.py: any handset off the rack
    handset_failure_v13: float = 0.0005  # longest on the rack first + retire repeat failures
    tailgate_success_peak: float = 0.9  # getting through the gates behind a paying customer
    tailgate_success_quiet: float = 0.7
    colleague_free_peak: float = 0.6  # a colleague can answer a prompt before the shopper leaves
    colleague_free_quiet: float = 0.8
    intercept: float = 0.8  # a walk-out met by a colleague does not happen
    gate_counts_people: bool = False  # gates flag two people through one receipt scan
    gate_flag_recovers: float = 0.25  # share of tailgates stopped when the gate flags them
    prompt_seconds: float = 60.0
    # dead device: today the customer rescans everything picked so far; v1.3 resumes
    rescan_fixed_seconds: float = 120.0
    rescan_seconds_per_item: float = 4.0
    rescan_colleague_seconds: float = 60.0
    resume_seconds: float = 30.0
    risk_share: float = 0.5  # "higher-risk": cancellations scoring in the top half of sessions

    def walk_scaled(self, k: float) -> ExitConfig:
        return replace(self, walk_rate={t: min(1.0, v * k) for t, v in self.walk_rate.items()})


def simulate_exits(s: pd.DataFrame, arm: str, cc: ExitConfig, seed: int) -> dict:
    """One realisation for one arm.

    s: test-period sessions with gt_type, value, units, hour_of_day, channel, score.
    Draws are the same for every arm with the same seed (common random numbers).
    """
    n = len(s)
    rng = np.random.default_rng(seed)
    u_walk, u_cancel, u_fail, u_stop, u_tail, u_free, u_int, u_gate = rng.random((8, n))
    typ = s.gt_type.to_numpy()
    peak = np.isin(s.hour_of_day.to_numpy(), PEAK_HOURS)
    units = s.units.to_numpy()
    value = s.value.to_numpy()
    handset = s.channel.to_numpy() == "handset"

    walk = u_walk < np.array([cc.walk_rate.get(t, 0.0) for t in typ])
    cancel = ~walk & (u_cancel < cc.genuine_cancel_rate)
    fail_rate = np.where(
        handset,
        cc.handset_failure_today if arm == TODAY else cc.handset_failure_v13,
        cc.phone_failure_rate,
    )
    device = ~walk & ~cancel & (u_fail < fail_rate)
    scanned = np.floor(u_stop * units)  # stop point: early or late, any session
    cancelled_with_items = (walk | cancel) & (scanned > 0)

    # prompts to a colleague near the exit
    if arm == TODAY:
        prompted = np.zeros(n, bool)
    elif arm == HELP_ALL:
        prompted = cancelled_with_items
    else:
        cut = np.quantile(s.score.to_numpy(), 1 - cc.risk_share)
        prompted = cancelled_with_items & (s.score.to_numpy() >= cut)
    free = u_free < np.where(peak, cc.colleague_free_peak, cc.colleague_free_quiet)
    served = prompted & free
    stopped_by_colleague = walk & served & (u_int < cc.intercept)

    tail_ok = u_tail < np.where(peak, cc.tailgate_success_peak, cc.tailgate_success_quiet)
    gate_stop = cc.gate_counts_people & (u_gate < cc.gate_flag_recovers)
    walked_out = walk & ~stopped_by_colleague & tail_ok & ~gate_stop
    walk_loss = np.where(walked_out, value, 0.0)

    if arm == TODAY:
        cust = cc.rescan_fixed_seconds + cc.rescan_seconds_per_item * scanned
        coll_dev = np.full(n, cc.rescan_colleague_seconds)
    else:
        cust = np.full(n, cc.resume_seconds)
        coll_dev = np.full(n, cc.resume_seconds)
    honest_cancel = cancel & (scanned > 0)
    return {
        "arm": arm,
        "walk-out attempts": walk.sum(),
        "walk-outs": walked_out.sum(),
        "walk-out loss": walk_loss.sum(),
        "stopped by a colleague": stopped_by_colleague.sum(),
        "prompts": prompted.sum(),
        "prompts answered": served.sum(),
        "prompts that were walk-outs": (served & walk).sum(),
        "genuine cancellations with items": honest_cancel.sum(),
        "genuine cancellers helped": (served & cancel).sum(),
        "colleague hours on prompts": served.sum() * cc.prompt_seconds / 3600,
        "dead devices": device.sum(),
        "dead handsets": (device & handset).sum(),
        "customer hours lost to dead devices": (device * cust).sum() / 3600,
        "colleague hours on dead devices": (device * coll_dev).sum() / 3600,
    }


def run_arms(s: pd.DataFrame, cc: ExitConfig, seeds=range(20)) -> pd.DataFrame:
    rows = [
        simulate_exits(s, arm, cc, seed) for seed in seeds for arm in (TODAY, HELP_ALL, HELP_RISK)
    ]
    return pd.DataFrame(rows).groupby("arm", sort=False).mean()


def prompt_rates_by_group(
    s: pd.DataFrame, groups: pd.DataFrame, cc: ExitConfig, min_sessions: int = 1000
) -> pd.DataFrame:
    """'Help higher-risk': the share of genuine cancellers a prompt would go to, by group.

    Expected value (no draws): a genuine canceller is prompted if their session scores in the
    top `risk_share`. Being offered help is not an adverse outcome, but being singled out among
    cancellers should still not track a protected characteristic.
    """
    cut = np.quantile(s.score.to_numpy(), 1 - cc.risk_share)
    d = s.assign(prompted=s.score >= cut).merge(groups, on="basket_id")
    d = d[d.gt_type.isin(["honest", "careless"])]
    rows = []
    for col in [c for c in groups.columns if c != "basket_id"]:
        r = d.dropna(subset=[col]).groupby(col).prompted.agg(["mean", "size"])
        r = r[r["size"] >= min_sessions]
        if len(r) > 1:
            rows.append(
                {
                    "attribute": col,
                    "lowest": r["mean"].min(),
                    "highest": r["mean"].max(),
                    "ratio": r["mean"].min() / r["mean"].max(),
                }
            )
    return pd.DataFrame(rows)
