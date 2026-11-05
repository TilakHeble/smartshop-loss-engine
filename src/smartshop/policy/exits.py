"""Sessions that never reach the till: genuine cancellations, cancel-and-walk, dead devices (v1.5).

How it happens in store
-----------------------
- **Genuine cancellation.** A shopper scans a few items (or most of them), gets an urgent call or
  can't find what they came for, cancels SmartShop and leaves the items. Nothing is lost. It can
  happen early or late in the shop.
- **Cancel-and-walk.** A shoplifter scans items so the session looks normal, cancels, pays
  nothing, and leaves through the self-scan gates behind a customer who has scanned their receipt
  barcode to open them (easier when it is busy). The whole basket is lost, and no check ever
  sees it. It too can happen early or late, so timing does not tell the two apart.
- **Dead device.** The store's handset (or the shopper's phone) dies part-way. Today the customer
  must scan everything again on another handset, at self-scan or at a checkout. A handset is the
  store's equipment, so that is the store's failure, not the customer's.

What the store has: colleagues carry no device. The SmartShop check device (at the SmartShop
checkout point) logs in and scans the barcode a shopper gets when there is a check. Customers
pick their own handset from the stand, or use their phone, and start with their Nectar card.

What v1.5 changes (v1.3's colleague prompt needed a device colleagues don't have)
--------------------------------------------------------------------------------
- **Idea 1: finish a cancel at the SmartShop point.** Cancelling with items scanned no longer
  just ends the session: the handset or app asks the shopper to leave the basket at the
  SmartShop checkout and shows a cancel barcode, which a colleague scans on the check device.
  Genuine shoppers who do it are helped and the items go back; a cancel never finished there is
  an "unfinished cancel", a signal the store can count for the first time.
- **Idea 2: an unfinished cancel brings a full check next time.** SmartShop starts with a Nectar
  card, so the member is known. Their next SmartShop shop that reaches the till gets a full
  check, with a neutral reason ("your last SmartShop shop wasn't finished"). Two rules: after one
  unfinished cancel, or only after two in 90 days (fewer honest shoppers checked). Whether it
  also deters cancel-and-walk is unmeasured: the base assumes it does not, and other worlds test
  25% and 50%.
- **Idea D: resume after a dead device.** The scanned basket is kept; the shopper carries on with
  another handset, or a till brings it up from their Nectar card. Nothing is scanned twice, the
  normal till decision still applies, and a device failure is never a risk signal.
- **Fewer dead handsets** (ops/handsets.py): replace batteries that keep dying, and a handset back
  on the stand only briefly asks the customer to take another.

All rates are stated assumptions (no published figures exist), varied in the sensitivity worlds.
Each run draws which sessions cancel, walk or lose their device; outcomes are averaged over runs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

TODAY = "Today"
FINISH = "1: finish cancels at the SmartShop point"
FLAG_ONE = "1+2: full check after an unfinished cancel"
FLAG_TWO = "1+2: full check after two in 90 days"
ARMS = (TODAY, FINISH, FLAG_ONE, FLAG_TWO)
PEAK_HOURS = (11, 12, 13, 16, 17, 18)


@dataclass(frozen=True)
class ExitConfig:
    # cancel-and-walk attempts per session, by shopper type (others never)
    walk_rate: dict = field(
        default_factory=lambda: {"opportunist": 0.01, "deliberate": 0.05, "adaptive": 0.03}
    )
    genuine_cancel_rate: float = 0.03  # every kind of shopper
    genuine_finish: float = 0.7  # genuine cancellers who leave the basket at the SmartShop point
    flag_deters: float = 0.0  # cancel-and-walk attempts put off by idea 2 (unmeasured)
    flag_window_days: float = 90.0
    phone_failure_rate: float = 0.005  # the shopper's own phone dies mid-shop
    handset_failure_today: float = 0.0064  # from ops/handsets.py: customers pick any handset
    handset_failure_v15: float = 0.002  # replace worn batteries + "take another" on the screen
    tailgate_success_peak: float = 0.9  # getting through the gates behind a paying customer
    tailgate_success_quiet: float = 0.7
    gate_counts_people: bool = False  # gates flag two people through one receipt scan (idea 3)
    gate_flag_recovers: float = 0.25  # share of tailgates stopped when the gate flags them
    finish_seconds: float = 60.0  # colleague scans the cancel barcode and takes the basket
    full_detect_nonscan: float = 0.97
    full_detect_overscan: float = 0.95
    check_fixed_seconds: float = 45.0
    check_seconds_per_unit: float = 4.0
    # dead device: today the customer rescans everything picked so far; v1.5 resumes
    rescan_fixed_seconds: float = 120.0
    rescan_seconds_per_item: float = 4.0
    rescan_colleague_seconds: float = 60.0
    resume_seconds: float = 30.0

    def walk_scaled(self, k: float) -> ExitConfig:
        return replace(self, walk_rate={t: min(1.0, v * k) for t, v in self.walk_rate.items()})


def _flags(member, t, unfinished, checkout, window: float, need: int) -> np.ndarray:
    """Checkout sessions that get a full check under idea 2.

    A checkout session is flagged when the member has had an unfinished cancel since their last
    checkout, and at least `need` unfinished cancels in the `window` days before it. Sessions must
    be sorted by member, then time.
    """
    n = len(t)
    c = np.concatenate([[0], np.cumsum(unfinished)])  # c[i]: unfinished before session i
    key = member.astype(np.float64) * 1e6 + t
    lo = np.searchsorted(key, key - window, side="left")
    in_window = c[np.arange(n)] - c[lo]
    # unfinished cancels since the member's previous checkout (or since their first session)
    first = np.searchsorted(member, member, side="left")  # member's first row (sorted)
    last_co = pd.Series(np.where(checkout, np.arange(n), np.nan)).ffill().shift(1).to_numpy()
    prev = np.where(np.isnan(last_co), -1, last_co).astype(int)
    prev = np.where(prev >= first, prev, -1)
    since = c[np.arange(n)] - np.where(prev >= 0, c[prev + 1], c[first])
    return checkout & (since > 0) & (in_window >= need)


def simulate_exits(s: pd.DataFrame, arm: str, cc: ExitConfig, seed: int) -> dict:
    """One realisation for one arm.

    s: test-period sessions sorted by member then time, with member_id, t (days), gt_type,
    value, units, hour_of_day, channel, nonscan_value, overscan_value, deliberate_value.
    Draws are the same for every arm with the same seed (common random numbers).
    """
    n = len(s)
    rng = np.random.default_rng(seed)
    u_walk, u_cancel, u_fail, u_stop, u_tail, u_gate, u_finish, u_deter = rng.random((8, n))
    typ = s.gt_type.to_numpy()
    peak = np.isin(s.hour_of_day.to_numpy(), PEAK_HOURS)
    units = s.units.to_numpy()
    value = s.value.to_numpy()
    handset = s.channel.to_numpy() == "handset"
    flagging = arm in (FLAG_ONE, FLAG_TWO)

    attempt = u_walk < np.array([cc.walk_rate.get(t, 0.0) for t in typ])
    deterred = attempt & flagging & (u_deter < cc.flag_deters)
    walk = attempt & ~deterred
    cancel = ~attempt & (u_cancel < cc.genuine_cancel_rate)
    fail_rate = np.where(
        handset,
        cc.handset_failure_today if arm == TODAY else cc.handset_failure_v15,
        cc.phone_failure_rate,
    )
    device = ~walk & ~cancel & (u_fail < fail_rate)
    scanned = np.floor(u_stop * units)  # stop point: early or late, any session
    with_items = scanned > 0

    # idea 1: genuine cancellers leave the basket at the SmartShop point; a walk-out never does
    finished = (arm != TODAY) & cancel & with_items & (u_finish < cc.genuine_finish)
    unfinished = (walk | cancel) & with_items & ~finished

    tail_ok = u_tail < np.where(peak, cc.tailgate_success_peak, cc.tailgate_success_quiet)
    gate_stop = cc.gate_counts_people & (u_gate < cc.gate_flag_recovers)
    walked_out = walk & tail_ok & ~gate_stop
    walk_loss = np.where(walked_out, value, 0.0)

    # idea 2: the member's next shop that reaches the till gets a full check
    checkout = ~walk & ~cancel  # deterred attempts shop normally; dead devices resume
    if flagging:
        need = 1 if arm == FLAG_ONE else 2
        flag = _flags(
            s.member_id.to_numpy(), s.t.to_numpy(), unfinished, checkout, cc.flag_window_days, need
        )
    else:
        flag = np.zeros(n, bool)
    found = (
        cc.full_detect_nonscan * s.nonscan_value.to_numpy()
        - cc.full_detect_overscan * s.overscan_value.to_numpy()
    )
    honest = s.deliberate_value.to_numpy() == 0
    check_s = cc.check_fixed_seconds + cc.check_seconds_per_unit * units

    if arm == TODAY:
        cust = cc.rescan_fixed_seconds + cc.rescan_seconds_per_item * scanned
        coll_dev = np.full(n, cc.rescan_colleague_seconds)
    else:
        cust = np.full(n, cc.resume_seconds)
        coll_dev = np.full(n, cc.resume_seconds)
    honest_cancel = cancel & with_items
    honest_members_flagged = pd.Series(flag & honest).groupby(s.member_id.to_numpy()).any()
    return {
        "arm": arm,
        "walk-out attempts": walk.sum(),
        "attempts put off": deterred.sum(),
        "walk-outs": walked_out.sum(),
        "walk-out loss": walk_loss.sum(),
        "genuine cancellations with items": honest_cancel.sum(),
        "genuine cancels finished at the SmartShop point": finished.sum(),
        "unfinished cancels": unfinished.sum() if arm != TODAY else np.nan,
        "unfinished cancels that were walk-outs": (unfinished & walk).sum()
        if arm != TODAY
        else np.nan,
        "colleague hours finishing cancels": finished.sum() * cc.finish_seconds / 3600,
        "full checks from unfinished cancels": flag.sum(),
        "of which honest shoppers": (flag & honest).sum(),
        "honest members checked this way": honest_members_flagged.sum(),
        "recovered by these checks": (flag * found).sum(),
        "colleague hours on these checks": (flag * check_s).sum() / 3600,
        "dead devices": device.sum(),
        "dead handsets": (device & handset).sum(),
        "customer hours lost to dead devices": (device * cust).sum() / 3600,
        "colleague hours on dead devices": (device * coll_dev).sum() / 3600,
    }


def run_arms(s: pd.DataFrame, cc: ExitConfig, seeds=range(20)) -> pd.DataFrame:
    rows = [simulate_exits(s, arm, cc, seed) for seed in seeds for arm in ARMS]
    return pd.DataFrame(rows).groupby("arm", sort=False).mean()


def flag_rates_by_group(
    s: pd.DataFrame, groups: pd.DataFrame, cc: ExitConfig, arm: str, seeds=range(20)
) -> pd.DataFrame:
    """Idea 2: the share of honest members who ever get a full check from an unfinished cancel.

    Honest members: those who never take anything on purpose. Being checked is the harm, so the
    rate should not track a protected characteristic (four-fifths ratio across groups).
    """
    hits = []
    honest_members = s.groupby("member_id").deliberate_value.max().eq(0)
    honest_members = honest_members[honest_members].index
    for seed in seeds:
        out = _flagged_members(s, arm, cc, seed)
        hits.append(pd.Series(1.0, index=list(out)).reindex(honest_members, fill_value=0.0))
    rate = pd.concat(hits, axis=1).mean(axis=1).rename("rate")
    g = groups.drop_duplicates("member_id").set_index("member_id")
    d = g.join(rate, how="inner")
    rows = []
    for col in g.columns:
        r = d.dropna(subset=[col]).groupby(col).rate.agg(["mean", "size"])
        r = r[r["size"] >= 20]
        if len(r) > 1 and r["mean"].max() > 0:
            rows.append(
                {
                    "attribute": col,
                    "lowest": r["mean"].min(),
                    "highest": r["mean"].max(),
                    "ratio": r["mean"].min() / r["mean"].max(),
                }
            )
    return pd.DataFrame(rows)


def _flagged_members(s: pd.DataFrame, arm: str, cc: ExitConfig, seed: int) -> set:
    n = len(s)
    rng = np.random.default_rng(seed)
    u_walk, u_cancel, _, u_stop, _, _, u_finish, u_deter = rng.random((8, n))
    typ = s.gt_type.to_numpy()
    attempt = u_walk < np.array([cc.walk_rate.get(t, 0.0) for t in typ])
    walk = attempt & ~(u_deter < cc.flag_deters)
    cancel = ~attempt & (u_cancel < cc.genuine_cancel_rate)
    with_items = np.floor(u_stop * s.units.to_numpy()) > 0
    finished = cancel & with_items & (u_finish < cc.genuine_finish)
    unfinished = (walk | cancel) & with_items & ~finished
    need = 1 if arm == FLAG_ONE else 2
    flag = _flags(
        s.member_id.to_numpy(),
        s.t.to_numpy(),
        unfinished,
        ~walk & ~cancel,
        cc.flag_window_days,
        need,
    )
    honest = s.deliberate_value.to_numpy() == 0
    return set(s.member_id.to_numpy()[flag & honest])
