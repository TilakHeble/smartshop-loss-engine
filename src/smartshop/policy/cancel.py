"""Cancelled SmartShop sessions: genuine ones, and cancelling to dodge a rescan (v1.2).

A shopper can cancel a SmartShop session part-way through (phone battery, app crash, changed
their mind) and pay at a staffed till instead. That is ordinary. But a session can also be
cancelled to dodge a rescan: told "you've been selected", the shopper cancels and pays at a till
for what they put on the belt, keeping whatever they hid. If cancelling has no consequence, any
rescan policy can be dodged, and loss measured from rescans quietly misses the sessions most
likely to hold it.

Two sets of rules are compared:

    open (today)   a mid-shop cancellation skips the exit decision; a shopper selected for a
                   rescan can cancel and pay at a till
    closed (v1.2)  the session locks once a rescan is requested (a colleague completes it);
                   a cancellation is treated as an exit and gets the same decision, except a
                   cancellation the app itself logged as a crash or flat battery, which gets
                   only the random floor; some would-be dodgers abandon the trolley and leave

At a staffed till everything put on the belt is scanned, so honest mistakes are paid for and
double scans disappear; only items deliberately hidden are lost.

The economics are expected values over each session's chance of cancelling and of being
rescanned, on the rescan game's versions of reality, and shoppers who steal respond to the
rescan chance they effectively face (so an open loophole invites more theft).
Numbers here are stated assumptions (no published figure exists); week-8-style sensitivity
runs vary them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from smartshop.policy.game import GameData, responses

OPEN, CLOSED = "open", "closed"


@dataclass(frozen=True)
class CancelConfig:
    # Genuine mid-shop cancellations, every kind of shopper (assumption).
    midshop_rate: float = 0.03
    # Share of those the app logs itself as a crash or flat battery (observable, not claimable).
    app_error_share: float = 0.4
    # Chance of cancelling when told "rescan", for a session where items are hidden (by type),
    # and for anyone else (in a hurry, embarrassed). Assumptions.
    avoid_when_stealing: dict = field(
        default_factory=lambda: {"opportunist": 0.4, "deliberate": 0.6, "adaptive": 0.6}
    )
    avoid_otherwise: float = 0.02
    # Under the lock, would-be dodgers who abandon the trolley and leave (hidden items lost).
    walk_away_share: float = 0.15
    redirect_seconds: float = 30.0  # colleague time to handle a cancellation

    def scaled(self, k: float) -> CancelConfig:
        """The same world with dodging k times as likely (capped at 1)."""
        return replace(
            self,
            avoid_when_stealing={t: min(1.0, v * k) for t, v in self.avoid_when_stealing.items()},
        )


def _avoid_matrix(g: GameData, cc: CancelConfig) -> np.ndarray:
    """(versions x sessions) chance of cancelling when selected."""
    n_v = len(g.levels) + 1
    typ = g.sessions.gt_type.to_numpy()
    stealing = np.array([cc.avoid_when_stealing.get(t, cc.avoid_otherwise) for t in typ])
    a = np.vstack([np.full(len(typ), cc.avoid_otherwise)] + [stealing] * (n_v - 1))
    # a version with nothing hidden is honest whatever the type
    return np.where(g.matrix("deliberate_value") > 0, a, cc.avoid_otherwise)


def outcome_with_cancellations(
    g: GameData,
    policy,
    rules: str,
    cc: CancelConfig,
    today: bool = False,
    today_scale: float = 1.0,
) -> dict:
    """Expected outcomes when shoppers can cancel, under open or closed rules.

    policy: score -> full-rescan probability (ignored when today=True: today's random partial
    and full rescans, with both rates multiplied by today_scale).
    """
    cfg = g.cfg
    n_v, n = len(g.levels) + 1, len(g.sessions)
    a = _avoid_matrix(g, cc)
    m, e, w = cc.midshop_rate, cc.app_error_share, cc.walk_away_share
    floor = cfg.status_quo_full_rate  # the random floor of full rescans (2%)

    if today:
        p_full = np.full((n_v, n), cfg.status_quo_full_rate * today_scale)
        p_part = np.full((n_v, n), cfg.status_quo_partial_rate * today_scale)
    else:
        p_full = np.vstack([policy(g.sessions[f"v{k}_score"].to_numpy()) for k in range(n_v)])
        p_part = np.zeros_like(p_full)
    p_any = np.clip(p_full + p_part, 0, 1)

    # The rescan chance a shopper effectively faces, which drives whether and how much they steal.
    if rules == OPEN:
        q_eff = (1 - m) * p_any * (1 - a)
    else:
        q_eff = p_any * (1 - a * w)
    choice = responses(g, None, q_matrix=q_eff)
    cols = np.arange(n)
    pick = lambda M: M[choice, cols]  # noqa: E731

    net_loss, hidden = pick(g.matrix("net_loss")), pick(g.matrix("deliberate_value"))
    full_find, part_find = pick(g.matrix("full_find")), pick(g.matrix("partial_find"))
    units = pick(g.matrix("true_units"))
    pf, pp, pa, av = pick(p_full), pick(p_part), pick(p_any), pick(a)
    # expected value found by a completed rescan, given one is requested
    find = np.where(pa > 0, (pf * full_find + pp * part_find) / np.maximum(pa, 1e-12), 0.0)
    full_cost = cfg.rescan_fixed_seconds + cfg.full_seconds_per_unit * units
    part_cost = cfg.rescan_fixed_seconds + cfg.partial_seconds_per_unit * np.minimum(
        cfg.partial_items, units
    )
    cost = np.where(pa > 0, (pf * full_cost + pp * part_cost) / np.maximum(pa, 1e-12), 0.0)
    r = cc.redirect_seconds

    if rules == OPEN:
        # mid-shop cancel: no decision, pay at a till (hidden items lost)
        # reach the exit: if selected, cancel with chance a (till), else complete the rescan
        loss = m * hidden + (1 - m) * (
            (1 - pa) * net_loss + pa * (av * hidden + (1 - av) * (net_loss - find))
        )
        rescans_done = (1 - m) * pa * (1 - av)
        dodged = (1 - m) * pa * av
        seconds = rescans_done * cost + (m + dodged) * r
        asked = (1 - m) * pa
        cancel_asked = np.zeros(n)
    else:
        # a cancellation is an exit: crash/battery cancellations get the floor only
        pa_cancel = e * np.minimum(pa, floor) + (1 - e) * pa
        complete_find = net_loss - find
        at_exit = (1 - pa) * net_loss + pa * (av * w * hidden + (1 - av * w) * complete_find)
        on_cancel = (1 - pa_cancel) * hidden + pa_cancel * (
            av * w * hidden + (1 - av * w) * complete_find
        )
        loss = (1 - m) * at_exit + m * on_cancel
        rescans_done = (1 - m) * pa * (1 - av * w) + m * pa_cancel * (1 - av * w)
        dodged = ((1 - m) * pa + m * pa_cancel) * av * w  # walked away
        seconds = rescans_done * cost + (m + dodged) * r
        asked = (1 - m) * pa + m * pa_cancel
        cancel_asked = pa_cancel

    honest = hidden == 0
    typ = g.sessions.gt_type.to_numpy()
    return {
        "loss_remaining": loss.sum(),
        # hidden items kept by shoppers who dodged a requested rescan (or walked away)
        "hidden_loss_in_dodged_rescans": (dodged * hidden).sum(),
        "rescans_dodged_or_refused": dodged.sum(),
        "colleague_hours": seconds.sum() / 3600,
        "honest_stops": (asked * honest).sum(),
        "genuine_cancellers_stopped": (m * cancel_asked * honest).sum(),
        "genuine_cancellations": m * honest.sum(),
        "sessions_with_theft": (hidden > 0).sum(),
        "adaptive_stealing": (choice[typ == "adaptive"] > 0).mean(),
        "opportunists_stealing": (choice[typ == "opportunist"] > 0).mean(),
        "loss_before_rescans": net_loss.sum(),
        "rescans_done": rescans_done.sum(),
    }


def fit_hours(
    evaluate, hours: float, lo: float, hi: float, decreasing: bool = True, iters: int = 40
) -> float:
    """Bisect x so evaluate(x)["colleague_hours"] meets `hours` (time falls as x rises when
    `decreasing`)."""
    for _ in range(iters):
        mid = (lo + hi) / 2
        over = evaluate(mid)["colleague_hours"] > hours
        if over == decreasing:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
