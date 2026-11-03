"""Cancelled SmartShop sessions: genuine ones, and cancelling to skip the rescan check (v1.2).

How SmartShop works at the end of a shop: the shopper scans the barcode at the SmartShop till,
and that is when a rescan may be requested. Once the barcode is scanned the session cannot be
cancelled. Before it, a shopper can cancel SmartShop and pay at a staffed till instead.

Usually that is ordinary: a flat battery, an app crash, a change of mind. But a shopper who has
hidden items can also cancel just before the till, pay at a staffed till for what they put on the
belt, keep what they hid, and never face any chance of a rescan. If a cancellation skips the
decision, any rescan policy can be side-stepped, and loss measured from rescans quietly misses the
sessions most likely to hold it.

Two sets of rules are compared:

    open (today)   a cancellation skips the exit decision: pay at a staffed till
    closed (v1.2)  a cancellation is treated as an exit and gets the same decision (scored on the
                   scan log so far); a cancellation the app itself logged as a crash or flat
                   battery gets only the random floor

Under both, once the barcode is scanned and a rescan requested, the session cannot be cancelled;
a shopper with hidden items may still abandon the trolley and walk out (their hidden items are
lost), and the store logs it.

At a staffed till everything put on the belt is scanned, so honest mistakes are paid for and
double scans disappear; only items deliberately hidden are lost.

The economics are expected values over each session's chance of cancelling and of being
rescanned, on the rescan game's versions of reality, and shoppers who steal respond to the
rescan chance they effectively face (so an open loophole invites more theft). The rates are
stated assumptions (no published figure exists); the sensitivity worlds vary them.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from smartshop.policy.game import GameData, responses

OPEN, CLOSED = "open", "closed"


@dataclass(frozen=True)
class CancelConfig:
    # Genuine cancellations before the till, every kind of shopper (assumption).
    midshop_rate: float = 0.03
    # Share of those the app logs itself as a crash or flat battery (observable, not claimable).
    app_error_share: float = 0.4
    # Chance a shopper with hidden items cancels before the till to skip the check, by type
    # (assumption). Under closed rules this gains them nothing: they get the same decision.
    cancel_when_stealing: dict = field(
        default_factory=lambda: {"opportunist": 0.3, "deliberate": 0.5, "adaptive": 0.5}
    )
    # Chance a shopper with hidden items, once a rescan is requested, abandons the trolley and
    # walks out instead (both rules; hidden items lost; logged by the store).
    walk_away_when_selected: float = 0.1
    redirect_seconds: float = 30.0  # colleague time to handle a cancellation

    def scaled(self, k: float) -> CancelConfig:
        """The same world with cancelling-to-skip k times as likely (capped at 1)."""
        return replace(
            self,
            cancel_when_stealing={t: min(1.0, v * k) for t, v in self.cancel_when_stealing.items()},
        )


def _stealing_matrix(g: GameData, by_type: dict, default: float = 0.0) -> np.ndarray:
    """(versions x sessions): the type's value where a version hides items, else `default`."""
    n_v = len(g.levels) + 1
    typ = g.sessions.gt_type.to_numpy()
    v = np.array([by_type.get(t, default) for t in typ])
    return np.where(g.matrix("deliberate_value") > 0, np.vstack([v] * n_v), default)


def cancel_matrix(g: GameData, cc: CancelConfig) -> np.ndarray:
    """(versions x sessions) chance of cancelling before the till."""
    skip = _stealing_matrix(g, cc.cancel_when_stealing)
    return cc.midshop_rate + (1 - cc.midshop_rate) * skip


def walk_matrix(g: GameData, cc: CancelConfig) -> np.ndarray:
    """(versions x sessions) chance of walking out when a rescan is requested."""
    return np.where(g.matrix("deliberate_value") > 0, cc.walk_away_when_selected, 0.0)


def effective_rescan(p_any, kappa, walk, rules: str):
    """The chance a rescan actually happens, which drives whether and how much shoppers steal."""
    if rules == OPEN:
        return (1 - kappa) * p_any * (1 - walk)
    return p_any * (1 - walk)  # a cancellation gets the same decision


def outcome_with_cancellations(
    g: GameData,
    policy,
    rules: str,
    cc: CancelConfig,
    today: bool = False,
    today_scale: float = 1.0,
) -> dict:
    """Expected outcomes when shoppers can cancel before the till, under open or closed rules.

    policy: score -> full-rescan probability (ignored when today=True: today's random partial
    and full rescans, with both rates multiplied by today_scale).
    """
    cfg = g.cfg
    n_v, n = len(g.levels) + 1, len(g.sessions)
    kappa_all, walk_all = cancel_matrix(g, cc), walk_matrix(g, cc)
    m, e = cc.midshop_rate, cc.app_error_share
    floor = cfg.status_quo_full_rate  # the random floor of full rescans (2%)

    if today:
        p_full = np.full((n_v, n), cfg.status_quo_full_rate * today_scale)
        p_part = np.full((n_v, n), cfg.status_quo_partial_rate * today_scale)
    else:
        p_full = np.vstack([policy(g.sessions[f"v{k}_score"].to_numpy()) for k in range(n_v)])
        p_part = np.zeros_like(p_full)
    p_any = np.clip(p_full + p_part, 0, 1)

    choice = responses(g, None, q_matrix=effective_rescan(p_any, kappa_all, walk_all, rules))
    cols = np.arange(n)
    pick = lambda M: M[choice, cols]  # noqa: E731

    net_loss, hidden = pick(g.matrix("net_loss")), pick(g.matrix("deliberate_value"))
    full_find, part_find = pick(g.matrix("full_find")), pick(g.matrix("partial_find"))
    units = pick(g.matrix("true_units"))
    pf, pp, pa = pick(p_full), pick(p_part), pick(p_any)
    kappa, walk = pick(kappa_all), pick(walk_all)
    # expected value found by a completed rescan, given one is requested
    find = np.where(pa > 0, (pf * full_find + pp * part_find) / np.maximum(pa, 1e-12), 0.0)
    full_cost = cfg.rescan_fixed_seconds + cfg.full_seconds_per_unit * units
    part_cost = cfg.rescan_fixed_seconds + cfg.partial_seconds_per_unit * np.minimum(
        cfg.partial_items, units
    )
    cost = np.where(pa > 0, (pf * full_cost + pp * part_cost) / np.maximum(pa, 1e-12), 0.0)
    r = cc.redirect_seconds

    # Reaching the barcode, or a cancellation that gets a decision: if selected, walk out
    # (hidden lost) or complete the rescan; if not selected, the loss as scanned. A cancelled
    # session that is not selected pays at a staffed till: only hidden items are lost.
    rescanned_loss = walk * hidden + (1 - walk) * (net_loss - find)
    at_exit = (1 - pa) * net_loss + pa * rescanned_loss

    if rules == OPEN:
        loss = kappa * hidden + (1 - kappa) * at_exit
        p_cancel = np.zeros(n)  # a cancellation gets no decision
    else:
        # the genuine cancellations the app logged as crashes get the floor only
        genuine_share = np.where(kappa > 0, m / np.maximum(kappa, 1e-12), 0.0)
        crash = genuine_share * e
        p_cancel = crash * np.minimum(pa, floor) + (1 - crash) * pa
        on_cancel = (1 - p_cancel) * hidden + p_cancel * rescanned_loss
        loss = kappa * on_cancel + (1 - kappa) * at_exit

    asked = (1 - kappa) * pa + kappa * p_cancel
    walked = asked * walk
    rescans_done = asked * (1 - walk)
    seconds = rescans_done * cost + (kappa + walked) * r
    # rescans the policy would have asked for but never got to: skipped by cancelling (open
    # rules) or walked out of
    skipped = (kappa * pa - kappa * p_cancel) + walked
    escaped_hidden = kappa * (pa - p_cancel) * hidden + walked * hidden

    honest = hidden == 0
    typ = g.sessions.gt_type.to_numpy()
    return {
        "loss_remaining": loss.sum(),
        # hidden items kept in sessions that would have been rescanned but escaped it
        "hidden_loss_escaping_rescans": escaped_hidden.sum(),
        "rescans_skipped_or_walked": skipped.sum(),
        "cancellations": kappa.sum(),
        "colleague_hours": seconds.sum() / 3600,
        "honest_stops": (asked * honest).sum(),
        "genuine_cancellers_stopped": (m * p_cancel * honest).sum(),
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
