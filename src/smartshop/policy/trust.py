"""Nectar-linked trust: honest regulars earn fewer rescans, over repeated rescan cycles.

The rule is deliberately simple enough to explain to a shopper or a colleague:

    every clean full rescan in the last `window_days` (since the last reset) lowers your rescan
    chance by `step`, down to at most `max_discount` off;
    a rescan that finds more than a small slip resets it; a small slip (at most `slip_units`
    item worth under `slip_value`, corrected on the spot) counts as clean, because honest
    shoppers with big baskets make them often;
    the random floor always applies, so nobody is ever beyond a check.

Trust is per Nectar member, so it follows the shopper across stores (week 1: most members use
several). Because it changes how likely each member is to be rescanned, and rescans are what
build trust, it is evaluated over several cycles: each cycle's rescans (drawn at the policy's
probabilities) update trust for the next, and shoppers respond to the probabilities they face.

Which rescans count matters. A partial check rarely sees items skipped on purpose (they are
concealed), so trust built from clean *partial* checks can be earned by a thief. The default
counts full rescans only; `count_partial=True` is the variant that shows why.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smartshop.policy.game import (
    GameData,
    fit_to_budget,
    outcome,
    responses,
    smooth_policy,
    today_hours,
)


@dataclass(frozen=True)
class TrustRule:
    step: float = 0.25
    max_discount: float = 0.5
    window_days: int = 365
    count_partial: bool = False
    enabled: bool = True
    slip_units: int = 1  # a find this small is an honest slip, not a reset (0 = any find resets)
    slip_value: float = 10.0


# Chosen in week 7 (notebooks/06_trust_and_targeting.ipynb): full rescans only, slips allowed.
RECOMMENDED_TRUST = TrustRule(step=0.25, max_discount=0.75)


def history_events(con, before_day: int) -> pd.DataFrame:
    """Rescans each member had before `before_day` (observable): day, kind, found or clean."""
    return con.sql(
        f"""
        SELECT s.member_id, s.day, r.rescan_type AS kind, r.found_missing_units AS found_units,
               r.found_missing_value AS found_value
        FROM sim_obs.rescans r JOIN sim_obs.sessions s USING (basket_id)
        WHERE s.day < {before_day}
        """
    ).df()


def multipliers(events: pd.DataFrame, members: np.ndarray, day: int, rule: TrustRule):
    """Each member's rescan multiplier at `day` (1 = no trust, 1 - max_discount = full trust)."""
    out = pd.Series(1.0, index=pd.Index(np.unique(members), name="member_id"))
    if not rule.enabled or events.empty:
        return out
    e = events[(events.day < day) & (events.day >= day - rule.window_days)]
    slip = (e.found_units <= rule.slip_units) & (e.found_value < rule.slip_value)
    reset = (e.found_units > 0) & ~slip
    last_find = e[reset].groupby("member_id").day.max()
    counts = e[~reset & (rule.count_partial | (e.kind == "full"))]
    counts = counts.assign(last_find=counts.member_id.map(last_find).fillna(-1))
    clean = counts[counts.day > counts.last_find].groupby("member_id").size()
    discount = np.minimum(rule.max_discount, rule.step * clean)
    out.loc[discount.index.intersection(out.index)] = 1 - discount
    return out


def with_trust(base, mult: np.ndarray, floor: float):
    """Policy that scales the base probability by each session's trust multiplier."""
    return lambda score: np.maximum(floor, np.asarray(base(score)) * mult)


def run_cycles(
    g: GameData,
    events: pd.DataFrame,
    params: dict,
    rule: TrustRule,
    n_cycles: int = 4,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Play the game cycle by cycle; realised rescans update trust between cycles.

    Returns (per-cycle outcomes, every realised rescan). The same random numbers decide which
    sessions are rescanned and what a rescan finds in every arm, so arms differ only by policy.
    """
    s = g.sessions
    rng = np.random.default_rng(seed)
    u_rescan = rng.random(len(s))
    u_detect = rng.random(len(s))
    edges = np.linspace(s.day.min(), s.day.max() + 1, n_cycles + 1).astype(int)
    events = events.copy()
    rows, realised = [], []
    cols_all = np.arange(len(s))
    for c in range(n_cycles):
        lo, hi = edges[c], edges[c + 1]
        mask = ((s.day >= lo) & (s.day < hi)).to_numpy()
        gc = g.subset(mask)
        mult_m = multipliers(events, gc.sessions.member_id.to_numpy(), lo, rule)
        mult = gc.sessions.member_id.map(mult_m).to_numpy()

        def make(t, mult=mult):
            base = smooth_policy(t, params["floor"], params["cap"], params["tau"])
            return with_trust(base, mult, params["floor"])

        policy, _ = fit_to_budget(gc, make, today_hours(gc), -12, 12, iters=30)
        o = outcome(gc, policy)
        choice = responses(gc, policy)
        cols = np.arange(len(gc))
        p = policy(gc.matrix("score")[choice, cols])
        honest = gc.matrix("deliberate_value")[choice, cols] == 0
        missed_units = gc.matrix("nonscan_units")[choice, cols]
        missed_value = (
            np.maximum(gc.matrix("full_find")[choice, cols], 0) / g.cfg.full_detect_nonscan
        )
        idx = cols_all[mask]
        rescanned = u_rescan[idx] < p
        found = rescanned & (missed_units > 0) & (u_detect[idx] < g.cfg.full_detect_nonscan)
        r = pd.DataFrame(
            {
                "basket_id": gc.sessions.basket_id,
                "member_id": gc.sessions.member_id,
                "day": gc.sessions.day,
                "gt_type": gc.sessions.gt_type,
                "p": p,
                "trust_mult": mult,
                "honest": honest,
                "rescanned": rescanned,
                "found": found,
                "found_units": np.where(found, missed_units, 0),
                "found_value": np.where(found, missed_value, 0.0),
                "cycle": c + 1,
            }
        )
        realised.append(r)
        new = r[r.rescanned][["member_id", "day", "found_units", "found_value"]].assign(kind="full")
        events = pd.concat([events, new], ignore_index=True)
        typ = gc.sessions.gt_type.to_numpy()
        rows.append(
            {
                "cycle": c + 1,
                **o,
                "members_trusted": int((mult_m < 1).sum()),
                "trust_discount_to_thieves": _discount_share(mult, typ),
            }
        )
    return pd.DataFrame(rows), pd.concat(realised, ignore_index=True)


def _discount_share(mult: np.ndarray, typ: np.ndarray) -> float:
    """Share of all trust discount (1 - multiplier) that went to opportunist/deliberate/adaptive."""
    d = 1 - mult
    total = d.sum()
    if total == 0:
        return float("nan")
    return float(d[np.isin(typ, ["opportunist", "deliberate", "adaptive"])].sum() / total)
