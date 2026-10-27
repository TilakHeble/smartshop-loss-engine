"""The rescan game: a store commits to a rescan policy; shoppers respond to it.

A policy maps each session's risk score to a probability of a full rescan. Shoppers who decide
whether, and how much, to steal look at the rescan probability they would face for each choice:
stealing changes their scan log, so it changes their score. This is a Stackelberg game: the store
leads (commits to the policy, which shoppers learn over time), shoppers follow (best-respond).

    opportunists   steal on impulse, less often the likelier a rescan (as in the simulator)
    adaptive       choose how much to take, 0%, 25%, 50% or 100% of their usual haul, to
                   maximise haul x (1 - q x d) - q x d x penalty, where q is the rescan
                   probability that choice would bring. Taking less keeps the scan log calmer
                   and the score lower: the way to slip under a threshold.
    everyone else  unaffected by the policy (deliberate shoppers are not deterred)

Every version of reality is simulated once from the same random draws: honest, and each steal
level. Evaluating a policy is then exact and fast: work out each shopper's choice under it, pick
that version, add up.

Simplification, stated: member history (earlier rescan outcomes) is held at what happened under
today's policy, so the game is played one session at a time, not over months.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import pandas as pd

from smartshop.risk.features import GAP_FEATURES, HISTORY_COLUMNS, derive, session_sql
from smartshop.simulate.config import SimConfig
from smartshop.simulate.simulator import (
    build_world,
    load_baskets,
    observed_sessions,
    scan_events,
    session_truth,
    simulate_behaviour,
)

STRATEGIC = ("opportunist", "adaptive")
LEVELS = (0.25, 0.5, 1.0)  # share of the usual haul an adaptive shopper can choose to take
VALUES = ["net_loss", "deliberate_value", "full_find", "partial_find", "true_units", "score"]


@dataclass
class GameData:
    """Per-session values for one period, one column block per version of reality.

    Version 0 is honest; versions 1..len(LEVELS) take LEVELS[k-1] of the usual haul.
    """

    sessions: pd.DataFrame
    cfg: SimConfig
    levels: tuple = field(default=LEVELS)

    def __len__(self):
        return len(self.sessions)

    def matrix(self, value: str) -> np.ndarray:
        """(versions x sessions) array of one value."""
        return np.vstack([self.sessions[f"v{k}_{value}"] for k in range(len(self.levels) + 1)])


def _world_values(s, u, cfg) -> pd.DataFrame:
    """Loss, what each kind of rescan would find, and size, for one version of reality."""
    t = session_truth(s, u)
    detect_partial = np.where(
        u.gt_deliberate, cfg.partial_detect_deliberate, cfg.partial_detect_nonscan
    )
    n = u.groupby("basket_id").basket_id.transform("size")
    pick = np.minimum(1.0, cfg.partial_items / n)
    part = (
        u.assign(f=np.where(u.gt_nonscan, u.unit_price * pick * detect_partial, 0.0))
        .groupby("basket_id")
        .f.sum()
    )
    return pd.DataFrame(
        {
            "net_loss": t.gt_net_loss.to_numpy(),
            "deliberate_value": t.gt_deliberate_value.to_numpy(),
            # expected net recovery of a full rescan: missing units found, double scans refunded
            "full_find": (
                cfg.full_detect_nonscan * t.gt_nonscan_value
                - cfg.full_detect_overscan * t.gt_overscan_value
            ).to_numpy(),
            "partial_find": t.basket_id.map(part).fillna(0.0).to_numpy(),
            "true_units": t.gt_true_units.to_numpy(),
        },
        index=t.basket_id.to_numpy(),
    )


def _scores(con, s, u, cfg, base_features: pd.DataFrame, model, cols) -> pd.Series:
    """Risk score of each session given one version of its scan log."""
    con.register("cf_sessions", observed_sessions(s))
    con.register("cf_events", scan_events(s, u, cfg))
    sess = con.sql(session_sql("cf_sessions", "cf_events")).df()
    con.unregister("cf_sessions")
    con.unregister("cf_events")
    keep = ["basket_id"] + HISTORY_COLUMNS + [c for c in GAP_FEATURES if c in base_features]
    # History is the past (unchanged). The basket-model gap is held at today's value: its weight
    # in the risk model is small, and re-scoring it per version would add minutes for little.
    df = derive(sess.merge(base_features[keep], on="basket_id", how="inner"))
    df[cols] = df[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    return pd.Series(model.predict_proba(df[cols])[:, 1], index=df.basket_id.to_numpy())


def build_game(
    con: duckdb.DuckDBPyConnection,
    base_features: pd.DataFrame,
    model,
    cols,
    days: tuple[int, int],
    cfg: SimConfig | None = None,
    levels: tuple = LEVELS,
) -> GameData:
    """Simulate every version of reality for each SmartShop session in `days`."""
    cfg = cfg or SimConfig()
    world = build_world(*load_baskets(con), cfg)
    strategic = world.sessions.gt_type.isin(STRATEGIC).to_numpy()
    lo, hi = days
    base = world.sessions[["basket_id", "member_id", "day", "gt_type", "u_intent"]]
    out = base[(base.day >= lo) & (base.day <= hi)].set_index("basket_id")

    versions = [(0, 1.0)] + [(1, lv) for lv in levels]
    for k, (force, scale) in enumerate(versions):
        f = np.where(strategic, force, -1)  # non-strategic shoppers decide as usual
        sc = np.where(strategic, scale, 1.0)
        s, u = simulate_behaviour(world, cfg, rescan_prob=0.0, force_intent=f, skip_scale=sc)
        v = _world_values(s, u, cfg)
        v["score"] = _scores(con, s, u, cfg, base_features, model, cols)
        v["intent"] = s.set_index("basket_id").gt_intent
        out = out.join(v.add_prefix(f"v{k}_"), how="inner")
    cols_score = [f"v{k}_score" for k in range(len(versions))]
    out = out.dropna(subset=cols_score).reset_index().rename(columns={"index": "basket_id"})
    return GameData(sessions=out, cfg=cfg, levels=levels)


# --------------------------------------------------------------------------------------------
# Shoppers' best responses and the store's outcomes
# --------------------------------------------------------------------------------------------


def responses(g: GameData, policy, q_today: float | None = None) -> np.ndarray:
    """Each session's chosen version (0 = honest, k = LEVELS[k-1]) under a policy.

    q_today: a flat rescan probability for everyone (today's random rescans) instead of a policy.
    """
    s, cfg = g.sessions, g.cfg
    n_v = len(g.levels) + 1
    if q_today is not None:
        q = np.full((n_v, len(s)), q_today)
    else:
        q = np.vstack([policy(s[f"v{k}_score"].to_numpy()) for k in range(n_v)])
    q = np.clip(q, 0, 1)
    typ = s.gt_type.to_numpy()
    full = n_v - 1  # the "take everything" version

    # Opportunists: impulse, deterred by the rescan chance of the full haul.
    p_opp = cfg.intent_prob["opportunist"] * (1 - q[full]) ** cfg.opportunist_deterrence
    opp = np.where(s.u_intent.to_numpy() < p_opp, full, 0)

    # Adaptive: best of honest and each haul size.
    haul = g.matrix("deliberate_value")
    d, pen = cfg.adaptive_perceived_detection, cfg.adaptive_penalty
    util = haul * (1 - q * d) - q * d * pen
    util[0] = 0.0  # honest: nothing gained, nothing risked
    stealing = np.arange(n_v)[:, None] > 0
    util = np.where(stealing & (haul <= 0), -np.inf, util)  # a haul of nothing is not a choice
    ada = np.argmax(util, axis=0)

    # Everyone else as simulated: deliberate shoppers steal per their own draw (version full).
    fixed = np.where(s[f"v{full}_intent"].to_numpy().astype(bool), full, 0)
    return np.select([typ == "opportunist", typ == "adaptive"], [opp, ada], fixed)


def outcome(g: GameData, policy, today: bool = False) -> dict:
    """Expected outcomes of a policy (a function score -> full-rescan probability)."""
    s, cfg = g.sessions, g.cfg
    q_total = cfg.status_quo_partial_rate + cfg.status_quo_full_rate
    choice = responses(g, policy, q_today=q_total if today else None)
    cols = np.arange(len(s))
    pick = lambda value: g.matrix(value)[choice, cols]  # noqa: E731
    net_loss, deliberate, units = pick("net_loss"), pick("deliberate_value"), pick("true_units")
    full_find, partial_find, score = pick("full_find"), pick("partial_find"), pick("score")
    full_cost = cfg.rescan_fixed_seconds + cfg.full_seconds_per_unit * units
    if today:
        p_full = np.full(len(s), cfg.status_quo_full_rate)
        p_part = np.full(len(s), cfg.status_quo_partial_rate)
        part_cost = cfg.rescan_fixed_seconds + cfg.partial_seconds_per_unit * np.minimum(
            cfg.partial_items, units
        )
        recovered = (p_full * full_find + p_part * partial_find).sum()
        seconds = (p_full * full_cost + p_part * part_cost).sum()
        p_any = p_full + p_part
    else:
        p_full = np.asarray(policy(score))
        recovered = (p_full * full_find).sum()
        seconds = (p_full * full_cost).sum()
        p_any = p_full
    honest = deliberate == 0
    typ = s.gt_type.to_numpy()
    ada = typ == "adaptive"
    full = len(g.levels)
    return {
        "loss_before_rescans": net_loss.sum(),
        "recovered": recovered,
        "loss_remaining": net_loss.sum() - recovered,
        "deliberate_loss": deliberate.sum(),
        "colleague_hours": seconds / 3600,
        "rescans": p_any.sum(),
        "honest_stops": (p_any * honest).sum(),
        "honest_stop_rate": (p_any * honest).sum() / honest.sum(),
        "opportunists_stealing": (choice[typ == "opportunist"] > 0).mean(),
        "adaptive_stealing": (choice[ada] > 0).mean(),
        "adaptive_taking_all": (choice[ada] == full).mean(),
        "adaptive_holding_back": ((choice[ada] > 0) & (choice[ada] < full)).mean(),
        "recovered_per_hour": recovered / max(seconds / 3600, 1e-9),
    }


# --------------------------------------------------------------------------------------------
# Policies (score -> full-rescan probability), each fitted to a colleague-time budget
# --------------------------------------------------------------------------------------------


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def random_policy(rate: float):
    return lambda score: np.full(len(score), rate)


def threshold_policy(t: float, floor: float):
    """Rescan everyone above a score threshold; a random floor for everyone else."""
    return lambda score: np.where(np.asarray(score) >= t, 1.0, floor)


def smooth_policy(t: float, floor: float, cap: float, tau: float):
    """Randomised: probability rises smoothly with the score, from the floor up to a cap.

    Every level of risk carries some chance of a rescan, so there is no safe score just under a
    threshold; a cap below 1 means nobody is rescanned every time.
    """

    def f(score):
        z = np.clip((_logit(np.asarray(score)) - t) / tau, -50, 50)
        return floor + (cap - floor) / (1 + np.exp(-z))

    return f


def today_hours(g: GameData) -> float:
    return outcome(g, None, today=True)["colleague_hours"]


def fit_to_budget(g: GameData, make, hours: float, lo: float, hi: float, iters: int = 40):
    """Bisect the policy's location parameter so expected colleague time meets the budget.

    Shoppers respond to every candidate, so time is evaluated with their responses. `make(x)`
    builds the policy; time must fall as x rises. Returns the policy using at most `hours`.
    """
    for _ in range(iters):
        mid = (lo + hi) / 2
        if outcome(g, make(mid))["colleague_hours"] > hours:
            lo = mid
        else:
            hi = mid
    return make(hi), hi
