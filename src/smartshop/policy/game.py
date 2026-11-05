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
from smartshop.simulate.rescan import partial_count
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
VALUES = [
    "net_loss",
    "deliberate_value",
    "full_find",
    "partial_find",
    "partial_hit",
    "partial_units",
    "true_units",
    "nonscan_units",
    "score",
]


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

    def subset(self, mask) -> GameData:
        """The same game restricted to some sessions (e.g. one rescan cycle)."""
        return GameData(
            self.sessions[np.asarray(mask)].reset_index(drop=True), self.cfg, self.levels
        )

    def matrix(self, value: str) -> np.ndarray:
        """(versions x sessions) array of one value."""
        return np.vstack([self.sessions[f"v{k}_{value}"] for k in range(len(self.levels) + 1)])


def _partial_values(t, u, cfg) -> dict:
    """What a partial check finds, how often it escalates and how many units it covers."""
    detect_partial = np.where(
        u.gt_deliberate & ~u.gt_switched, cfg.partial_detect_deliberate, cfg.partial_detect_nonscan
    )
    n = u.groupby("basket_id").basket_id.transform("size")
    k = partial_count(n, cfg)
    pick = k / n
    h = np.where(u.gt_nonscan, pick * detect_partial, 0.0)  # chance this unit is found
    by = u.basket_id.to_numpy()
    # chance a partial check finds anything (then the till suggests a full check)
    hit = 1 - np.exp(pd.Series(np.log1p(-np.minimum(h, 1 - 1e-12))).groupby(by).sum())
    part_units = pd.Series(np.asarray(k)).groupby(by).first()
    no_escalation = u.assign(f=u.loss_price * h).groupby("basket_id").f.sum()
    full_find = (
        cfg.full_detect_nonscan * t.gt_nonscan_value
        - cfg.full_detect_overscan * t.gt_overscan_value
    ).to_numpy()
    return {
        # expected recovery of a partial check: if it finds anything it escalates to a full
        # check, which then finds what a full rescan would
        "partial_find": (
            t.basket_id.map(hit).to_numpy() * full_find
            if cfg.escalate_partial
            else t.basket_id.map(no_escalation).fillna(0.0).to_numpy()
        ),
        "partial_hit": t.basket_id.map(hit).fillna(0.0).to_numpy(),
        "partial_units": t.basket_id.map(part_units).to_numpy(),
    }


def _world_values(s, u, cfg, partial_rules: dict | None = None) -> pd.DataFrame:
    """Loss, what each kind of rescan would find, and size, for one version of reality.

    partial_rules: optional {name: cfg} of other partial-check sizes, stored as
    partial_find@name, partial_hit@name, partial_units@name (see use_partial_rule).
    """
    t = session_truth(s, u)
    extra = {}
    for name, c in (partial_rules or {}).items():
        extra.update({f"{k}@{name}": v for k, v in _partial_values(t, u, c).items()})
    return pd.DataFrame(
        {
            "net_loss": t.gt_net_loss.to_numpy(),
            "deliberate_value": t.gt_deliberate_value.to_numpy(),
            # expected net recovery of a full rescan: missing units found, double scans refunded
            "full_find": (
                cfg.full_detect_nonscan * t.gt_nonscan_value
                - cfg.full_detect_overscan * t.gt_overscan_value
            ).to_numpy(),
            **_partial_values(t, u, cfg),
            **extra,
            "true_units": t.gt_true_units.to_numpy(),
            "nonscan_units": t.gt_nonscan_units.to_numpy(),
        },
        index=t.basket_id.to_numpy(),
    )


def _scores(con, s, u, cfg, base_features: pd.DataFrame, model, cols) -> pd.DataFrame:
    """Risk score (and units scanned) of each session given one version of its scan log."""
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
    return pd.DataFrame(
        {
            "score": model.predict_proba(df[cols])[:, 1],
            "scanned_units": df.scanned_units.to_numpy(),
        },
        index=df.basket_id.to_numpy(),
    )


def build_game(
    con: duckdb.DuckDBPyConnection,
    base_features: pd.DataFrame,
    model,
    cols,
    days: tuple[int, int],
    cfg: SimConfig | None = None,
    levels: tuple = LEVELS,
    partial_rules: dict | None = None,
) -> GameData:
    """Simulate every version of reality for each SmartShop session in `days`.

    partial_rules: other partial-check sizes to price in the same pass (use_partial_rule).
    """
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
        v = _world_values(s, u, cfg, partial_rules)
        v = v.join(_scores(con, s, u, cfg, base_features, model, cols))
        v["intent"] = s.set_index("basket_id").gt_intent
        out = out.join(v.add_prefix(f"v{k}_"), how="inner")
    cols_score = [f"v{k}_score" for k in range(len(versions))]
    out = out.dropna(subset=cols_score).reset_index().rename(columns={"index": "basket_id"})
    return GameData(sessions=out, cfg=cfg, levels=levels)


def use_partial_rule(g: GameData, name: str, cfg: SimConfig) -> GameData:
    """The same game with partial checks sized by another rule priced in build_game."""
    s = g.sessions.copy()
    for k in range(len(g.levels) + 1):
        for v in ("partial_find", "partial_hit", "partial_units"):
            s[f"v{k}_{v}"] = s[f"v{k}_{v}@{name}"]
    return GameData(s, cfg, g.levels)


def split_policy(policy, score) -> tuple[np.ndarray, np.ndarray]:
    """(full, partial) check probabilities from a policy that returns either."""
    out = policy(score)
    if isinstance(out, tuple):
        return np.asarray(out[0], float), np.asarray(out[1], float)
    p = np.asarray(out, float)
    return p, np.zeros_like(p)


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
        w = getattr(cfg, "partial_deterrence", 1.0)
        rows = []
        for k in range(n_v):
            full_p, part_p = split_policy(policy, s[f"v{k}_score"].to_numpy())
            if w == "share":
                cover = s[f"v{k}_partial_units"] / s[f"v{k}_true_units"].clip(lower=1)
                wk = np.clip(cover.to_numpy(), 0, 1)
            else:
                wk = float(w)
            rows.append(full_p + wk * part_p)  # how much the checks deter, as thieves see it
        q = np.vstack(rows)
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
    """Expected outcomes of a policy.

    policy: score -> full-check probability, or score -> (full, partial) probabilities for a
    policy that also uses partial checks (which escalate to a full check when they find
    anything). today=True: today's random partial and full checks.
    """
    s, cfg = g.sessions, g.cfg
    if today:
        rates = (cfg.status_quo_full_rate, cfg.status_quo_partial_rate)
        policy = lambda score: tuple(np.full(len(score), r) for r in rates)  # noqa: E731
    choice = responses(g, policy)
    cols = np.arange(len(s))
    pick = lambda value: g.matrix(value)[choice, cols]  # noqa: E731
    net_loss, deliberate, units = pick("net_loss"), pick("deliberate_value"), pick("true_units")
    full_find, partial_find, score = pick("full_find"), pick("partial_find"), pick("score")
    full_cost = cfg.rescan_fixed_seconds + cfg.full_seconds_per_unit * units
    # a partial check, plus the full check it escalates to when it finds anything
    part_cost = (
        cfg.rescan_fixed_seconds
        + cfg.partial_seconds_per_unit * pick("partial_units")
        + cfg.escalate_partial * pick("partial_hit") * cfg.full_seconds_per_unit * units
    )
    if today:
        p_full = np.full(len(s), cfg.status_quo_full_rate)
        p_part = np.full(len(s), cfg.status_quo_partial_rate)
    else:
        p_full, p_part = split_policy(policy, score)
    recovered = (p_full * full_find + p_part * partial_find).sum()
    seconds = (p_full * full_cost + p_part * part_cost).sum()
    p_any = p_full + p_part
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


def mix_policy(t: float, floor: float, cap: float, tau: float, s_full: float):
    """v1.4: the same overall check probability as smooth_policy, split by risk.

    The random floor is always a full check (unbiased measurement and training labels). Above
    the floor, a session scoring at least `s_full` gets a full check; one below it gets a partial
    check of about 20-25% of the basket, which escalates to a full check if it finds anything.
    """
    base = smooth_policy(t, floor, cap, tau)

    def policy(score):
        score = np.asarray(score)
        p = np.asarray(base(score))
        extra = np.maximum(p - floor, 0.0)
        high = score >= s_full
        return np.minimum(p, floor) + np.where(high, extra, 0.0), np.where(high, 0.0, extra)

    return policy


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
