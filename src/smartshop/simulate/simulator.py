"""SmartShop session simulator on top of real baskets.

Real (from the dbt marts): who shopped, where, when, and exactly what was in the basket, at
normal and Nectar price. Simulated (documented in SimConfig): who used SmartShop, how they
scanned, what they missed and why, and what today's random rescans found.

Two kinds of output, kept in separate schemas so models can never see the truth by accident:
    sim_obs    what SmartShop would actually record: sessions, scan events, rescan results
    sim_truth  hidden ground truth (gt_ columns): shopper type, every missed or double-scanned
               unit and why. Used only to evaluate.

Randomness is drawn once per unit and per session ("common random numbers"). Re-running the
behaviour under a different rescan policy (week 6) reuses the same draws, so policies are
compared on identical shoppers and baskets.

Run:
    python -m smartshop.simulate.simulator
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import duckdb
import numpy as np
import pandas as pd

from smartshop.config import WAREHOUSE_PATH
from smartshop.simulate.config import SimConfig
from smartshop.simulate.rescan import FULL, NONE, PARTIAL, rescan_outcomes

log = logging.getLogger(__name__)

TYPES = ["honest", "careless", "opportunist", "deliberate", "adaptive"]


@dataclass
class World:
    """Everything fixed before any rescan policy is chosen."""

    members: pd.DataFrame  # member_id, gt_type, channel, adoption_day, gt_error_mult
    sessions: pd.DataFrame  # one row per SmartShop session (+ per-session random draws)
    units: pd.DataFrame  # one row per physical unit in the basket (+ per-unit random draws)


# --------------------------------------------------------------------------------------------
# 1. The world: members, SmartShop sessions and the units in each basket
# --------------------------------------------------------------------------------------------


def load_baskets(con: duckdb.DuckDBPyConnection) -> tuple[pd.DataFrame, pd.DataFrame]:
    trips = con.sql(
        """
        SELECT basket_id, member_id, store_id, day, week_no, checkout_minute, hour_of_day,
               n_units, n_lines, nectar_value, trip_seq
        FROM marts.fct_trip
        """
    ).df()
    lines = con.sql(
        """
        SELECT basket_id, product_id, department, is_high_risk, quantity, nectar_unit_price
        FROM marts.fct_scan_line
        """
    ).df()
    return trips, lines


def build_world(trips: pd.DataFrame, lines: pd.DataFrame, cfg: SimConfig) -> World:
    rng = np.random.default_rng(cfg.seed)

    # Members: hidden type, channel, when they started using SmartShop.
    members = pd.DataFrame({"member_id": np.sort(trips.member_id.unique())})
    n = len(members)
    probs = np.array([cfg.type_mix[t] for t in TYPES])
    members["gt_type"] = rng.choice(TYPES, size=n, p=probs / probs.sum())
    members["channel"] = np.where(rng.random(n) < cfg.app_share, "app", "handset")
    lo, hi = cfg.adoption_day_range
    members["adoption_day"] = rng.integers(lo, hi + 1, n)
    members["gt_error_mult"] = np.exp(rng.normal(0, cfg.member_error_sd, n))

    # Sessions: trips after adoption, used with a probability that grows with basket size.
    t = trips.merge(members, on="member_id")
    logit = cfg.usage_logit_intercept + cfg.usage_logit_per_unit * t.n_units
    use = (t.day >= t.adoption_day) & (rng.random(len(t)) < 1 / (1 + np.exp(-logit)))
    s = t.loc[use].reset_index(drop=True)
    # session-level draws
    s["u_intent"] = rng.random(len(s))
    s["u_policy"] = rng.random(len(s))
    s["duration_min"] = np.exp(
        np.log(cfg.minutes_base + cfg.minutes_per_unit * s.n_units)
        + rng.normal(0, cfg.minutes_sd_log, len(s))
    )

    # Units: expand each line by quantity (SmartShop scans every unit individually [SBY]).
    ln = lines[lines.basket_id.isin(s.basket_id)].reset_index(drop=True)
    rep = ln.quantity.to_numpy().astype(int)
    u = ln.loc[ln.index.repeat(rep)].reset_index(drop=True)
    u["unit_idx"] = u.groupby(["basket_id", "product_id"]).cumcount()
    u = u.rename(columns={"nectar_unit_price": "unit_price"}).drop(columns="quantity")
    m = len(u)
    for name in ["u_skip", "u_acc", "u_over", "u_delete", "u_order", "u_time", "u_pick", "u_det"]:
        u[name] = rng.random(m)
    u["n_honest_deletes"] = rng.poisson(1.0, m)  # scaled later by the delete rate

    # Member habits unrelated to honesty, from a separate stream so every draw above (types,
    # misses, rescans) is unchanged by adding them.
    rng_h = np.random.default_rng(cfg.seed + 1000)
    for col, sd in [("gt_delete_mult", cfg.member_delete_sd), ("gt_pace_mult", cfg.member_pace_sd)]:
        members[col] = np.exp(rng_h.normal(-(sd**2) / 2, sd, n))
    s["duration_min"] = s.duration_min * s.member_id.map(
        members.set_index("member_id").gt_pace_mult
    )
    return World(members=members, sessions=s, units=u)


# --------------------------------------------------------------------------------------------
# 2. Behaviour: what each shopper scans, given the rescan probability they face
# --------------------------------------------------------------------------------------------


def simulate_behaviour(world: World, cfg: SimConfig, rescan_prob: np.ndarray | float) -> tuple:
    """Return (sessions, units) with hidden truth, for a given per-session rescan probability.

    rescan_prob is the chance each session is rescanned under the policy in force (aligned with
    world.sessions). Opportunists are deterred by it; adaptive shoppers best-respond to it.
    """
    s = world.sessions.copy()
    s["rescan_prob"] = np.broadcast_to(np.asarray(rescan_prob, dtype=float), (len(s),))
    u = world.units.merge(
        s[["basket_id", "member_id", "n_units"]], on="basket_id", how="left", validate="many_to_one"
    ).merge(
        world.members[["member_id", "gt_type", "gt_error_mult", "gt_delete_mult"]], on="member_id"
    )
    stype = s.gt_type  # sessions carry the member's hidden type from build_world

    # Per-unit probability of a deliberate skip, if the session has intent.
    base = u.gt_type.map(cfg.skip_base).fillna(0.0).to_numpy()
    price_f = np.minimum(cfg.price_cap, np.sqrt(u.unit_price.to_numpy() / cfg.price_ref))
    risk_f = np.where(u.is_high_risk.to_numpy(), cfg.high_risk_skip_factor, 1.0)
    p_skip = np.minimum(cfg.skip_cap, base * price_f * risk_f)
    would_skip = u.u_skip.to_numpy() < p_skip

    # Session intent.
    q = s.rescan_prob.to_numpy()
    p_intent = stype.map(cfg.intent_prob).fillna(0.0).to_numpy()
    p_intent = np.where(
        stype == "opportunist", p_intent * (1 - q) ** cfg.opportunist_deterrence, p_intent
    )
    intent = s.u_intent.to_numpy() < p_intent
    # Adaptive shoppers: steal only if the planned haul is worth the expected cost.
    planned = (
        pd.Series(np.where(would_skip, u.unit_price, 0.0)).groupby(u.basket_id.to_numpy()).sum()
    )
    v = s.basket_id.map(planned).fillna(0.0).to_numpy()
    d = cfg.adaptive_perceived_detection
    worth_it = v * (1 - q * d) > q * d * cfg.adaptive_penalty
    intent = np.where(stype == "adaptive", worth_it & (v > 0), intent)
    s["gt_intent"] = intent

    u = u.merge(s[["basket_id", "gt_intent"]], on="basket_id")
    deliberate = u.gt_intent.to_numpy() & would_skip

    # Genuine mistakes, every type.
    acc_rate = u.gt_type.map(cfg.accidental_nonscan).fillna(cfg.accidental_nonscan["honest"])
    over_rate = u.gt_type.map(cfg.accidental_overscan).fillna(cfg.accidental_overscan["honest"])
    extra = np.where(u.unit_idx.to_numpy() > 0, cfg.extra_unit_multiplier, 1.0)
    size = (u.n_units.to_numpy() / 20.0) ** cfg.size_elasticity
    mult = u.gt_error_mult.to_numpy() * extra * size
    accidental = (~deliberate) & (u.u_acc.to_numpy() < acc_rate.to_numpy() * mult)
    nonscan = deliberate | accidental
    overscan = (~nonscan) & (u.u_over.to_numpy() < over_rate.to_numpy() * mult)

    u["gt_deliberate"] = deliberate
    u["gt_accidental"] = accidental
    u["gt_nonscan"] = nonscan
    u["gt_overscan"] = overscan
    return s, u


def session_truth(s: pd.DataFrame, u: pd.DataFrame) -> pd.DataFrame:
    price = u.unit_price
    agg = (
        u.assign(
            nonscan_value=np.where(u.gt_nonscan, price, 0.0),
            deliberate_value=np.where(u.gt_deliberate, price, 0.0),
            accidental_value=np.where(u.gt_accidental, price, 0.0),
            overscan_value=np.where(u.gt_overscan, price, 0.0),
        )
        .groupby("basket_id")
        .agg(
            gt_true_units=("unit_price", "size"),
            gt_true_value=("unit_price", "sum"),
            gt_nonscan_units=("gt_nonscan", "sum"),
            gt_deliberate_units=("gt_deliberate", "sum"),
            gt_accidental_units=("gt_accidental", "sum"),
            gt_overscan_units=("gt_overscan", "sum"),
            gt_nonscan_value=("nonscan_value", "sum"),
            gt_deliberate_value=("deliberate_value", "sum"),
            gt_accidental_value=("accidental_value", "sum"),
            gt_overscan_value=("overscan_value", "sum"),
        )
    )
    out = s[["basket_id", "member_id", "gt_intent", "rescan_prob"]].merge(
        agg, left_on="basket_id", right_index=True
    )
    out["gt_net_loss"] = out.gt_nonscan_value - out.gt_overscan_value
    out["gt_has_error"] = (out.gt_nonscan_units + out.gt_overscan_units) > 0
    return out


# --------------------------------------------------------------------------------------------
# 3. What SmartShop records: scan events with times
# --------------------------------------------------------------------------------------------


def scan_events(s: pd.DataFrame, u: pd.DataFrame, cfg: SimConfig) -> pd.DataFrame:
    """Observable event log: one row per scan or delete, timed within the session.

    Shopping time depends on the TRUE basket, so a session with unscanned units looks slow for
    what was scanned: a real signal, not a planted one.
    """
    u = u.merge(s[["basket_id", "duration_min"]], on="basket_id")
    late = u.department.isin(cfg.late_departments).to_numpy()
    u["order_key"] = u.u_order + 0.35 * late
    u["rank"] = u.groupby("basket_id").order_key.rank(method="first")
    n = u.groupby("basket_id").basket_id.transform("size")
    frac = 0.03 + 0.94 * (u["rank"] - u.u_time) / n
    u["t"] = (frac * u.duration_min * 60).round(1)

    base_cols = ["basket_id", "product_id", "unit_price", "is_high_risk"]
    scanned = u[~u.gt_nonscan]
    ev = [scanned[base_cols + ["t"]].assign(action="scan")]
    # double scans (over-scans): a second scan 2-4 seconds later
    over = u[u.gt_overscan]
    ev.append(over[base_cols].assign(t=over.t + 2 + 2 * over.u_time, action="scan"))
    # deliberate skips done as scan-then-delete
    sd = u[u.gt_deliberate & (u.u_delete < cfg.scan_then_delete_share)]
    ev.append(sd[base_cols + ["t"]].assign(action="scan"))
    ev.append(sd[base_cols].assign(t=sd.t + 3 + 5 * sd.u_time, action="delete"))
    # ordinary deletes on scanned units (mistake or changed mind): scan, delete, scan again
    rate = np.where(scanned.gt_intent, cfg.intent_delete_rate, cfg.honest_delete_rate)
    nd = scanned[scanned.u_delete < rate * scanned.gt_delete_mult]
    ev.append(nd[base_cols].assign(t=nd.t - 6, action="scan"))
    ev.append(nd[base_cols].assign(t=nd.t - 4, action="delete"))

    events = pd.concat(ev, ignore_index=True)
    events["t"] = events.t.clip(lower=0)
    events = events.sort_values(["basket_id", "t", "action"], kind="stable")
    events["event_seq"] = events.groupby("basket_id").cumcount() + 1
    return events.rename(columns={"t": "seconds_from_start"})[
        [
            "basket_id",
            "event_seq",
            "seconds_from_start",
            "action",
            "product_id",
            "unit_price",
            "is_high_risk",
        ]
    ].reset_index(drop=True)


def observed_sessions(s: pd.DataFrame) -> pd.DataFrame:
    o = s.copy()  # sessions carry the member's channel from build_world
    o["start_minute"] = (o.checkout_minute - o.duration_min).round(1)
    return o[
        [
            "basket_id",
            "member_id",
            "store_id",
            "day",
            "week_no",
            "hour_of_day",
            "channel",
            "start_minute",
            "checkout_minute",
            "duration_min",
        ]
    ]


# --------------------------------------------------------------------------------------------
# 4. Today's policy: random partial and full rescans
# --------------------------------------------------------------------------------------------


def status_quo_decisions(s: pd.DataFrame, cfg: SimConfig) -> np.ndarray:
    p, f = cfg.status_quo_partial_rate, cfg.status_quo_full_rate
    return np.select([s.u_policy < p, s.u_policy < p + f], [PARTIAL, FULL], NONE)


def simulate(con: duckdb.DuckDBPyConnection, cfg: SimConfig | None = None) -> dict:
    cfg = cfg or SimConfig()
    t0 = time.time()
    trips, lines = load_baskets(con)
    world = build_world(trips, lines, cfg)
    q = cfg.status_quo_partial_rate + cfg.status_quo_full_rate
    s, u = simulate_behaviour(world, cfg, rescan_prob=q)
    decisions = status_quo_decisions(s, cfg)
    rescans = rescan_outcomes(s.basket_id.to_numpy(), decisions, u, cfg)
    out = {
        "sim_obs.sessions": observed_sessions(s),
        "sim_obs.scan_events": scan_events(s, u, cfg),
        "sim_obs.rescans": rescans,
        "sim_truth.members": world.members,
        "sim_truth.sessions": session_truth(s, u),
        "sim_truth.units": u[
            [
                "basket_id",
                "product_id",
                "unit_idx",
                "unit_price",
                "is_high_risk",
                "gt_deliberate",
                "gt_accidental",
                "gt_nonscan",
                "gt_overscan",
            ]
        ],
    }
    log.info("simulated %s sessions, %s units in %.1fs", len(s), len(u), time.time() - t0)
    return out


def write(tables: dict, path=WAREHOUSE_PATH) -> None:
    con = duckdb.connect(str(path))
    for schema in ("sim_obs", "sim_truth"):
        con.execute(f"DROP SCHEMA IF EXISTS {schema} CASCADE")
        con.execute(f"CREATE SCHEMA {schema}")
    for name, df in tables.items():
        con.register("df", df)
        con.execute(f"CREATE TABLE {name} AS SELECT * FROM df")
        con.unregister("df")
        log.info("%-24s %10s rows", name, f"{len(df):,}")
    con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    con = duckdb.connect(args.warehouse, read_only=True)
    tables = simulate(con)
    con.close()
    write(tables, args.warehouse)


if __name__ == "__main__":
    main()
