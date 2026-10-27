"""Features for (basket, commodity) pairs: what is visible in the basket now, what this member
usually buys (from earlier trips only), and how commodities go together.

Masked-basket training: a complete till basket has a random share of its units hidden, and the
model learns to say which commodities hold hidden units, from the visible rest. It never sees
theft labels, so it learns "what should be in this trolley", not "who steals".
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

FEATURES = [
    "vis_units",
    "is_visible",
    "n_vis_units",
    "n_vis_comm",
    "lift_mean",
    "lift_max",
    "lift_n_pos",
    "prior_trips",
    "m_rate",
    "m_units",
    "units_gap",
    "days_since",
    "popularity",
    "avg_units",
    "multi_share",
    "unit_price",
    "is_high_risk",
    "hour_of_day",
]


def masked_training_pairs(
    con: duckdb.DuckDBPyConnection,
    days: tuple[int, int],
    hide_rate: float = 0.1,
    n_neg: int = 25,
    seed: int = 0,
    max_baskets: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Masked till baskets -> (pairs with label and weight, visible units)."""
    rng = np.random.default_rng(seed)
    lo, hi = days
    b = con.sql(
        f"""
        SELECT b.basket_id, b.commodity, b.units
        FROM obs_basket_comm b JOIN obs_trips t USING (basket_id)
        WHERE NOT t.is_smartshop AND t.day BETWEEN {lo} AND {hi}
          AND b.commodity IN (SELECT commodity FROM comm_stats)
        """
    ).df()
    if max_baskets:
        keep = rng.permutation(b.basket_id.unique())[:max_baskets]
        b = b[b.basket_id.isin(keep)]
    b = b.sort_values(["basket_id", "commodity"]).reset_index(drop=True)
    b["hidden"] = rng.binomial(b.units.astype(int), hide_rate)
    b["vis_units"] = b.units - b.hidden
    # Only baskets with something left visible: an empty scan list has no context.
    has_vis = b.groupby("basket_id").vis_units.transform("sum") > 0
    b = b[has_vis]

    vocab = np.array(sorted(con.sql("SELECT commodity FROM comm_stats").df().commodity))
    pos = b[["basket_id", "commodity"]].assign(label=(b.hidden > 0).astype(int), weight=1.0)
    # Negatives: commodities not in the basket, sampled and up-weighted to stand for all of them.
    in_basket = b.groupby("basket_id").commodity.apply(set)
    neg_rows = []
    for basket_id, present in in_basket.items():
        pool = np.setdiff1d(vocab, list(present), assume_unique=True)
        k = min(n_neg, len(pool))
        pick = rng.choice(pool, size=k, replace=False)
        neg_rows.append(
            pd.DataFrame(
                {"basket_id": basket_id, "commodity": pick, "label": 0, "weight": len(pool) / k}
            )
        )
    pairs = pd.concat([pos, *neg_rows], ignore_index=True)
    visible = b.loc[b.vis_units > 0, ["basket_id", "commodity", "vis_units"]]
    return pairs, visible


def session_pairs(con: duckdb.DuckDBPyConnection, basket_ids) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every vocabulary commodity for each SmartShop session, with what was scanned."""
    ids = pd.DataFrame({"basket_id": np.asarray(basket_ids)})
    con.register("ids", ids)
    pairs = con.sql("SELECT i.basket_id, c.commodity FROM ids i CROSS JOIN comm_stats c").df()
    visible = con.sql(
        """
        SELECT b.basket_id, b.commodity, b.units AS vis_units
        FROM obs_basket_comm b JOIN ids USING (basket_id)
        WHERE b.commodity IN (SELECT commodity FROM comm_stats)
        """
    ).df()
    con.unregister("ids")
    return pairs, visible


def build_features(
    con: duckdb.DuckDBPyConnection, pairs: pd.DataFrame, visible: pd.DataFrame
) -> pd.DataFrame:
    con.register("pairs_df", pairs)
    con.register("visible_df", visible)
    df = con.sql(
        """
        WITH ctx AS (
            SELECT basket_id, SUM(vis_units) AS n_vis_units, COUNT(*) AS n_vis_comm
            FROM visible_df GROUP BY 1
        ),
        lf AS (
            SELECT p.basket_id, p.commodity,
                   AVG(COALESCE(l.log_lift, -1.0)) AS lift_mean,
                   MAX(COALESCE(l.log_lift, -1.0)) AS lift_max,
                   SUM((COALESCE(l.log_lift, -1.0) > 0.5)::INT) AS lift_n_pos
            FROM pairs_df p
            JOIN visible_df v ON v.basket_id = p.basket_id AND v.commodity <> p.commodity
            LEFT JOIN lift l ON l.c = p.commodity AND l.j = v.commodity
            GROUP BY 1, 2
        ),
        base AS (
            SELECT p.*, t.member_id, t.trip_order, t.day, t.hour_of_day,
                   COALESCE(v.vis_units, 0) AS vis_units,
                   COALESCE(ctx.n_vis_units, 0) AS n_vis_units,
                   COALESCE(ctx.n_vis_comm, 0) AS n_vis_comm,
                   lf.lift_mean, lf.lift_max, COALESCE(lf.lift_n_pos, 0) AS lift_n_pos,
                   cs.popularity, cs.avg_units, cs.multi_share, cs.unit_price,
                   cs.is_high_risk::INT AS is_high_risk
            FROM pairs_df p
            JOIN obs_trips t USING (basket_id)
            JOIN comm_stats cs USING (commodity)
            LEFT JOIN visible_df v USING (basket_id, commodity)
            LEFT JOIN ctx USING (basket_id)
            LEFT JOIN lf USING (basket_id, commodity)
        )
        SELECT base.*, h.cum_trips, h.cum_units, base.day - h.day AS days_since
        FROM base
        ASOF LEFT JOIN mc_hist h
          ON h.member_id = base.member_id AND h.commodity = base.commodity
         AND base.trip_order > h.trip_order
        """
    ).df()
    con.unregister("pairs_df")
    con.unregister("visible_df")

    df["prior_trips"] = df.trip_order - 1
    df["cum_trips"] = df.cum_trips.fillna(0)
    df["m_rate"] = np.where(df.prior_trips > 0, df.cum_trips / df.prior_trips.clip(lower=1), np.nan)
    df["m_units"] = np.where(df.cum_trips > 0, df.cum_units / df.cum_trips.clip(lower=1), np.nan)
    df["units_gap"] = df.m_units.fillna(df.avg_units) - df.vis_units
    df["is_visible"] = (df.vis_units > 0).astype(int)
    df[FEATURES] = df[FEATURES].astype("float32")
    return df.sort_values(["basket_id", "commodity"]).reset_index(drop=True)
