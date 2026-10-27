"""Risk features for every SmartShop session: only what the store can observe, as of the moment
the shopper reaches the exit.

    scan behaviour   from the session's own scan log (times, deletions, what was scanned)
    member history   from earlier trips only: Nectar purchase record (full at tills, scanned in
                     SmartShop), earlier SmartShop sessions, and earlier rescan outcomes
    basket gap       the basket model's expected unscanned value (week 4), optional

Never read: demographics, or anything in the hidden-truth schema. Channel, time of day and tenure
are built for fairness reporting only (NOT_FOR_SCORING).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

SESSION_FEATURES = [
    "scanned_units",
    "scanned_value",
    "n_products",
    "n_deletes",
    "delete_rate",
    "duration_min",
    "min_per_unit",
    "mean_unit_price",
    "hr_units",
    "hr_value_share",
    "max_gap_min",
    "gap_cv",
    "late_share",
    "first_scan_frac",
    "repeat_scans",
    "hour_of_day",
    "is_peak",
    "is_app",
]
HISTORY_FEATURES = [
    "prior_trips",
    "prior_ss_sessions",
    "days_since_first_ss",
    "m_avg_units",
    "units_vs_usual",
    "m_hr_share",
    "hr_share_gap",
    "m_min_per_unit",
    "min_per_unit_vs_usual",
    "m_delete_rate",
    "prior_rescans",
    "prior_found",
    "prior_found_rate",
    "prior_full_rescans",
    "prior_full_found",
]
GAP_FEATURES = ["gap_value", "gap_per_unit"]

# Built for reporting and fairness checks, but never used to score: channel and time of day have
# no causal role (a model learns spurious effects from few labels, and flagging peak-hour shoppers
# would also clash with busy exits), and tenure counts (trips, sessions, rescans so far) measure
# how long someone has been a customer, not how they behave, so they would flag new users.
NOT_FOR_SCORING = [
    "is_app",
    "hour_of_day",
    "is_peak",
    "prior_trips",
    "prior_ss_sessions",
    "days_since_first_ss",
    "prior_rescans",
    "prior_full_rescans",
]
SESSION_FEATURES = [f for f in SESSION_FEATURES if f not in NOT_FOR_SCORING]
SCORING_HISTORY = [f for f in HISTORY_FEATURES if f not in NOT_FOR_SCORING]


def session_sql(sessions: str, events: str, peak_hours=(16, 17, 18)) -> str:
    """SELECT for one row of scan-log features per session, from any sessions/events tables.

    Used for the real log (sim_obs) and for counterfactual logs ("what if this shopper had / had
    not stolen") when the rescan policy is evaluated against shoppers who respond to it.
    """
    peaks = ", ".join(str(h) for h in peak_hours)
    return f"""
        WITH ev AS (
            SELECT e.basket_id, e.seconds_from_start AS t, e.action, e.product_id, e.unit_price,
                   e.is_high_risk,
                   (e.action = 'scan' AND e.seconds_from_start > 0.8 * s.duration_min * 60)::INT
                       AS is_late_scan,
                   e.seconds_from_start - LAG(e.seconds_from_start) OVER (
                       PARTITION BY e.basket_id ORDER BY e.event_seq) AS gap
            FROM {events} e JOIN {sessions} s USING (basket_id)
        ),
        agg AS (
            SELECT basket_id,
                   SUM(CASE WHEN action = 'scan' THEN 1 ELSE -1 END)             AS scanned_units,
                   SUM(CASE WHEN action = 'scan' THEN unit_price ELSE -unit_price END)
                                                                                 AS scanned_value,
                   SUM(CASE WHEN action = 'scan' THEN 1 ELSE -1 END
                       * is_high_risk::INT)                                      AS hr_units,
                   SUM(CASE WHEN action = 'scan' THEN unit_price ELSE -unit_price END
                       * is_high_risk::INT)                                      AS hr_value,
                   COUNT(DISTINCT product_id)                                    AS n_products,
                   COUNT(*) FILTER (WHERE action = 'delete')                     AS n_deletes,
                   COUNT(*) FILTER (WHERE action = 'scan')                       AS n_scans,
                   MAX(gap) / 60.0                                               AS max_gap_min,
                   STDDEV_SAMP(gap) / NULLIF(AVG(gap), 0)                        AS gap_cv,
                   MIN(t)                                                        AS first_t,
                   SUM(is_late_scan)                                             AS late_scans
            FROM ev GROUP BY 1
        ),
        rep AS (
            SELECT basket_id, COUNT(*) AS repeat_scans FROM (
                SELECT basket_id, product_id FROM {events} WHERE action = 'scan'
                GROUP BY 1, 2 HAVING COUNT(*) > 1
            ) GROUP BY 1
        )
        SELECT s.basket_id, s.member_id, s.store_id, s.day, s.hour_of_day, s.channel,
               s.duration_min,
               t.trip_order,
               COALESCE(a.scanned_units, 0)  AS scanned_units,
               COALESCE(a.scanned_value, 0)  AS scanned_value,
               COALESCE(a.hr_units, 0)       AS hr_units,
               COALESCE(a.hr_value, 0)       AS hr_value,
               COALESCE(a.n_products, 0)     AS n_products,
               COALESCE(a.n_deletes, 0)      AS n_deletes,
               a.max_gap_min, a.gap_cv,
               a.first_t / (s.duration_min * 60) AS first_scan_frac,
               COALESCE(a.late_scans * 1.0 / NULLIF(a.n_scans, 0), 0) AS late_share,
               COALESCE(r.repeat_scans, 0)   AS repeat_scans,
               (s.hour_of_day IN ({peaks}))::INT AS is_peak,
               (s.channel = 'app')::INT      AS is_app
        FROM {sessions} s
        JOIN obs_trips t USING (basket_id)
        LEFT JOIN agg a USING (basket_id)
        LEFT JOIN rep r USING (basket_id)
    """


def build(con: duckdb.DuckDBPyConnection, peak_hours=(16, 17, 18)) -> pd.DataFrame:
    """One row per SmartShop session with all observable features.

    Requires smartshop.basket.data.prepare(con) to have created obs_trips and obs_basket_comm.
    """
    con.execute(
        "CREATE OR REPLACE TEMP TABLE risk_session AS "
        + session_sql("sim_obs.sessions", "sim_obs.scan_events", peak_hours)
    )
    # Every trip the store observed (till and SmartShop), with what it recorded.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE risk_trip AS
        WITH hr AS (
            SELECT commodity, BOOL_OR(is_high_risk) AS hr FROM marts.dim_product GROUP BY 1
        )
        SELECT t.basket_id, t.member_id, t.trip_order, t.day, t.is_smartshop,
               COALESCE(SUM(b.units), 0) AS units, COALESCE(SUM(b.value), 0) AS value,
               COALESCE(SUM(b.value * COALESCE(hr.hr, false)::INT), 0) AS hr_value
        FROM obs_trips t
        -- every trip counts, including a SmartShop session where nothing was scanned
        LEFT JOIN obs_basket_comm b USING (basket_id)
        LEFT JOIN hr USING (commodity)
        GROUP BY 1, 2, 3, 4, 5
        """
    )
    # Member history at each SmartShop session: frames end at 1 PRECEDING (earlier trips only).
    df = con.sql(
        """
        WITH trips AS (
            SELECT rt.basket_id, rt.member_id, rt.trip_order, rt.day, rt.is_smartshop,
                   rt.units, rt.value, rt.hr_value,
                   rs.duration_min, rs.n_deletes, rs.scanned_units,
                   rr.rescan_type, rr.found_missing_units
            FROM risk_trip rt
            LEFT JOIN risk_session rs USING (basket_id)
            LEFT JOIN sim_obs.rescans rr USING (basket_id)
        ),
        hist AS (
            SELECT basket_id,
                   COUNT(*) OVER past                                           AS prior_trips,
                   SUM(is_smartshop::INT) OVER past                        AS prior_ss_sessions,
                   MIN(CASE WHEN is_smartshop THEN day END) OVER past           AS first_ss_day,
                   AVG(units) OVER past                                         AS m_avg_units,
                   SUM(hr_value) OVER past / NULLIF(SUM(value) OVER past, 0)    AS m_hr_share,
                   SUM(duration_min) OVER past
                       / NULLIF(SUM(scanned_units) OVER past, 0)                AS m_min_per_unit,
                   SUM(n_deletes) OVER past
                       / NULLIF(SUM(scanned_units) OVER past, 0)                AS m_delete_rate,
                   COUNT(rescan_type) OVER past                                 AS prior_rescans,
                   SUM((found_missing_units > 0)::INT) OVER past                AS prior_found,
                   COUNT(*) FILTER (WHERE rescan_type = 'full') OVER past
                       AS prior_full_rescans,
                   SUM((rescan_type = 'full' AND found_missing_units > 0)::INT) OVER past
                       AS prior_full_found
            FROM trips
            WINDOW past AS (
                PARTITION BY member_id ORDER BY trip_order
                ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
            )
        )
        SELECT rs.*, h.* EXCLUDE (basket_id)
        FROM risk_session rs JOIN hist h USING (basket_id)
        """
    ).df()
    return derive(df)


HISTORY_COLUMNS = [
    "prior_trips",
    "prior_ss_sessions",
    "first_ss_day",
    "m_avg_units",
    "m_hr_share",
    "m_min_per_unit",
    "m_delete_rate",
    "prior_rescans",
    "prior_found",
    "prior_full_rescans",
    "prior_full_found",
]


def derive(df: pd.DataFrame) -> pd.DataFrame:
    """Ratios and comparisons with the member's usual behaviour."""
    df = df.copy()
    n = df.scanned_units.clip(lower=1)
    df["delete_rate"] = df.n_deletes / n
    df["min_per_unit"] = df.duration_min / n
    df["mean_unit_price"] = df.scanned_value / n
    df["hr_value_share"] = df.hr_value / df.scanned_value.where(df.scanned_value > 0)
    df["days_since_first_ss"] = (df.day - df.first_ss_day).fillna(-1)
    df["units_vs_usual"] = df.scanned_units / df.m_avg_units
    df["hr_share_gap"] = df.m_hr_share - df.hr_value_share
    df["min_per_unit_vs_usual"] = df.min_per_unit / df.m_min_per_unit
    for c in ["prior_found", "prior_rescans", "prior_full_rescans", "prior_full_found"]:
        df[c] = df[c].fillna(0)
    df["prior_found_rate"] = df.prior_found / df.prior_rescans.where(df.prior_rescans > 0)
    cols = list(dict.fromkeys(SESSION_FEATURES + HISTORY_FEATURES + NOT_FOR_SCORING))
    df[cols] = df[cols].astype("float32").replace([np.inf, -np.inf], np.nan)
    return df.sort_values("basket_id").reset_index(drop=True)


def add_basket_gap(con, df: pd.DataFrame, model, chunk: int = 4000) -> pd.DataFrame:
    """Total expected unscanned value from the basket model (week 4), scored in chunks."""
    from smartshop.basket.model import score_sessions

    parts = []
    ids = df.basket_id.to_numpy()
    for i in range(0, len(ids), chunk):
        sc = score_sessions(con, model, ids[i : i + chunk])
        parts.append(sc.groupby("basket_id").exp_value.sum())
    gap = pd.concat(parts).rename("gap_value")
    out = df.merge(gap, left_on="basket_id", right_index=True, how="left")
    out["gap_per_unit"] = out.gap_value / (out.scanned_units + 1)
    out[GAP_FEATURES] = out[GAP_FEATURES].astype("float32")
    return out
