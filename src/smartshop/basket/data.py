"""What Nectar observes about each shopper, and the commodity statistics the basket model uses.

Nectar sees a complete basket for a staffed-till trip, but only the *scanned* items for a
SmartShop session (unscanned items are, by definition, not recorded). Member history is built
from exactly that record, so the model never learns from something the store could not know.

Three periods keep statistics, training and evaluation apart:
    STATS  days 113-330  commodity popularity and co-occurrence (lift)
    TRAIN  days 331-500  masked till baskets to train on
    TEST   days 501-711  evaluation: masked till baskets and real SmartShop sessions
"""

from __future__ import annotations

import duckdb

STATS_DAYS = (113, 330)
TRAIN_DAYS = (331, 500)
TEST_DAYS = (501, 711)
MIN_BASKETS = 200  # a commodity enters the vocabulary if bought in >= 200 STATS baskets


def prepare(con: duckdb.DuckDBPyConnection, min_baskets: int = MIN_BASKETS) -> None:
    """Create temp tables: obs_trips, obs_basket_comm, mc_hist, comm_stats, lift."""
    con.execute("SET enable_progress_bar = false")
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE obs_trips AS
        WITH ss AS (SELECT basket_id FROM sim_obs.sessions)
        SELECT t.basket_id, t.member_id, t.store_id, t.day, t.checkout_minute, t.hour_of_day,
               t.basket_id IN (SELECT basket_id FROM ss) AS is_smartshop,
               ROW_NUMBER() OVER (
                   PARTITION BY t.member_id ORDER BY t.day, t.checkout_minute, t.basket_id
               ) AS trip_order
        FROM marts.fct_trip t
        """
    )
    # Observed units per basket x commodity: full basket at a till, scanned (net) in SmartShop.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE obs_basket_comm AS
        WITH till AS (
            SELECT l.basket_id, l.commodity, SUM(l.quantity) AS units, SUM(l.nectar_value) AS value
            FROM marts.fct_scan_line l
            JOIN obs_trips t USING (basket_id)
            WHERE NOT t.is_smartshop
            GROUP BY 1, 2
        ),
        ss AS (
            SELECT e.basket_id, p.commodity,
                   SUM(CASE WHEN e.action = 'scan' THEN 1 ELSE -1 END) AS units,
                   SUM(CASE WHEN e.action = 'scan' THEN e.unit_price ELSE -e.unit_price END)
                       AS value
            FROM sim_obs.scan_events e
            JOIN marts.dim_product p USING (product_id)
            GROUP BY 1, 2
        )
        SELECT * FROM till
        UNION ALL
        SELECT * FROM ss WHERE units > 0
        """
    )
    # Running member x commodity history, one row per trip that contained the commodity.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE mc_hist AS
        SELECT t.member_id, b.commodity, t.trip_order, t.day,
               ROW_NUMBER() OVER w AS cum_trips,
               SUM(b.units) OVER w AS cum_units
        FROM obs_basket_comm b JOIN obs_trips t USING (basket_id)
        WINDOW w AS (PARTITION BY t.member_id, b.commodity ORDER BY t.trip_order)
        """
    )
    lo, hi = STATS_DAYS
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE stats_baskets AS
        SELECT b.* FROM obs_basket_comm b JOIN obs_trips t USING (basket_id)
        WHERE t.day BETWEEN {lo} AND {hi} AND NOT t.is_smartshop
        """
    )
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE comm_stats AS
        WITH n AS (SELECT COUNT(DISTINCT basket_id) AS n FROM stats_baskets),
        risk AS (
            SELECT commodity, BOOL_OR(is_high_risk) AS is_high_risk
            FROM marts.dim_product GROUP BY 1
        )
        SELECT s.commodity,
               COUNT(*) * 1.0 / ANY_VALUE(n.n)          AS popularity,
               AVG(s.units)                             AS avg_units,
               AVG((s.units > 1)::INT)                  AS multi_share,
               SUM(s.value) / SUM(s.units)              AS unit_price,
               ANY_VALUE(r.is_high_risk)                AS is_high_risk,
               COUNT(*)                                 AS n_baskets
        FROM stats_baskets s
        JOIN risk r USING (commodity)
        CROSS JOIN n
        GROUP BY s.commodity
        HAVING COUNT(*) >= {min_baskets}
        """
    )
    # Log-lift between commodities: how much more often c appears with j than by chance.
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE lift AS
        WITH b AS (
            SELECT basket_id, commodity FROM stats_baskets
            WHERE commodity IN (SELECT commodity FROM comm_stats)
        ),
        n AS (SELECT COUNT(DISTINCT basket_id) AS n FROM stats_baskets),
        pair AS (
            SELECT a.commodity AS c, b2.commodity AS j, COUNT(*) AS n_cj
            FROM b a JOIN b b2 ON a.basket_id = b2.basket_id AND a.commodity <> b2.commodity
            GROUP BY 1, 2
        )
        SELECT p.c, p.j,
               LN((p.n_cj + 1) / (sc.popularity * sj.popularity * n.n + 1)) AS log_lift
        FROM pair p
        CROSS JOIN n
        JOIN comm_stats sc ON sc.commodity = p.c
        JOIN comm_stats sj ON sj.commodity = p.j
        """
    )


def vocabulary(con: duckdb.DuckDBPyConnection) -> list[str]:
    return [r[0] for r in con.execute("SELECT commodity FROM comm_stats ORDER BY 1").fetchall()]
