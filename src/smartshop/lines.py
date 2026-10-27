"""The `scan_lines` view: one row per item line a SmartShop shopper would scan.

Week 1 builds it as a DuckDB view so the EDA and tests share one definition. Week 2 moves the
same logic into dbt staging models.

Prices, in the units the data records (US dollars):
    shelf_value   normal price of the line (what a non-member pays)
    nectar_value  member price of the line (shelf price less the loyalty-card discount)
    member_discount = shelf_value - nectar_value
"""

from __future__ import annotations

import duckdb

from smartshop.config import (
    FIRST_FULL_WEEK,
    HIGH_RISK_COMMODITIES,
    HIGH_RISK_DEPARTMENTS,
    NON_SCAN_DEPARTMENTS,
    WAREHOUSE_PATH,
)


def _quote(values) -> str:
    return ", ".join("'" + v.replace("'", "''") + "'" for v in values)


def risk_group_sql(dept: str = "p.department", commodity: str = "p.commodity") -> str:
    """SQL CASE expression mapping a product to its high-risk group, or NULL."""
    whens = [
        f"WHEN {commodity} IN ({_quote(items)}) THEN '{group}'"
        for group, items in HIGH_RISK_COMMODITIES.items()
    ]
    whens += [f"WHEN {dept} = '{d}' THEN '{group}'" for d, group in HIGH_RISK_DEPARTMENTS.items()]
    return "CASE " + " ".join(whens) + " END"


SCAN_LINES_SQL = f"""
CREATE OR REPLACE TEMP VIEW scan_lines AS
SELECT
    t.household_id                                  AS member_id,
    t.basket_id,
    t.store_id,
    t.day,
    t.week_no,
    t.trans_time_hhmm // 100                        AS hour_of_day,
    t.product_id,
    p.department,
    p.commodity,
    t.quantity,
    -- 3 of 2.3m lines record a positive "discount" (a data error): floor it at zero.
    t.sales_value - t.coupon_match_disc + GREATEST(-t.retail_disc, 0)  AS shelf_value,
    t.sales_value - t.coupon_match_disc             AS nectar_value,
    GREATEST(-t.retail_disc, 0)                     AS member_discount,
    {risk_group_sql()}                              AS risk_group
FROM raw.transactions t
JOIN raw.products p USING (product_id)
WHERE p.department NOT IN ({_quote(NON_SCAN_DEPARTMENTS)})
  AND t.quantity > 0
  AND t.sales_value > 0
  AND t.week_no >= {FIRST_FULL_WEEK}
"""


def connect(path=WAREHOUSE_PATH) -> duckdb.DuckDBPyConnection:
    """Read-only connection to the warehouse with the `scan_lines` view defined."""
    con = duckdb.connect(str(path), read_only=True)
    con.execute("SET enable_progress_bar = false")
    con.execute(SCAN_LINES_SQL)
    return con
