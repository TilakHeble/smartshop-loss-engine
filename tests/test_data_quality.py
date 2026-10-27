"""Data-quality contract for the real warehouse (raw tables and dbt marts).

Runs against data/warehouse/smartshop.duckdb and is skipped when it has not been built
(e.g. in CI, where raw data is not committed). Expected figures pin the Week 1 analysis.
"""

import pytest

from smartshop.config import WAREHOUSE_PATH

pytestmark = pytest.mark.skipif(not WAREHOUSE_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def con():
    from smartshop.warehouse import connect

    c = connect()
    yield c
    c.close()


def scalar(con, sql):
    return con.execute(sql).fetchone()[0]


@pytest.mark.parametrize(
    "table, expected_rows",
    [
        ("transactions", 2_595_732),
        ("products", 92_353),
        ("households", 801),
        ("campaigns", 30),
        ("campaign_members", 7_208),
        ("coupons", 124_548),
        ("coupon_redemptions", 2_318),
    ],
)
def test_raw_row_counts(con, table, expected_rows):
    assert scalar(con, f"SELECT COUNT(*) FROM raw.{table}") == expected_rows


def test_marts_reproduce_week_1(con):
    assert scalar(con, "SELECT COUNT(*) FROM marts.fct_scan_line") == 2_348_715
    assert scalar(con, "SELECT COUNT(*) FROM marts.fct_trip") == 230_489
    assert scalar(con, "SELECT COUNT(*) FROM marts.dim_member") == 2_492
    assert scalar(con, "SELECT COUNT(*) FROM marts.dim_store") == 532
    assert scalar(con, "SELECT ROUND(SUM(nectar_value)) FROM marts.fct_trip") == 6_817_025


def test_high_risk_share_is_stable(con):
    share = scalar(con, "SELECT SUM(high_risk_value) / SUM(nectar_value) FROM marts.fct_trip")
    assert 0.15 < share < 0.25


def test_most_trips_have_history(con):
    share = scalar(con, "SELECT AVG((prior_trips >= 10)::INT) FROM marts.fct_trip_history")
    assert share > 0.85


def test_each_basket_has_one_member_store_and_day(con):
    bad = scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT basket_id FROM marts.fct_scan_line GROUP BY 1
            HAVING COUNT(DISTINCT member_id) > 1 OR COUNT(DISTINCT store_id) > 1
                OR COUNT(DISTINCT day) > 1
        )""",
    )
    assert bad == 0
