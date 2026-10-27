"""Data-quality contract for the warehouse and the scan_lines view.

Runs against data/warehouse/smartshop.duckdb and is skipped when it has not been built
(e.g. in CI, where raw data is not committed).
"""

import pytest

from smartshop.config import NON_SCAN_DEPARTMENTS, WAREHOUSE_PATH

pytestmark = pytest.mark.skipif(not WAREHOUSE_PATH.exists(), reason="warehouse not built")


@pytest.fixture(scope="module")
def con():
    from smartshop.lines import connect

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
def test_row_counts(con, table, expected_rows):
    assert scalar(con, f"SELECT COUNT(*) FROM raw.{table}") == expected_rows


def test_scan_lines_size(con):
    assert scalar(con, "SELECT COUNT(*) FROM scan_lines") == 2_348_715
    assert scalar(con, "SELECT COUNT(DISTINCT basket_id) FROM scan_lines") == 230_489


def test_no_non_scan_departments(con):
    depts = ", ".join(f"'{d}'" for d in NON_SCAN_DEPARTMENTS)
    assert scalar(con, f"SELECT COUNT(*) FROM scan_lines WHERE department IN ({depts})") == 0


def test_prices_are_consistent(con):
    bad = scalar(
        con,
        """
        SELECT COUNT(*) FROM scan_lines
        WHERE nectar_value <= 0 OR shelf_value < nectar_value - 1e-9 OR member_discount < 0
        """,
    )
    assert bad == 0


def test_quantities_are_plausible(con):
    assert scalar(con, "SELECT MIN(quantity) FROM scan_lines") >= 1
    assert scalar(con, "SELECT MAX(quantity) FROM scan_lines") <= 200


def test_each_basket_has_one_member_store_and_day(con):
    bad = scalar(
        con,
        """
        SELECT COUNT(*) FROM (
            SELECT basket_id FROM scan_lines GROUP BY 1
            HAVING COUNT(DISTINCT member_id) > 1 OR COUNT(DISTINCT store_id) > 1
                OR COUNT(DISTINCT day) > 1
        )""",
    )
    assert bad == 0


def test_high_risk_share_is_stable(con):
    share = scalar(
        con,
        "SELECT SUM(nectar_value) FILTER (WHERE risk_group IS NOT NULL) / SUM(nectar_value) "
        "FROM scan_lines",
    )
    assert 0.15 < share < 0.25
