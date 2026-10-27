"""Unit tests for the scan_lines definition on a tiny synthetic warehouse (runs in CI)."""

import duckdb
import pandas as pd
import pytest

from smartshop.config import HIGH_RISK_COMMODITIES, TABLES_USED
from smartshop.ingest.load_raw import SPECS
from smartshop.lines import connect, risk_group_sql


@pytest.fixture()
def tiny_warehouse(tmp_path):
    path = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA raw")
    con.execute(
        """
        CREATE TABLE raw.products AS SELECT * FROM (VALUES
            (1, 'MEAT', 'BEEF'),
            (2, 'GROCERY', 'BEERS/ALES'),
            (3, 'GROCERY', 'SOFT DRINKS'),
            (4, 'KIOSK-GAS', 'COUPON/MISC ITEMS')
        ) t(product_id, department, commodity)
        """
    )
    # household, basket, week, hhmm, store, product, qty, sales_value, retail_disc, coupon_match
    con.execute(
        """
        CREATE TABLE raw.transactions AS SELECT * FROM (VALUES
            (10, 100, 20, 1730, 5, 1, 1, 6.00, -2.00, 0.0),  -- beef with a Nectar price
            (10, 100, 20, 1730, 5, 2, 2, 8.00,  0.00, 0.0),  -- beer, multi-buy, no discount
            (10, 100, 20, 1730, 5, 3, 1, 1.00,  0.00, 0.0),  -- soft drink
            (10, 100, 20, 1730, 5, 4, 9000, 30.0, 0.00, 0.0),-- fuel: not a scanned item
            (10, 101, 20, 1800, 5, 3, 1, -1.00, 0.00, 0.0),  -- return: excluded
            (11, 102, 5, 900, 6, 3, 1, 1.00, 0.00, 0.0)      -- before week 17: excluded
        ) t(household_id, basket_id, week_no, trans_time_hhmm, store_id, product_id, quantity,
            sales_value, retail_disc, coupon_match_disc)
        """
    )
    con.execute("ALTER TABLE raw.transactions ADD COLUMN day SMALLINT DEFAULT 140")
    con.close()
    return path


def test_scan_lines_keeps_only_scanned_items(tiny_warehouse):
    con = connect(tiny_warehouse)
    df = con.sql("SELECT * FROM scan_lines ORDER BY product_id").df()
    assert df.product_id.tolist() == [1, 2, 3]
    assert set(df.basket_id) == {100}


def test_prices_and_discount(tiny_warehouse):
    df = connect(tiny_warehouse).sql("SELECT * FROM scan_lines ORDER BY product_id").df()
    beef = df.iloc[0]
    assert beef.shelf_value == pytest.approx(8.0)  # normal price
    assert beef.nectar_value == pytest.approx(6.0)  # Nectar price
    assert beef.member_discount == pytest.approx(2.0)
    assert (df.nectar_value <= df.shelf_value).all()
    assert df.hour_of_day.tolist() == [17, 17, 17]


def test_risk_groups(tiny_warehouse):
    df = connect(tiny_warehouse).sql("SELECT product_id, risk_group FROM scan_lines").df()
    groups = dict(zip(df.product_id, df.risk_group, strict=True))
    assert groups[1] == "Fresh meat & fish"
    assert groups[2] == "Alcohol"
    assert pd.isna(groups[3])


def test_risk_group_sql_handles_quotes():
    sql = risk_group_sql()
    assert sql.startswith("CASE") and sql.endswith("END")
    for items in HIGH_RISK_COMMODITIES.values():
        for item in items:
            assert item.replace("'", "''") in sql


def test_ingest_reuses_nectar_specs_without_promotions():
    names = {s.name for s in SPECS}
    assert names == set(TABLES_USED)
    assert "promotions" not in names
