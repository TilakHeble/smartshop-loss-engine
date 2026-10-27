"""Behaviour of the dbt marts on the synthetic sample warehouse (runs in CI)."""

import pandas as pd
import pytest

from smartshop.config import TABLES_USED, read_seed
from smartshop.ingest.load_raw import SPECS
from smartshop.warehouse import connect


@pytest.fixture(scope="module")
def con(sample_warehouse):
    c = connect(sample_warehouse)
    yield c
    c.close()


def df(con, sql):
    return con.sql(sql).df()


def test_only_scanned_items_reach_fct_scan_line(con):
    bad = df(
        con,
        """
        SELECT l.* FROM marts.fct_scan_line l
        JOIN seeds.non_scan_departments n USING (department)
        UNION ALL
        SELECT * FROM marts.fct_scan_line WHERE quantity < 1 OR nectar_value <= 0
        """,
    )
    assert bad.empty
    # returns and fuel exist in the raw sample, so the filter is doing real work
    raw = df(con, "SELECT MIN(sales_value) v, MAX(quantity) q FROM raw.transactions").iloc[0]
    assert raw.v < 0 and raw.q > 200


def test_nectar_and_normal_price(con):
    lines = df(con, "SELECT * FROM marts.fct_scan_line")
    assert (lines.nectar_value <= lines.shelf_value + 1e-9).all()
    assert lines.member_discount.ge(0).all()
    assert (lines.shelf_value - lines.nectar_value).sub(lines.member_discount).abs().max() < 1e-9
    assert 0.3 < lines.has_nectar_price.mean() < 0.7


def test_every_seed_entry_maps_to_its_group(con):
    groups = df(con, "SELECT DISTINCT risk_group FROM marts.fct_scan_line WHERE is_high_risk")
    expected = {r["risk_group"] for r in read_seed("high_risk_categories")}
    assert set(groups.risk_group) == expected


def test_value_rank_one_is_the_most_valuable_line(con):
    t = df(
        con,
        """
        SELECT basket_id, MAX(nectar_value) AS top,
               MAX(nectar_value) FILTER (WHERE value_rank = 1) AS rank1
        FROM marts.fct_scan_line GROUP BY 1
        """,
    )
    assert (t.top == t.rank1).all()


def test_trip_totals_match_lines(con):
    t = df(con, "SELECT * FROM marts.fct_trip").set_index("basket_id")
    lines = df(con, "SELECT * FROM marts.fct_scan_line")
    agg = lines.groupby("basket_id").agg(v=("nectar_value", "sum"), n=("product_id", "size"))
    pd.testing.assert_series_equal(
        t.nectar_value.sort_index(), agg.v.sort_index(), check_names=False
    )
    assert (t.n_lines.sort_index() == agg.n.sort_index()).all()
    assert t.top3_value_share.between(0, 1 + 1e-9).all()


def test_history_counts_against_pandas(con):
    """Recompute history in pandas from fct_trip and compare with the SQL model."""
    t = df(con, "SELECT * FROM marts.fct_trip").sort_values(["member_id", "trip_seq"])
    h = df(con, "SELECT * FROM marts.fct_trip_history").set_index("basket_id")
    g = t.groupby("member_id")
    t["prior_trips"] = g.cumcount()
    t["prior_trips_this_store"] = t.groupby(["member_id", "store_id"]).cumcount()
    t["prior_avg_value"] = g.nectar_value.transform(lambda s: s.shift().expanding().mean())
    t["days_since_last_trip"] = g.day.diff()
    t = t.set_index("basket_id").loc[h.index]
    assert (t.prior_trips == h.prior_trips).all()
    assert (t.prior_trips_this_store == h.prior_trips_this_store).all()
    pd.testing.assert_series_equal(
        t.prior_avg_value, h.prior_avg_value, check_names=False, check_dtype=False
    )
    pd.testing.assert_series_equal(
        t.days_since_last_trip, h.days_since_last_trip.astype(float), check_names=False
    )


def test_demographics_stay_out_of_feature_tables(con):
    demo_cols = set(df(con, "SELECT * FROM staging.stg_member_demographics LIMIT 0").columns)
    for table in ["fct_scan_line", "fct_trip", "fct_trip_history", "dim_member", "dim_product"]:
        cols = set(df(con, f"SELECT * FROM marts.{table} LIMIT 0").columns)
        assert not (cols & (demo_cols - {"member_id"})), table


def test_ingest_reuses_nectar_specs_without_promotions():
    names = {s.name for s in SPECS}
    assert names == set(TABLES_USED)
    assert "promotions" not in names
