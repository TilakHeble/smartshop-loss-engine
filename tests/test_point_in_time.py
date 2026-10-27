"""No leakage: shopper history must only use the past.

The dbt project is rebuilt with the data cut off at an earlier day (var max_day). Every trip on
or before the cut-off must get exactly the same history as in the full build. If any history
column used a later trip, the two builds would differ.

Runs on the synthetic sample in CI, and on the real warehouse when it has been built.
"""

import shutil

import pandas as pd
import pytest

from smartshop.config import WAREHOUSE_PATH
from smartshop.warehouse import connect, run_dbt

CUTOFF_DAY = 400


def history(path):
    con = connect(path)
    h = con.sql("SELECT * FROM marts.fct_trip_history").df()
    con.close()
    return h.set_index("basket_id").sort_index()


def check_point_in_time(full_path, tmp_path):
    truncated_path = tmp_path / "truncated.duckdb"
    shutil.copy(full_path, truncated_path)
    run_dbt(
        "run",
        warehouse=truncated_path,
        vars={"max_day": CUTOFF_DAY},
        target_path=tmp_path / "target",
    )
    full, cut = history(full_path), history(truncated_path)

    assert cut.day.max() <= CUTOFF_DAY
    assert len(cut) > 0
    expected = full[full.day <= CUTOFF_DAY]
    assert set(cut.index) == set(expected.index)
    pd.testing.assert_frame_equal(cut, expected.loc[cut.index], check_exact=False, rtol=1e-12)


def test_history_is_point_in_time_on_sample(sample_warehouse, tmp_path):
    check_point_in_time(sample_warehouse, tmp_path)


@pytest.mark.skipif(not WAREHOUSE_PATH.exists(), reason="warehouse not built")
def test_history_is_point_in_time_on_real_data(tmp_path):
    check_point_in_time(WAREHOUSE_PATH, tmp_path)
