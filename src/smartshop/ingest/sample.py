"""Write a small synthetic warehouse with the same raw schema as the real data.

CI has no raw data (it is never committed), so this lets CI run the full dbt build, every dbt
test and the point-in-time check on data shaped like the real thing. It includes a product for
every high-risk seed entry, fuel and till lines, returns, multi-buys, members who use several
stores, and same-day repeat trips.

Run:
    python -m smartshop.ingest.sample data/warehouse/sample.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from smartshop.config import LAST_DAY, read_seed


def make_products() -> pd.DataFrame:
    rows = []
    pid = 1000
    for r in read_seed("high_risk_categories"):
        dept = r["value"] if r["match_on"] == "department" else "GROCERY"
        commodity = r["value"] if r["match_on"] == "commodity" else "BEEF"
        rows.append((pid, dept, commodity))
        pid += 1
    for i in range(40):
        rows.append((pid, ["GROCERY", "PRODUCE", "DELI", "PASTRY"][i % 4], f"ORDINARY {i}"))
        pid += 1
    for r in read_seed("non_scan_departments"):
        rows.append((pid, r["department"], "NON-SCAN"))
        pid += 1
    df = pd.DataFrame(rows, columns=["product_id", "department", "commodity"])
    df["manufacturer_id"] = 1
    df["brand_type"] = "National"
    df["sub_commodity"] = df.commodity
    df["pack_size"] = None
    return df


def make_transactions(products: pd.DataFrame, n_members: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    scan = products[products.commodity != "NON-SCAN"].product_id.to_numpy()
    non_scan = products[products.commodity == "NON-SCAN"].product_id.to_numpy()
    stores = np.array([101, 102, 103, 104])
    rows = []
    basket_id = 1
    for member in range(1, n_members + 1):
        home = rng.choice(stores)
        n_trips = int(rng.integers(5, 40))
        days = np.sort(rng.integers(1, LAST_DAY + 1, n_trips))
        for day in days:
            store = home if rng.random() < 0.7 else rng.choice(stores)
            hhmm = int(rng.integers(7, 23)) * 100 + int(rng.integers(0, 60))
            items = rng.choice(scan, size=int(rng.integers(1, 15)), replace=False)
            for pid in items:
                qty = int(rng.choice([1, 1, 1, 2, 4]))
                value = round(float(rng.uniform(0.5, 12)) * qty, 2)
                disc = -round(value * 0.2, 2) if rng.random() < 0.5 else 0.0
                rows.append((member, basket_id, day, hhmm, store, pid, qty, value, disc))
            if rng.random() < 0.1:  # fuel / till line
                rows.append(
                    (member, basket_id, day, hhmm, store, rng.choice(non_scan), 9000, 30.0, 0)
                )
            if rng.random() < 0.05:  # a return
                pid = rng.choice(np.setdiff1d(scan, items))
                rows.append((member, basket_id, day, hhmm, store, pid, 1, -3.0, 0.0))
            basket_id += 1
    df = pd.DataFrame(
        rows,
        columns=[
            "household_id",
            "basket_id",
            "day",
            "trans_time_hhmm",
            "store_id",
            "product_id",
            "quantity",
            "sales_value",
            "retail_disc",
        ],
    )
    df["week_no"] = (df.day - 1) // 7 + 1
    df["coupon_disc"] = 0.0
    df["coupon_match_disc"] = 0.0
    return df


def write_sample(path: Path, n_members: int = 80, seed: int = 0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    products = make_products()
    transactions = make_transactions(products, n_members, seed)
    households = pd.DataFrame(
        {
            "household_id": range(1, n_members // 3 + 1),
            "age_group": "35-44",
            "marital_group": "A",
            "income_level": "Level 4",
            "household_size": "2",
            "hh_composition_group": "Group 1",
            "homeowner": "Homeowner",
            "kids": "None/Unknown",
        }
    )
    con = duckdb.connect(str(path))
    con.execute("CREATE SCHEMA raw")
    for name, df in [
        ("products", products),
        ("transactions", transactions),
        ("households", households),
    ]:
        con.register("df", df)
        con.execute(f"CREATE TABLE raw.{name} AS SELECT * FROM df")
        con.unregister("df")
    con.close()
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    args = parser.parse_args()
    print(write_sample(args.path))


if __name__ == "__main__":
    main()
