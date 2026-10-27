"""Project paths and constants.

Business rules that SQL needs (high-risk categories, non-scan departments) live in dbt seeds
(dbt/seeds/*.csv) so there is one source of truth; read_seed() exposes them to Python.
"""

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "smartshop.duckdb"
FIGURES_DIR = ROOT / "reports" / "figures"
DBT_DIR = ROOT / "dbt"
SEEDS_DIR = DBT_DIR / "seeds"

# Households join the panel over weeks 1-16 (Nectar finding 1), so analysis windows start at 17.
# Mirrors the dbt var first_stable_week.
FIRST_FULL_WEEK = 17
LAST_DAY = 711

# Tables SmartShop needs from the Complete Journey. The 700 MB in-store promotions file
# (causal_data) is not used: rescan decisions depend on baskets, prices and visits.
TABLES_USED = (
    "transactions",
    "products",
    "households",
    "campaigns",
    "campaign_members",
    "coupons",
    "coupon_redemptions",
)


def read_seed(name: str) -> list[dict[str, str]]:
    """Rows of a dbt seed CSV, e.g. read_seed('high_risk_categories')."""
    with open(SEEDS_DIR / f"{name}.csv", newline="") as f:
        return list(csv.DictReader(f))
