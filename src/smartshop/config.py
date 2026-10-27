"""Project paths and constants."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
WAREHOUSE_PATH = DATA_DIR / "warehouse" / "smartshop.duckdb"
FIGURES_DIR = ROOT / "reports" / "figures"

# Households join the panel over weeks 1-16 (Nectar finding 1), so analysis windows start at 17.
FIRST_FULL_WEEK = 17

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

# Lines that are not items a shopper scans: fuel and till-only entries (same rule as Nectar's
# stg_transactions, plus misc till and coupon lines). Returns, voids and zero lines are also out.
NON_SCAN_DEPARTMENTS = ("KIOSK-GAS", "MISC SALES TRAN", "MISC. TRANS.", "COUP/STR & MFG")

# Week 1 working list of high-risk categories: the kinds of product UK retailers commonly
# security-tag or keep behind the counter (fresh meat, alcohol, tobacco, health & beauty, baby,
# coffee). It is a starting assumption; v2 replaces it with a data-driven hotspot ranking.
HIGH_RISK_DEPARTMENTS = {"MEAT": "Fresh meat & fish", "SEAFOOD": "Fresh meat & fish"}
HIGH_RISK_COMMODITIES = {
    "Fresh meat & fish": ["SEAFOOD-FRESH", "MEAT - MISC"],
    "Alcohol": ["BEERS/ALES", "DOMESTIC WINE", "IMPORTED WINE", "MISC WINE", "LIQUOR"],
    "Tobacco": ["CIGARETTES", "CIGARS", "TOBACCO OTHER"],
    "Health & beauty": [
        "COLD AND FLU",
        "ANALGESICS",
        "ANTACIDS",
        "VITAMINS",
        "NATURAL VITAMINS",
        "SHAVING CARE PRODUCTS",
        "ORAL HYGIENE PRODUCTS",
        "HAIR CARE PRODUCTS",
        "DEODORANTS",
        "FRAGRANCES",
        "MAKEUP AND TREATMENT",
        "COSMETIC ACCESSORIES",
        "BATTERIES",
    ],
    "Baby": ["INFANT FORMULA", "DIAPERS & DISPOSABLES", "BABY HBC", "INFANT CARE PRODUCTS"],
    "Coffee": ["COFFEE"],
}
