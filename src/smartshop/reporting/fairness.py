"""The full fairness check of the recommended policy, as it would run (week 10).

Being asked for a full rescan is the adverse outcome. The check looks at **honest shoppers**
(nothing skipped on purpose), because groups can differ in how much deliberate non-scanning they
contain, and being stopped when honest is the harm.

For every attribute it reports:
- each group's honest-stop rate under the policy, and under today's flat random rescans;
- the four-fifths ratio (lowest / highest group rate), with an interval that resamples members;
- the same ratio after standardising for basket size (bigger baskets carry more risk for every
  shopper, so this separates "stopped more because of the group" from "stopped more because
  of the basket");
- the most-stopped group's rate against today's: whether any group is worse off than now.

Demographics are joined here only; no model or policy sees them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SESSIONS = 200
N_BOOT = 300
SIZE_BANDS = [0, 5, 10, 20, 40, np.inf]

ATTRIBUTES = {
    "channel": "channel",
    "new SmartShop user (<5 sessions)": "new_smartshop_user",
    "evening peak": "is_peak",
    "age group": "age_group",
    "household size": "household_size",
    "income level": "income_level",
}


def session_attributes(con, feats: pd.DataFrame) -> pd.DataFrame:
    demo = con.sql(
        "SELECT member_id, age_group, income_level, household_size "
        "FROM staging.stg_member_demographics"
    ).df()
    a = feats[
        [
            "basket_id",
            "member_id",
            "channel",
            "is_peak",
            "prior_ss_sessions",
            "scanned_units",
            "hour_of_day",
        ]
    ].copy()
    a["new_smartshop_user"] = a.prior_ss_sessions.fillna(0) < 5
    a["size_band"] = pd.cut(a.scanned_units.clip(lower=0), SIZE_BANDS, right=False)
    return a.merge(demo, on="member_id", how="left")


def _rates(d: pd.DataFrame, col: str) -> pd.Series:
    return d.groupby(col, observed=True).p.mean()


def _standardised(d: pd.DataFrame, col: str) -> pd.Series:
    """Group rates if every group had the overall basket-size mix (direct standardisation)."""
    w = d.size_band.value_counts(normalize=True)
    band = d.groupby([col, "size_band"], observed=True).p.mean().unstack()
    band = band.reindex(columns=w.index)
    # a group missing a band takes the overall band rate there
    band = band.fillna(d.groupby("size_band", observed=True).p.mean())
    return (band * w).sum(axis=1)


def _ratio(r: pd.Series) -> float:
    return float(r.min() / r.max()) if r.max() > 0 else 1.0


def permutation_ratios(d: pd.DataFrame, col: str, rng, n: int = N_BOOT) -> np.ndarray:
    """Four-fifths ratios if households were assigned to groups at random (same group sizes).

    With many groups and few households in each, the lowest/highest ratio is well below 1 by
    chance alone, because households differ in their habits. This is the yardstick for "is the
    gap bigger than chance?".
    """
    m = d.groupby("member_id").agg(sp=("p", "sum"), n=("p", "size"), g=(col, "first"))
    labels = m.g.to_numpy()
    out = np.empty(n)
    for i in range(n):
        shuffled = rng.permutation(labels)
        sp = pd.Series(m.sp.to_numpy()).groupby(shuffled).sum()
        cnt = pd.Series(m.n.to_numpy()).groupby(shuffled).sum()
        out[i] = _ratio(sp / cnt)
    return out


def fairness_table(
    sessions: pd.DataFrame, today_rate: float, seed: int = 0, n_boot: int = N_BOOT
) -> pd.DataFrame:
    """`sessions`: one row per session with p (mean over runs), honest, and the attributes."""
    rng = np.random.default_rng(seed)
    honest = sessions[sessions.honest]
    rows = []
    for label, col in ATTRIBUTES.items():
        d = honest.dropna(subset=[col])
        size = d.groupby(col).size()
        d = d[d[col].isin(size[size >= MIN_SESSIONS].index)]
        if d[col].nunique() < 2:
            continue
        rates = _rates(d, col)
        members = d.member_id.unique()
        by_member = dict(list(d.groupby("member_id")))
        boot = []
        for _ in range(n_boot):
            pick = rng.choice(members, len(members))
            b = pd.concat([by_member[m] for m in pick])
            if b[col].nunique() == d[col].nunique():
                boot.append(_ratio(_rates(b, col)))
        null = permutation_ratios(d, col, rng, n_boot)
        rows.append(
            {
                "attribute": label,
                "groups": int(d[col].nunique()),
                "fewest households in a group": int(d.groupby(col).member_id.nunique().min()),
                "honest sessions": int(len(d)),
                "lowest group rate": float(rates.min()),
                "highest group rate": float(rates.max()),
                "most-stopped group": str(rates.idxmax()),
                "four-fifths ratio": _ratio(rates),
                "ratio CI low": float(np.percentile(boot, 2.5)),
                "ratio CI high": float(np.percentile(boot, 97.5)),
                "ratio, same basket sizes": _ratio(_standardised(d, col)),
                "ratio if groups were random (median)": float(np.median(null)),
                "chance of a gap this large": float((null <= _ratio(rates)).mean()),
                "highest rate vs today": float(rates.max() / today_rate),
            }
        )
    out = pd.DataFrame(rows).set_index("attribute")
    out["verdict"] = np.select(
        [
            out["four-fifths ratio"] >= 0.8,
            out["ratio, same basket sizes"] >= 0.8,
            out["chance of a gap this large"] >= 0.05,
        ],
        ["pass", "gap explained by basket size", "gap within chance for groups this small"],
        "review",
    )
    return out


def group_rates(sessions: pd.DataFrame, today_rate: float) -> pd.DataFrame:
    """Honest-stop rate of every group (for the dashboard and the model card)."""
    honest = sessions[sessions.honest]
    rows = []
    for label, col in ATTRIBUTES.items():
        d = honest.dropna(subset=[col])
        for g, x in d.groupby(col):
            if len(x) < MIN_SESSIONS:
                continue
            rows.append(
                {
                    "attribute": label,
                    "group": str(g),
                    "honest sessions": len(x),
                    "households": x.member_id.nunique(),
                    "policy": x.p.mean(),
                    "today": today_rate,
                }
            )
    return pd.DataFrame(rows)
