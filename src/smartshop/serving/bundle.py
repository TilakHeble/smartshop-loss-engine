"""The serving bundle, built by the weekly job, and the rescan decision it makes at the exit.

    artifacts/
        risk_model.joblib     session risk model (logistic: every decision has reasons)
        profiles.parquet      per member: history features, trust, usual high-risk products
        products.parquet      product -> product type (commodity), high-risk flag
        manifest.json         version, cutoff, policy, trust rule, budget, drift reference

Served model: the week 5 configuration without the basket-gap feature, which would need the
whole basket model at checkout. The stress tests ran on exactly this configuration and give the
same conclusion (11.4% less loss vs 11.7% with the gap).

Build it:
    python -m smartshop.serving.bundle
"""

from __future__ import annotations

import argparse
import functools
import json
import logging
import secrets
import shutil
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smartshop.config import ROOT, WAREHOUSE_PATH
from smartshop.monitoring.drift import profile
from smartshop.policy.neutral import SizeNeutraliser
from smartshop.serving.features import feature_row, session_features

log = logging.getLogger(__name__)
DEFAULT_DIR = ROOT / "artifacts"
RECENT_DAYS = 28  # the latest cycle: calibrates rescan volume and is the drift "current" sample

REASONS = {
    "max_gap_min": "long pause between scans",
    "gap_cv": "irregular scanning pace",
    "n_deletes": "items deleted this shop",
    "delete_rate": "many deletions for the items scanned",
    "m_delete_rate": "usually deletes a lot",
    "min_per_unit": "slow for the items scanned",
    "min_per_unit_vs_usual": "slower than usual for the items scanned",
    "m_min_per_unit": "usually slow for items scanned",
    "first_scan_frac": "late first scan",
    "late_share": "many scans right at the end",
    "scanned_units": "many items",
    "units_vs_usual": "more items than usual",
    "scanned_value": "large basket value",
    "mean_unit_price": "low average price for the items scanned",
    "hr_units": "many high-risk items",
    "hr_value_share": "high share of high-risk items",
    "hr_share_gap": "fewer high-risk items than usual",
    "m_hr_share": "usually buys many high-risk items",
    "prior_found": "earlier rescans found items",
    "prior_found_rate": "earlier rescans often found items",
    "prior_full_found": "an earlier full rescan found items",
    "repeat_scans": "repeated scans of one product",
    "n_products": "many different products",
    "m_avg_units": "usually large baskets",
    "duration_min": "long shop",
}


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except Exception:  # noqa: BLE001 - best effort
        return "unknown"


def smooth_probability(score, t: float, floor: float, cap: float, tau: float):
    s = np.clip(np.asarray(score, float), 1e-6, 1 - 1e-6)
    z = np.clip((np.log(s / (1 - s)) - t) / tau, -50, 50)
    return floor + (cap - floor) / (1 + np.exp(-z))


@dataclass
class Bundle:
    model: object
    profiles: pd.DataFrame  # index member_id
    products: pd.DataFrame  # index product_id
    manifest: dict
    # The latest cycle's sessions with their rescan probabilities: used by the weekly job's
    # release gates, never saved.
    recent: pd.DataFrame | None = None

    @property
    def features(self) -> list[str]:
        return self.manifest["features"]

    @functools.cached_property
    def neutraliser(self) -> SizeNeutraliser | None:
        d = self.manifest["policy"].get("size_neutral")
        return SizeNeutraliser.from_dict(d) if d else None

    # ---------- decisions ----------
    def adjusted(self, score: float, scanned_units: float) -> float:
        """v1.1: the score compared with sessions of a similar size (unchanged in v1.0 bundles)."""
        strength = self.manifest["policy"].get("size_neutral_strength", 0.0)
        if self.neutraliser is None or not strength:
            return float(score)
        return float(self.neutraliser(score, scanned_units, strength)[0])

    def probability(self, score: float, trust_mult: float) -> float:
        """Rescan probability from the (size-adjusted) score and the member's trust."""
        p = self.manifest["policy"]
        base = smooth_probability(score, p["t"], p["floor"], p["cap"], p["tau"])
        return float(max(p["floor"], base * trust_mult))

    def reasons(self, X: pd.DataFrame, k: int = 3) -> list[str]:
        lr = self.model.steps[-1][1]
        contrib = pd.Series(
            self.model[:-1].transform(X[self.features])[0] * lr.coef_[0], index=self.features
        )
        top = contrib[contrib > 0].sort_values(ascending=False).head(k).index
        return list(dict.fromkeys(REASONS.get(f, f) for f in top))

    def decide(
        self,
        member_id: int,
        duration_min: float,
        events: pd.DataFrame,
        day: int | None = None,
        draw: float | None = None,
    ) -> dict:
        ev = events.copy()
        known = ev.product_id.isin(self.products.index)
        ev["is_high_risk"] = (
            ev.product_id.map(self.products.is_high_risk).where(known, False).astype(bool)
        )
        sess = session_features(ev, duration_min) | {"duration_min": duration_min}
        prof = self.profiles.loc[member_id].to_dict() if member_id in self.profiles.index else None
        X = feature_row(sess, prof, day=day or self.manifest["cutoff_day"] + 1)
        score = float(self.model.predict_proba(X[self.features])[:, 1][0])
        adjusted = self.adjusted(score, sess["scanned_units"])
        trust = float(prof["trust_mult"]) if prof else 1.0
        p = self.probability(adjusted, trust)
        u = float(draw) if draw is not None else secrets.randbelow(10**9) / 10**9
        floor = self.manifest["policy"]["floor"]
        rescan = u < p
        look = []
        if prof and isinstance(prof.get("usual_high_risk"), str) and prof["usual_high_risk"]:
            scanned = set(ev.product_id.map(self.products.commodity).dropna())
            look = [c for c in prof["usual_high_risk"].split("|") if c not in scanned]
        return {
            "decision": "full_rescan" if rescan else "no_rescan",
            "rescan_probability": round(p, 4),
            "draw": round(u, 6),
            "audit_sample": bool(u < floor),
            "risk_score": round(score, 4),
            "size_adjusted_score": round(adjusted, 4),
            "reasons": self.reasons(X) if rescan else [],
            "look_here_first": look[:3] if rescan else [],
            "trust_multiplier": round(trust, 3),
            "known_member": prof is not None,
            "unknown_products": int((~known).sum()),
        }

    # ---------- persistence ----------
    def save(self, out_dir: Path) -> Path:
        out_dir = Path(out_dir)
        tmp = out_dir.with_name(out_dir.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir(parents=True)
        joblib.dump(self.model, tmp / "risk_model.joblib")
        self.profiles.reset_index().to_parquet(tmp / "profiles.parquet", index=False)
        self.products.reset_index().to_parquet(tmp / "products.parquet", index=False)
        (tmp / "manifest.json").write_text(json.dumps(self.manifest, indent=2, default=float))
        # Swap in atomically so a running API never reads a half-written bundle.
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp.rename(out_dir)
        return out_dir

    @classmethod
    def load(cls, path: Path) -> Bundle:
        path = Path(path)
        return cls(
            model=joblib.load(path / "risk_model.joblib"),
            profiles=pd.read_parquet(path / "profiles.parquet").set_index("member_id"),
            products=pd.read_parquet(path / "products.parquet").set_index("product_id"),
            manifest=json.loads((path / "manifest.json").read_text()),
        )


# --------------------------------------------------------------------------------------------
# The weekly batch build
# --------------------------------------------------------------------------------------------


def member_profiles(con, cutoff_day: int) -> pd.DataFrame:
    """History features as of the cutoff: the same aggregates a session would see (all trips)."""
    return con.sql(
        f"""
        WITH trips AS (
            SELECT rt.*, rs.duration_min, rs.n_deletes, rs.scanned_units,
                   rr.rescan_type, rr.found_missing_units
            FROM risk_trip rt
            LEFT JOIN risk_session rs USING (basket_id)
            LEFT JOIN sim_obs.rescans rr USING (basket_id)
            WHERE rt.day <= {cutoff_day}
        )
        SELECT member_id,
               COUNT(*)                                                AS prior_trips,
               SUM(is_smartshop::INT)                                  AS prior_ss_sessions,
               MIN(CASE WHEN is_smartshop THEN day END)                AS first_ss_day,
               AVG(units)                                              AS m_avg_units,
               SUM(hr_value) / NULLIF(SUM(value), 0)                   AS m_hr_share,
               SUM(duration_min) / NULLIF(SUM(scanned_units), 0)       AS m_min_per_unit,
               SUM(n_deletes) / NULLIF(SUM(scanned_units), 0)          AS m_delete_rate,
               COUNT(rescan_type)                                      AS prior_rescans,
               SUM((found_missing_units > 0)::INT)                     AS prior_found,
               COUNT(*) FILTER (WHERE rescan_type = 'full')            AS prior_full_rescans,
               SUM((rescan_type = 'full' AND found_missing_units > 0)::INT) AS prior_full_found
        FROM trips GROUP BY member_id
        """
    ).df()


def usual_high_risk(con, cutoff_day: int, min_share: float = 0.2, top: int = 3) -> pd.Series:
    """Each member's regular high-risk product types over the last year: 'look here first'."""
    d = con.sql(
        f"""
        WITH hr AS (SELECT DISTINCT commodity FROM marts.dim_product WHERE is_high_risk),
        trips AS (
            SELECT member_id, COUNT(*) AS n FROM obs_trips
            WHERE day BETWEEN {cutoff_day - 364} AND {cutoff_day} GROUP BY 1
        )
        SELECT t.member_id, b.commodity, COUNT(*) * 1.0 / ANY_VALUE(n.n) AS share
        FROM obs_basket_comm b JOIN obs_trips t USING (basket_id) JOIN trips n USING (member_id)
        WHERE t.day BETWEEN {cutoff_day - 364} AND {cutoff_day}
          AND b.commodity IN (SELECT commodity FROM hr)
        GROUP BY 1, 2
        """
    ).df()
    d = d[d.share >= min_share].sort_values(["member_id", "share"], ascending=[True, False])
    return d.groupby("member_id").commodity.apply(lambda c: "|".join(c.head(top)))


def build_bundle(
    con, cutoff_day: int | None = None, policy: dict | None = None, min_baskets: int | None = None
) -> Bundle:
    """Everything the API needs as of `cutoff_day` (default: the latest day in the data)."""
    from smartshop.basket.data import prepare
    from smartshop.policy.search import POLICY_PATH
    from smartshop.policy.trust import RECOMMENDED_TRUST, history_events, multipliers
    from smartshop.risk.features import build
    from smartshop.risk.model import FEATURE_SETS, LABELS, candidates, observed_labels

    prepare(con) if min_baskets is None else prepare(con, min_baskets)
    feats = build(con)
    cutoff = int(cutoff_day or feats.day.max())
    feats = feats[feats.day <= cutoff]
    cols = FEATURE_SETS["session+history"]
    labels = feats.merge(observed_labels(con), on="basket_id")
    model = candidates()["logistic"].fit(labels[cols], LABELS["found anything"](labels))

    prof = member_profiles(con, cutoff).set_index("member_id")
    ev = history_events(con, cutoff + 1)
    trust = multipliers(ev, prof.index.to_numpy(), cutoff + 1, RECOMMENDED_TRUST)
    prof["trust_mult"] = trust.reindex(prof.index).fillna(1.0)
    prof["usual_high_risk"] = usual_high_risk(con, cutoff).reindex(prof.index).fillna("")
    products = (
        con.sql("SELECT product_id, commodity, is_high_risk FROM marts.dim_product")
        .df()
        .set_index("product_id")
    )

    # Rescan volume: today's colleague time per session, on the latest cycle's sessions.
    pol = dict(policy or json.loads(POLICY_PATH.read_text())["recommended"])
    recent = feats[feats.day > cutoff - RECENT_DAYS]
    raw = model.predict_proba(recent[cols])[:, 1]
    score = raw
    strength = pol.get("size_neutral_strength", 0.0)
    if strength:
        # v1.1 size bands from the last year's sessions before this cycle (quantiles only)
        year = feats[(feats.day > cutoff - 365) & (feats.day <= cutoff - RECENT_DAYS)]
        neutral = SizeNeutraliser.fit(
            model.predict_proba(year[cols])[:, 1], year.scanned_units.clip(lower=0)
        )
        pol["size_neutral"] = neutral.to_dict()
        score = neutral(raw, recent.scanned_units.clip(lower=0).to_numpy(), strength)
    mult = recent.member_id.map(prof.trust_mult).fillna(1.0).to_numpy()
    units = recent.scanned_units.clip(lower=0).to_numpy()
    budget = (0.10 * (45 + 6 * np.minimum(3, units)) + 0.02 * (45 + 4 * units)).sum()
    cost = 45 + 4 * units
    lo, hi = -12.0, 12.0
    for _ in range(50):
        t = (lo + hi) / 2
        p = np.maximum(
            pol["floor"], smooth_probability(score, t, pol["floor"], pol["cap"], pol["tau"]) * mult
        )
        lo, hi = (t, hi) if (p * cost).sum() > budget else (lo, t)
    pol["t"] = hi
    p = np.maximum(
        pol["floor"], smooth_probability(score, hi, pol["floor"], pol["cap"], pol["tau"]) * mult
    )

    recent = recent.assign(risk_score=raw, size_adjusted_score=score, p=p, cost_s=cost)

    # Drift reference: the latest cycle, i.e. what this bundle was calibrated on. Not the whole
    # training history: member-history features mature over time, so early sessions (little or
    # no history) would make every new cycle look drifted.
    now = datetime.now(timezone.utc)
    manifest = {
        "bundle_version": now.strftime("%Y%m%dT%H%M%SZ") + "-" + _git_sha(),
        "built_at": now.isoformat(),
        "cutoff_day": cutoff,
        "features": cols,
        "training_labels": {
            "random_full_rescans": int(len(labels)),
            "found_anything": int(LABELS["found anything"](labels).sum()),
        },
        "policy": pol,
        "trust_rule": asdict(RECOMMENDED_TRUST),
        "budget": {
            "colleague_seconds_per_session": float(budget / len(recent)),
            "expected_rescan_rate": float(p.mean()),
            "recent_sessions": int(len(recent)),
        },
        "members_profiled": int(len(prof)),
        "members_trusted": int((prof.trust_mult < 1).sum()),
        "products": int(len(products)),
        "drift_reference": profile(recent, cols + ["risk_score"]),
        "library_versions": {
            p: version(p) for p in ["scikit-learn", "numpy", "pandas", "joblib", "pyarrow"]
        },
    }
    return Bundle(model=model, profiles=prof, products=products, manifest=manifest, recent=recent)


def main() -> None:
    from smartshop.warehouse import connect

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    parser.add_argument("--out", default=str(DEFAULT_DIR))
    parser.add_argument("--cutoff-day", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    con = connect(args.warehouse)
    b = build_bundle(con, args.cutoff_day)
    path = b.save(Path(args.out))
    m = b.manifest
    log.info(
        "bundle %s -> %s (%s members, rescan rate %.1f%%)",
        m["bundle_version"],
        path,
        m["members_profiled"],
        100 * m["budget"]["expected_rescan_rate"],
    )


if __name__ == "__main__":
    main()
