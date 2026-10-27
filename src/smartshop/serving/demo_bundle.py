"""A small synthetic bundle with the real structure, for CI and the Docker smoke test.

The real bundle needs the dunnhumby data, which is never committed; this one is fitted on random
numbers so the API, the container and the hot reload can be tested anywhere.

    python -m smartshop.serving.demo_bundle --out artifacts
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from smartshop.monitoring.drift import profile
from smartshop.policy.trust import RECOMMENDED_TRUST
from smartshop.risk.model import FEATURE_SETS, candidates
from smartshop.serving.bundle import Bundle


def demo_bundle(seed: int = 0, n: int = 2000) -> Bundle:
    rng = np.random.default_rng(seed)
    cols = FEATURE_SETS["session+history"]
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    y = (X[cols[0]] + rng.normal(size=n) > 1).astype(int)
    model = candidates()["logistic"].fit(X, y)
    members = np.arange(1, 51)
    profiles = pd.DataFrame(
        {
            "member_id": members,
            "prior_trips": 20,
            "prior_ss_sessions": 10,
            "first_ss_day": 100,
            "m_avg_units": 12.0,
            "m_hr_share": 0.15,
            "m_min_per_unit": 1.2,
            "m_delete_rate": 0.05,
            "prior_rescans": 2,
            "prior_found": 0,
            "prior_full_rescans": 1,
            "prior_full_found": 0,
            "trust_mult": np.where(members % 2 == 0, 0.75, 1.0),
            "usual_high_risk": np.where(members % 3 == 0, "BEEF|CHEESE", ""),
        }
    ).set_index("member_id")
    products = pd.DataFrame(
        {
            "product_id": np.arange(1, 201),
            "commodity": np.where(np.arange(1, 201) <= 20, "BEEF", "GROCERY"),
            "is_high_risk": np.arange(1, 201) <= 20,
        }
    ).set_index("product_id")
    manifest = {
        "bundle_version": "demo",
        "built_at": "1970-01-01T00:00:00+00:00",
        "cutoff_day": 711,
        "features": cols,
        "training_labels": {"random_full_rescans": n, "found_anything": int(y.sum())},
        "policy": {"kind": "smooth", "floor": 0.02, "cap": 0.5, "tau": 0.05, "t": 0.0},
        "trust_rule": asdict(RECOMMENDED_TRUST),
        "budget": {
            "colleague_seconds_per_session": 0.0,
            "expected_rescan_rate": 0.0,
            "recent_sessions": 0,
        },
        "members_profiled": len(profiles),
        "members_trusted": int((profiles.trust_mult < 1).sum()),
        "products": len(products),
        "drift_reference": profile(X, cols),
        "library_versions": {},
    }
    return Bundle(model=model, profiles=profiles, products=products, manifest=manifest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="artifacts")
    args = parser.parse_args()
    print(demo_bundle().save(Path(args.out)))


if __name__ == "__main__":
    main()
