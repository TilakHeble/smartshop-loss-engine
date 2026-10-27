"""Train the basket model and score SmartShop sessions.

The model estimates, for each commodity in a session, the probability that it holds units that
are in the trolley but not on the scan list. Multiplying by the commodity's typical Nectar unit
price gives the expected value a check of that commodity would recover.

Run:
    python -m smartshop.basket.model
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score

from smartshop.basket.data import TEST_DAYS, TRAIN_DAYS, prepare
from smartshop.basket.features import (
    FEATURES,
    build_features,
    masked_training_pairs,
    session_pairs,
)
from smartshop.config import ROOT, WAREHOUSE_PATH
from smartshop.warehouse import connect

log = logging.getLogger(__name__)
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "basket_model.joblib"
METRICS_PATH = MODEL_DIR / "basket_metrics.json"

PARAMS = dict(
    n_estimators=400,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=100,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    verbose=-1,
)


def train(con, seed: int = 0, max_baskets: int | None = None) -> tuple[lgb.LGBMClassifier, dict]:
    pairs, visible = masked_training_pairs(con, TRAIN_DAYS, seed=seed, max_baskets=max_baskets)
    X = build_features(con, pairs, visible)
    model = lgb.LGBMClassifier(random_state=seed, **PARAMS)
    model.fit(X[FEATURES], X.label, sample_weight=X.weight)

    # Held-out check on masked till baskets from the later TEST period.
    vp, vv = masked_training_pairs(con, TEST_DAYS, seed=seed + 1, max_baskets=max_baskets)
    V = build_features(con, vp, vv)
    p = model.predict_proba(V[FEATURES])[:, 1]
    metrics = {
        "train_pairs": int(len(X)),
        "train_baskets": int(X.basket_id.nunique()),
        "masked_test_auc": float(roc_auc_score(V.label, p, sample_weight=V.weight)),
        "masked_test_ap": float(average_precision_score(V.label, p, sample_weight=V.weight)),
        "masked_test_logloss": float(log_loss(V.label, p, sample_weight=V.weight)),
        "masked_test_mean_pred": float(np.average(p, weights=V.weight)),
        "masked_test_base_rate": float(np.average(V.label, weights=V.weight)),
    }
    return model, metrics


def score_sessions(con, model, basket_ids) -> pd.DataFrame:
    """Per session x commodity: p_missing and expected value of checking it."""
    pairs, visible = session_pairs(con, basket_ids)
    X = build_features(con, pairs, visible)
    X["p_missing"] = model.predict_proba(X[FEATURES])[:, 1]
    X["exp_value"] = X.p_missing * X.unit_price
    return X


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    con = connect(Path(args.warehouse))
    prepare(con)
    model, metrics = train(con)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info(json.dumps(metrics, indent=2))
    log.info("trained in %.0fs -> %s", time.time() - t0, MODEL_PATH)


if __name__ == "__main__":
    main()
