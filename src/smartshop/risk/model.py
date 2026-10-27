"""Session risk model: how likely is a full rescan of this session to find unscanned items?

Labels come only from what a store observes: random full rescans (unbiased, but few) and
optionally random partial rescans (many, but they rarely see concealed items). The model is
chosen by cross-validation on those observable labels only; the hidden truth is used once, at
the end, to evaluate.

Run:
    python -m smartshop.risk.model
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from smartshop.basket.data import TEST_DAYS, prepare
from smartshop.config import PROCESSED_DIR, ROOT, WAREHOUSE_PATH
from smartshop.risk.features import GAP_FEATURES, SCORING_HISTORY, SESSION_FEATURES, build
from smartshop.risk.transforms import signed_log
from smartshop.warehouse import connect

log = logging.getLogger(__name__)
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METRICS_PATH = MODEL_DIR / "risk_metrics.json"
GAP_CACHE = PROCESSED_DIR / "basket_gap.parquet"
TRAIN_LAST_DAY = TEST_DAYS[0] - 1

FEATURE_SETS = {
    "session": SESSION_FEATURES,
    "session+history": SESSION_FEATURES + SCORING_HISTORY,
    "session+history+gap": SESSION_FEATURES + SCORING_HISTORY + GAP_FEATURES,
}


def candidates(seed: int = 0) -> dict:
    """Small-data model families: rescan labels number in the hundreds to thousands."""
    return {
        "logistic": make_pipeline(
            # Missing history (new SmartShop users) is filled with the typical value, with no
            # "is missing" flag: no history must not raise the score.
            SimpleImputer(strategy="median"),
            FunctionTransformer(signed_log),
            StandardScaler(),
            LogisticRegression(C=0.3, max_iter=2000),
        ),
        "lightgbm": lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.03,
            num_leaves=7,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.7,
            reg_lambda=5.0,
            random_state=seed,
            verbose=-1,
        ),
    }


def load_features(con, with_gap: bool = True) -> pd.DataFrame:
    df = build(con)
    if with_gap:
        if GAP_CACHE.exists():
            gap = pd.read_parquet(GAP_CACHE)
        else:
            import joblib as jl

            from smartshop.basket.model import MODEL_PATH as BASKET_PATH
            from smartshop.risk.features import add_basket_gap

            gap = add_basket_gap(con, df[["basket_id", "scanned_units"]], jl.load(BASKET_PATH))
            gap = gap[["basket_id"] + GAP_FEATURES]
            GAP_CACHE.parent.mkdir(parents=True, exist_ok=True)
            gap.to_parquet(GAP_CACHE)
        df = df.merge(gap, on="basket_id", how="left")
    return df


# Observable targets a store could train on, all from random full rescans.
LABELS = {
    "found anything": lambda r: (r.found_missing_units > 0).astype(int),
    "found $5+": lambda r: (r.found_missing_value >= 5).astype(int),
    "found $10+": lambda r: (r.found_missing_value >= 10).astype(int),
}


def observed_labels(con) -> pd.DataFrame:
    """Random full rescans and what they found (observable, unbiased: selection was random)."""
    return con.sql(
        """
        SELECT basket_id, found_missing_units, found_missing_value
        FROM sim_obs.rescans WHERE rescan_type = 'full'
        """
    ).df()


def value_capture(score: np.ndarray, value: np.ndarray, share: float) -> float:
    """Share of `value` held by the top `share` of sessions by score."""
    k = max(1, int(np.ceil(share * len(score))))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    return float(np.asarray(value)[top].sum() / np.asarray(value).sum())


def cv_value_capture(model, X, y, value, share=0.2, folds=5, repeats=10, seed=0) -> float:
    """Out-of-fold share of value found by full rescans that the top `share` would have caught.

    Full rescans in the history were chosen at random, so this is an unbiased, observable
    estimate of what the score is worth: the only selection criterion. With labels this scarce
    one split is noisy, so the estimate is averaged over repeated random splits.
    """
    from sklearn.base import clone

    caps = []
    for r in range(repeats):
        oof = np.zeros(len(y))
        for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed + r).split(X, y):
            m = clone(model).fit(X.iloc[tr], y.iloc[tr])
            oof[va] = m.predict_proba(X.iloc[va])[:, 1]
        caps.append(value_capture(oof, value, share))
    return float(np.mean(caps))


def select_and_fit(df: pd.DataFrame, labels: pd.DataFrame, seed: int = 0):
    """Pick label x feature set x model by observable CV value capture (training period)."""
    train = df[df.day <= TRAIN_LAST_DAY].merge(labels, on="basket_id")
    value = train.found_missing_value.to_numpy()
    results = []
    for l_name, make_y in LABELS.items():
        y = make_y(train)
        if y.sum() < 10:
            continue
        for fs_name, cols in FEATURE_SETS.items():
            if not set(cols) <= set(train.columns):
                continue
            for m_name, model in candidates(seed).items():
                score = cv_value_capture(model, train[cols], y, value, seed=seed)
                results.append(
                    {
                        "label": l_name,
                        "features": fs_name,
                        "model": m_name,
                        "cv_value_capture_top20": score,
                    }
                )
    res = pd.DataFrame(results).sort_values("cv_value_capture_top20", ascending=False)
    best = res.iloc[0]
    cols = FEATURE_SETS[best.features]
    model = candidates(seed)[best.model].fit(train[cols], LABELS[best.label](train))
    return model, cols, res, len(train)


def fit_oracle(df: pd.DataFrame, truth: pd.DataFrame, cols, seed: int = 0):
    """Reference only: the same model trained on hidden truth for every training session."""
    train = df[df.day <= TRAIN_LAST_DAY].merge(truth, on="basket_id")
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.7,
        random_state=seed,
        verbose=-1,
    )
    return model.fit(train[cols], train.deliberate)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warehouse", default=str(WAREHOUSE_PATH))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    t0 = time.time()
    con = connect(args.warehouse)
    prepare(con)
    df = load_features(con)
    model, cols, res, n = select_and_fit(df, observed_labels(con))
    log.info("\n%s", res.round(4).to_string(index=False))
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": cols}, MODEL_PATH)
    metrics = {
        "n_training_labels": n,
        "selection": res.round(4).to_dict(orient="records"),
        "chosen": {
            "label": res.iloc[0].label,
            "features": res.iloc[0].features,
            "model": res.iloc[0].model,
        },
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    log.info("trained in %.0fs -> %s", time.time() - t0, MODEL_PATH)


if __name__ == "__main__":
    main()
