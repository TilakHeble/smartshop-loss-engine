"""Evaluate session risk scores against the hidden truth (test period only).

"Flagging" the top share q of sessions for a full rescan, how much deliberately skipped value and
net loss sits in the flagged set, and how many flagged shoppers skipped nothing on purpose?
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import average_precision_score, roc_auc_score

from smartshop.risk.model import LABELS, TRAIN_LAST_DAY, candidates, value_capture

SIGNALS = [
    "delete_rate",
    "n_deletes",
    "max_gap_min",
    "gap_cv",
    "min_per_unit",
    "hr_value_share",
    "scanned_value",
    "m_delete_rate",
    "prior_found_rate",
    "gap_per_unit",
]


def truth(con) -> pd.DataFrame:
    return con.sql(
        """
        SELECT basket_id, gt_deliberate_value, gt_net_loss, gt_nonscan_value,
               gt_deliberate_value > 0 AS deliberate
        FROM sim_truth.sessions
        """
    ).df()


def flag_metrics(score, t: pd.DataFrame, share: float) -> dict:
    k = max(1, int(np.ceil(share * len(t))))
    top = np.argsort(-np.asarray(score), kind="stable")[:k]
    f = t.iloc[top]
    return {
        "deliberate_value_flagged": f.gt_deliberate_value.sum() / t.gt_deliberate_value.sum(),
        "net_loss_flagged": f.gt_net_loss.sum() / t.gt_net_loss.sum(),
        "deliberate_sessions_flagged": f.deliberate.sum() / t.deliberate.sum(),
        "flagged_with_nothing_deliberate": 1 - f.deliberate.mean(),
    }


def points_rule(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    """A store's rule of thumb: one point per red flag above its 80th percentile."""
    flags = ["delete_rate", "max_gap_min", "hr_value_share", "scanned_value"]
    cut = train[flags].quantile(0.8)
    pts = sum((test[c] > cut[c]).astype(int) for c in flags)
    return pts + 1e-6 * test.scanned_value.rank().to_numpy()  # ties: bigger baskets first


def best_single_signal(train_lab: pd.DataFrame) -> tuple[str, float]:
    """The single signal with the best observable value capture on random full rescans."""
    best = ("", -1.0)
    for s in SIGNALS:
        x = train_lab[s].fillna(train_lab[s].median()).to_numpy()
        v = value_capture(x, train_lab.found_missing_value.to_numpy(), 0.2)
        if v > best[1]:
            best = (s, v)
    return best


def scores_for_test(df, labels, risk_model, cols, oracle=None) -> tuple[pd.DataFrame, dict]:
    train = df[df.day <= TRAIN_LAST_DAY]
    test = df[df.day > TRAIN_LAST_DAY].reset_index(drop=True)
    signal, _ = best_single_signal(train.merge(labels, on="basket_id"))
    rng = np.random.default_rng(0)
    scores = {
        "Random (today)": rng.random(len(test)),
        "Biggest baskets": test.scanned_value.to_numpy(),
        "Rule of thumb (red-flag points)": points_rule(train, test),
        f"Best single signal ({signal})": test[signal].fillna(test[signal].median()).to_numpy(),
        "Risk model": risk_model.predict_proba(test[cols])[:, 1],
    }
    if oracle is not None:
        scores["Ceiling: trained on hidden truth"] = oracle.predict_proba(test[cols])[:, 1]
    return test, scores


def compare(test: pd.DataFrame, scores: dict, t: pd.DataFrame, share: float = 0.12):
    tt = test[["basket_id"]].merge(t, on="basket_id", how="left")
    rows = []
    for name, s in scores.items():
        rows.append(
            {
                "method": name,
                "auc_deliberate": roc_auc_score(tt.deliberate, s),
                "ap_deliberate": average_precision_score(tt.deliberate, s),
                **flag_metrics(s, tt, share),
            }
        )
    return pd.DataFrame(rows).set_index("method")


def capture_curve(test, scores, t, shares=None) -> pd.DataFrame:
    shares = np.linspace(0.01, 0.5, 50) if shares is None else shares
    tt = test[["basket_id"]].merge(t, on="basket_id", how="left")
    out = []
    for name, s in scores.items():
        order = np.argsort(-np.asarray(s), kind="stable")
        cum = np.cumsum(tt.gt_deliberate_value.to_numpy()[order]) / tt.gt_deliberate_value.sum()
        for q in shares:
            out.append({"method": name, "share": q, "captured": cum[int(np.ceil(q * len(tt))) - 1]})
    return pd.DataFrame(out)


def floor_experiment(con, df, cols, label: str, model_name: str, floors, seeds=(0, 1, 2)):
    """How good would the model be with more random full rescans to learn from?

    Re-creates the simulated world (same random draws), gives a random `floor` share of
    training-period sessions a full rescan, and trains the chosen configuration on what those
    rescans would have found. Evaluated on the hidden truth of the test period.
    """
    from smartshop.simulate.config import SimConfig
    from smartshop.simulate.rescan import FULL, rescan_outcomes
    from smartshop.simulate.simulator import build_world, load_baskets, simulate_behaviour

    cfg = SimConfig()
    world = build_world(*load_baskets(con), cfg)
    q = cfg.status_quo_partial_rate + cfg.status_quo_full_rate
    s, u = simulate_behaviour(world, cfg, rescan_prob=q)
    train_ids = df.loc[df.day <= TRAIN_LAST_DAY, "basket_id"].to_numpy()
    test = df[df.day > TRAIN_LAST_DAY]
    t = truth(con)
    tt = test[["basket_id"]].merge(t, on="basket_id", how="left")
    rows = []
    for floor in floors:
        for seed in seeds:
            rng = np.random.default_rng(1000 + seed)
            ids = rng.choice(train_ids, size=int(floor * len(train_ids)), replace=False)
            lab = rescan_outcomes(ids, np.full(len(ids), FULL), u, cfg)
            tr = df.merge(lab, on="basket_id")
            y = LABELS[label](tr)
            m = clone(candidates(seed)[model_name]).fit(tr[cols], y)
            p = m.predict_proba(test[cols])[:, 1]
            rows.append(
                {
                    "floor": floor,
                    "seed": seed,
                    "n_labels": len(ids),
                    "auc_deliberate": roc_auc_score(tt.deliberate, p),
                    "deliberate_value_flagged": flag_metrics(p, tt, 0.12)[
                        "deliberate_value_flagged"
                    ],
                }
            )
    return pd.DataFrame(rows)
