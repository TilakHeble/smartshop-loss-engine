"""Drift monitoring with the Population Stability Index (as in the Nectar project).

    PSI = sum_b (cur_b - ref_b) * ln(cur_b / ref_b)

over bins fixed on the reference (training) data. Reading: < 0.10 stable, 0.10-0.25 warning,
>= 0.25 alert. The reference is stored as a small profile (bin edges and proportions), so it
ships inside the serving bundle and monitoring never needs the training data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WARN_PSI = 0.10
ALERT_PSI = 0.25
EPS = 1e-4
STATUS_ORDER = {"ok": 0, "warn": 1, "alert": 2}


def status_for(psi: float) -> str:
    return "alert" if psi >= ALERT_PSI else "warn" if psi >= WARN_PSI else "ok"


def _edges(values: np.ndarray, bins: int) -> np.ndarray:
    return np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)[1:-1]))


def _props(values: np.ndarray, edges: np.ndarray, missing: bool) -> list[float]:
    """Bin shares; the last entry is the share of missing values."""
    v = np.asarray(values, float)
    nan = np.isnan(v)
    idx = np.searchsorted(edges, v[~nan], side="left")
    counts = np.bincount(idx, minlength=len(edges) + 1).astype(float)
    counts = np.append(counts, nan.sum() if missing else 0)
    return (counts / max(len(v), 1)).tolist()


def profile(df: pd.DataFrame, columns: list[str], bins: int = 10) -> dict:
    out = {}
    for c in columns:
        v = df[c].to_numpy(float)
        finite = v[~np.isnan(v)]
        edges = _edges(finite, bins) if len(finite) else np.array([])
        out[c] = {"edges": edges.tolist(), "props": _props(v, edges, True)}
    return out


def psi(ref: list[float], cur: list[float]) -> float:
    r = np.clip(np.asarray(ref, float), EPS, None)
    c = np.clip(np.asarray(cur, float), EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def compare(reference: dict, df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, ref in reference.items():
        cur = _props(df[c].to_numpy(float), np.asarray(ref["edges"]), True)
        value = psi(ref["props"], cur)
        rows.append({"feature": c, "psi": value, "status": status_for(value)})
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)


def overall(report: pd.DataFrame) -> str:
    if report.empty:
        return "ok"
    return max(report.status, key=STATUS_ORDER.get)
