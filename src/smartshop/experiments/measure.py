"""Measuring loss once rescans are targeted.

Under risk-based rescans the rescanned sessions are the risky ones, so "average loss found per
rescan x sessions" overstates total loss. Because every session's rescan probability is known,
inverse-probability (Horvitz-Thompson) weighting removes the bias: each rescanned session stands
for 1/p sessions like it. The 2% random floor gives a second, simpler unbiased estimate: the
sessions whose draw fell under the floor are a plain random sample.

Estimand: value left unscanned net of double scans, summed over sessions (what full rescans
measure). Detection by a full rescan (cfg.full_detect_nonscan) is corrected for.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimates(real: pd.DataFrame, floor: float, detect: float) -> dict:
    """Three estimates of total loss from one period's realised rescans (`run_cycles` output)."""
    r = real[real.rescanned]
    n = len(real)
    found = r.found_value.to_numpy()
    ht = (found / r.p.to_numpy()).sum() / detect
    naive = found.mean() * n / detect if len(r) else np.nan
    audit = real[real.u < floor]
    floor_only = audit.found_value.sum() / floor / detect
    return {
        "truth": real.net_loss.clip(lower=0).sum(),
        "inverse-probability weighted": ht,
        "random floor only": floor_only,
        "naive (average of rescans x sessions)": naive,
        "rescans": len(r),
        "audit sample": len(audit),
    }


def repeated_estimates(run, seeds, floor: float, detect: float) -> pd.DataFrame:
    """Estimates over many realisations; `run(seed)` returns a realised-rescans frame."""
    return pd.DataFrame([{"seed": s, **estimates(run(s), floor, detect)} for s in seeds])
