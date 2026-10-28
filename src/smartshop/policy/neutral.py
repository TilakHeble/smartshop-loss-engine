"""Size-neutral risk scores (policy v1.1).

The risk model learns "will a rescan find anything", and a big basket nearly always holds an
honest mistake or two. So raw scores favour big baskets, which are also the slowest to rescan,
and honest families who buy big baskets are stopped more (the week 10 age gap).

The fix compares each session only with sessions of a similar size: its score is replaced by
its rank among sessions with a similar number of units scanned, mapped back onto the overall
score distribution (so the policy's shape and volume still apply). `strength` blends the raw
and the size-neutral score on the log-odds scale: 0 is the v1.0 policy, 1 fully size-neutral.

Chosen on the tuning period (days 331-500): strength 0.5 gives most of the loss reduction for a
modest rise in honest stops; the test period then confirmed it (README, "v1.1").

Only units scanned is used, which the store sees; no demographics.
"""

from __future__ import annotations

import numpy as np

SIZE_NEUTRAL_STRENGTH = 0.5
N_BANDS = 10
GRID = np.linspace(0, 1, 201)


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class SizeNeutraliser:
    """Score quantiles overall and within bands of units scanned (no individual data kept)."""

    def __init__(self, edges, band_q: dict, overall_q):
        self.edges = np.asarray(edges, float)
        self.band_q = {int(b): np.asarray(q, float) for b, q in band_q.items()}
        self.overall_q = np.asarray(overall_q, float)

    @classmethod
    def fit(cls, score, units, n_bands: int = N_BANDS) -> SizeNeutraliser:
        score, units = np.asarray(score, float), np.asarray(units, float)
        edges = np.unique(np.quantile(units, np.linspace(0, 1, n_bands + 1)[1:-1]))
        band = np.searchsorted(edges, units, side="right")
        band_q = {int(b): np.quantile(score[band == b], GRID) for b in np.unique(band)}
        return cls(edges, band_q, np.quantile(score, GRID))

    def neutral(self, score, units) -> np.ndarray:
        """Fully size-neutral score: the session's rank within its band, on the overall scale."""
        score, units = np.atleast_1d(np.asarray(score, float)), np.atleast_1d(units)
        band = np.searchsorted(self.edges, np.asarray(units, float), side="right")
        out = np.empty(len(score))
        for b in np.unique(band):
            m = band == b
            q = self.band_q.get(int(b), self.overall_q)
            # rank within the band, then the overall score at that rank
            rank = np.interp(score[m], q, GRID, left=0, right=1)
            out[m] = np.interp(rank, GRID, self.overall_q)
        return out

    def __call__(self, score, units, strength: float = SIZE_NEUTRAL_STRENGTH) -> np.ndarray:
        score = np.atleast_1d(np.asarray(score, float))
        if strength == 0:
            return score
        z = (1 - strength) * _logit(score) + strength * _logit(self.neutral(score, units))
        return 1 / (1 + np.exp(-z))

    def to_dict(self) -> dict:
        return {
            "edges": self.edges.tolist(),
            "band_quantiles": {str(b): q.tolist() for b, q in self.band_q.items()},
            "overall_quantiles": self.overall_q.tolist(),
        }

    @classmethod
    def from_dict(cls, d: dict) -> SizeNeutraliser:
        return cls(d["edges"], d["band_quantiles"], d["overall_quantiles"])


def neutralised_game(g, neutral: SizeNeutraliser, strength: float = SIZE_NEUTRAL_STRENGTH):
    """A copy of a rescan game with every version's score made (partly) size-neutral."""
    from smartshop.policy.game import GameData

    s = g.sessions.copy()
    for k in range(len(g.levels) + 1):
        s[f"v{k}_score"] = neutral(
            s[f"v{k}_score"].to_numpy(), s[f"v{k}_scanned_units"].to_numpy(), strength
        )
    return GameData(s, g.cfg, g.levels)
