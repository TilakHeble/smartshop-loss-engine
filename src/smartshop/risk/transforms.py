"""Feature transforms used inside saved pipelines (a stable import path keeps pickles loadable)."""

import numpy as np


def signed_log(x):
    return np.sign(x) * np.log1p(np.abs(x))
