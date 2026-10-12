import numpy as np

from ...evaluators.metrics import robust_z


def rz_sum(scores, weights=None):
    """S = sum_i w_i * robust_z(s_i) over equal-length score series (truncated to the shortest)"""
    n = min(len(s) for s in scores)
    weights = weights or [1.0] * len(scores)
    return sum(w * robust_z(np.nan_to_num(np.asarray(s)[:n])) for w, s in zip(weights, scores))
