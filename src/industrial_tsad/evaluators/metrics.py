"""Point-score metrics. VUS-PR is the TSB-AD ranking metric."""
import numpy as np

from ..utils.external import ensure_tsbad


def robust_z(s, axis=-1):
    """label-free robust z of a score series: (s - median) / (1.4826 MAD), MAD floored (dtype kept, as in the runs)"""
    s = np.asarray(s)
    med = np.median(s, axis=axis, keepdims=True)
    mad = 1.4826 * np.median(np.abs(s - med), axis=axis, keepdims=True)
    return (s - med) / np.maximum(mad, 1e-4 * (1 + np.abs(med)))


def threshold_free_metrics(score, label, sliding_window):
    """the threshold-free part of TSB-AD get_metrics with the same calls and arguments: AUC-PR, AUC-ROC, VUS-PR,
    VUS-ROC (generate_curve, version 'opt', 250 thresholds) and the oracle point F1. The PA / event / range /
    affiliation F1 are left out (about 70 % of the run time on long series, not used for the ranking)."""
    ensure_tsbad()
    from TSB_AD.evaluation.basic_metrics import basic_metricor, generate_curve

    score = np.nan_to_num(np.asarray(score))  # dtype kept (float32 PaAno scores were evaluated as float32)
    label = np.asarray(label).astype(int)
    g = basic_metricor()
    _, _, _, _, _, _, vus_roc, vus_pr = generate_curve(label, score, sliding_window, "opt", 250)
    return {"AUC-PR": g.metric_PR(label, score), "AUC-ROC": g.metric_ROC(label, score), "VUS-PR": vus_pr,
            "VUS-ROC": vus_roc, "Standard-F1": g.metric_PointF1(label, score, preds=None)}
