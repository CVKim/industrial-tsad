"""Paired comparisons between runs (or a run and a published method) on the series both have."""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from ..datasets import tsbad
from ..evaluators.metrics import threshold_free_metrics
from ..evaluators.stats import paired
from .benchmark import load_run


def metric_of(source: str, benchmark: str = "M", metric: str = "VUS-PR") -> pd.Series:
    """a run folder (per_file*.csv) or 'published:<Method>' -> per-series metric"""
    if source.startswith("published:"):
        return tsbad.published(source.split(":", 1)[1], benchmark)
    return load_run(Path(source))[metric]


def compare(a: str, b: str, benchmark: str = "M", looks: int = 1, rng: Optional[np.random.Generator] = None) -> dict:
    return paired(metric_of(a, benchmark), metric_of(b, benchmark), rng=rng, looks=looks)


def fused_metric(files, benchmark, score_dirs, weights, combine) -> pd.Series:
    """VUS-PR of a label-free combination of saved scores: combine(list of float64 score arrays, weights) per series"""
    vals = {}
    for f in files:
        s = tsbad.load(f, benchmark)
        scores = [np.load(Path(d) / "scores" / (f[:-4] + ".npy")).astype(np.float64) for d in score_dirs]
        sc = combine(scores, weights)
        n = min(len(sc), len(s.label))
        vals[f] = threshold_free_metrics(sc[:n], s.label[:n], s.sliding_window)["VUS-PR"]
    return pd.Series(vals)
