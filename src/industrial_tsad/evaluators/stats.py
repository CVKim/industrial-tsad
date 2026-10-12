"""Paired comparisons of two methods on the same series, and the selection / claim rules of the protocol
(docs/protocol.md)."""
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def dataset_of(file_name: str) -> str:
    """TSB-AD file names: <index>_<Dataset>_id_<k>_<domain>_tr_<train end>_1st_<first anomaly>.csv"""
    return file_name.split("_")[1]


def paired(a: pd.Series, b: pd.Series, rng: Optional[np.random.Generator] = None, n_boot: int = 10000,
           looks: int = 1, min_gain: float = 0.008) -> dict:
    """a vs b on the series both have (index = file name): mean difference, series bootstrap and dataset-cluster bootstrap
    95 % intervals, Wilcoxon p, datasets up / down at |d| > 0.01 and the claim rule (gain >= min_gain and the series
    interval excludes 0; with looks > 1 the interval is taken at the Bonferroni level 0.05 / looks)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1).dropna()
    d = (j["a"] - j["b"]).values
    ds = np.array([dataset_of(i) for i in j.index])
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    names = np.unique(ds)
    by = {n: d[ds == n] for n in names}
    cl = np.array([np.concatenate([by[n] for n in rng.choice(names, len(names), replace=True)]).mean() for _ in range(n_boot)])
    per = pd.DataFrame({"d": d, "ds": ds}).groupby("ds")["d"].mean()

    def q(x, lo=0.025):
        return [round(float(np.quantile(x, lo)), 4), round(float(np.quantile(x, 1 - lo)), 4)]

    out = {"n": int(len(j)), "a_mean": round(float(j["a"].mean()), 4), "b_mean": round(float(j["b"].mean()), 4),
           "diff": round(float(d.mean()), 4), "series_CI95": q(bs), "dataset_cluster_CI95": q(cl),
           "wilcoxon_p": float(wilcoxon(d).pvalue) if (d != 0).any() else 1.0,
           "datasets_up": int((per > 0.01).sum()), "datasets_down": int((per < -0.01).sum()),
           "claim": bool(d.mean() >= min_gain and np.quantile(bs, 0.025 / looks) > 0),
           "by_dataset": per.round(4).to_dict()}
    if looks > 1:
        out["looks"] = looks
        out["series_CI_bonferroni"] = q(bs, 0.025 / looks)
    return out


@dataclass(frozen=True)
class DatasetRule:
    """selection rule on a tuning list, on dataset means: more datasets up than down at |d| > tol and no dataset
    dropping more than max_drop"""
    tol: float = 0.01
    max_drop: float = 0.05

    def __call__(self, variant: pd.Series, reference: pd.Series) -> dict:
        ds = pd.Series([dataset_of(i) for i in variant.index], index=variant.index)
        D = pd.DataFrame({"v": variant, "r": reference}).dropna().groupby(ds).mean()
        dd = D["v"] - D["r"]
        return {"dataset_mean": round(float(D["v"].mean()), 4), "ref_dataset_mean": round(float(D["r"].mean()), 4),
                "up": int((dd > self.tol).sum()), "down": int((dd < -self.tol).sum()), "worst": round(float(dd.min()), 4),
                "worst_dataset": str(dd.idxmin()), "pass": bool((dd > self.tol).sum() > (dd < -self.tol).sum()
                                                                 and dd.min() >= -self.max_drop)}


def select(variants: dict, reference: pd.Series, rule: DatasetRule = DatasetRule()) -> dict:
    """variants: name -> per-series metric. The passing variant with the highest dataset mean; if none passes, the
    best by dataset mean is returned with passed = False (to be run for reporting only)."""
    res = {k: rule(v, reference) for k, v in variants.items()}
    ok = [k for k, r in res.items() if r["pass"]]
    pool = ok or list(res)
    best = max(pool, key=lambda k: res[k]["dataset_mean"])
    return {"selected": best, "passed": bool(ok), "variants": res}
