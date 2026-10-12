"""Label-free score fusion of saved runs: S = sum_i w_i * robust_z(s_i), evaluated per series and compared (paired) with
each single run.

  python local/fuse.py --runs results/tsbad_U/paano_final_eva,results/tsbad_U/mscad_eva --weights 1,0.5 \
         --benchmark U --list TSB-AD-U-Eva.csv --out results/tsbad_U/fusion_paano_mscad_eva.json
"""
import argparse
import json

import numpy as np

import _bootstrap  # noqa: F401
from industrial_tsad.core.fusion import rz_sum
from industrial_tsad.datasets import file_list
from industrial_tsad.evaluators.stats import paired
from industrial_tsad.runners import fused_metric, load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", required=True, help="comma-separated run folders with saved scores")
    ap.add_argument("--weights", default=None, help="comma-separated weights (default 1 each)")
    ap.add_argument("--benchmark", choices=["M", "U"], default="M")
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    runs = a.runs.split(",")
    w = [float(x) for x in a.weights.split(",")] if a.weights else [1.0] * len(runs)
    fused = fused_metric(file_list(a.list), a.benchmark, runs, w, rz_sum)
    rng = np.random.default_rng(0)
    res = {"runs": runs, "weights": w, "fused_mean": round(float(fused.mean()), 4),
           "vs": {r: paired(fused, load_run(r)["VUS-PR"], rng=rng) for r in runs}}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({**res, "per_series": fused.round(6).to_dict()}, f, indent=1)
    print(json.dumps({k: (v if k != "vs" else {r: {q: x for q, x in d.items() if q != "by_dataset"} for r, d in v.items()})
                      for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
