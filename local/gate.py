"""Regime gate: base run(s) + AxonAD JEPA part where the median JEPA z over the test part exceeds tau.
Several base runs (seeds) are gated one by one and their VUS-PR averaged per series (as the reported 3-seed result).

  python local/gate.py --base results/tsbad_M/paano_final_eva,results/tsbad_M/paano_final_eva_s2 \
         --axonad results/tsbad_M/axonad_eva --list TSB-AD-M-Eva.csv --out results/tsbad_M/gate_eva.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import _bootstrap  # noqa: F401
from industrial_tsad.core.fusion import RegimeGate
from industrial_tsad.datasets import file_list, load, published
from industrial_tsad.evaluators.metrics import threshold_free_metrics
from industrial_tsad.evaluators.stats import paired
from industrial_tsad.runners import load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="comma-separated base runs (saved scores)")
    ap.add_argument("--axonad", required=True, help="AxonAD run with saved parts (parts/<file>.npz: mse, jepa)")
    ap.add_argument("--benchmark", choices=["M", "U"], default="M")
    ap.add_argument("--list", required=True)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--weight", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    gate = RegimeGate(a.tau, a.weight)
    bases = a.base.split(",")
    base_vus = {b: load_run(b)["VUS-PR"] for b in bases}
    rows = []
    for f in file_list(a.list):
        s = load(f, a.benchmark)
        j = np.load(Path(a.axonad) / "parts" / (f[:-4] + ".npz"))["jepa"].astype(np.float64)
        row = {"file": f, "dataset": s.dataset}
        vals, base_vals = [], []
        for b in bases:
            p = np.load(Path(b) / "scores" / (f[:-4] + ".npy")).astype(np.float64)
            n = min(len(p), len(j), len(s.label))
            sc, on, stat = gate(p[:n], j[:n], s.train_end)
            vals.append(threshold_free_metrics(sc, s.label[:n], s.sliding_window)["VUS-PR"] if on else base_vus[b][f])
            base_vals.append(base_vus[b][f])
        row.update({"gate": int(on), "statistic": round(stat, 2), "gated": float(np.mean(vals)), "base": float(np.mean(base_vals))})
        rows.append(row)
    d = pd.DataFrame(rows).set_index("file")
    rng = np.random.default_rng(0)
    res = {"tau": a.tau, "weight": a.weight, "gate_on": int(d["gate"].sum()), "gated_mean": round(float(d["gated"].mean()), 4),
           "base_mean": round(float(d["base"].mean()), 4), "gated_vs_base": paired(d["gated"], d["base"], rng=rng)}
    try:
        res["gated_vs_AxonAD_published"] = paired(d["gated"], published("AxonAD", a.benchmark), rng=rng)
    except (FileNotFoundError, KeyError):
        pass
    d.to_csv(Path(a.out).with_suffix(".csv"))
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(json.dumps({k: (v if not isinstance(v, dict) else {q: x for q, x in v.items() if q != "by_dataset"}) for k, v in res.items()}, indent=1))


if __name__ == "__main__":
    main()
