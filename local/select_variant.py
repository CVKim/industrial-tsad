"""Tuning-list selection (docs/protocol.md): variant runs vs a reference run with the dataset rule; the passing variant
with the highest dataset mean is selected (or, if none passes, the best one is returned for reporting only).

  python local/select_variant.py --reference results/tsbad_M/mscad_tun --variants results/tsbad_M/mscad_modes16_tun,results/tsbad_M/mscad_modes64_tun \
         --out results/tsbad_M/selection_mscad_modes.json
"""
import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401
from industrial_tsad.evaluators.stats import DatasetRule, select
from industrial_tsad.runners import load_run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--variants", required=True, help="comma-separated run folders")
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max_drop", type=float, default=0.05)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = load_run(Path(a.reference))["VUS-PR"]
    variants = {Path(v).name: load_run(Path(v))["VUS-PR"] for v in a.variants.split(",")}
    res = select(variants, ref, DatasetRule(a.tol, a.max_drop))
    res["reference"] = a.reference
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, r in res["variants"].items():
        print(f"{k:32s} dataset {r['dataset_mean']:.4f} (ref {r['ref_dataset_mean']:.4f}) up/down {r['up']}/{r['down']} "
              f"worst {r['worst']:+.3f} {r['worst_dataset']} {'PASS' if r['pass'] else ''}")
    print("selected:", res["selected"], "(passed)" if res["passed"] else "(no variant passed: reporting only)")


if __name__ == "__main__":
    main()
