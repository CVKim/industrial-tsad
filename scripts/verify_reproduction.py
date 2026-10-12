"""Check that the package reproduces saved reference runs (scores of earlier experiments) on a few series.

  python scripts/verify_reproduction.py --reference <folder with reference runs> --checks paano_final:eva_final \
         mscad:mscad_eva axonad:axonad_parts --files 002,031

For each check '<preset>:<reference run>' the preset is run on the selected TSB-AD-M eval series with the seed policy of
the preset and compared with <reference>/<run>/scores/<file>.npy (or, for AxonAD, <reference>/<run>/<file>.npz parts):
max |difference| of the point scores and of VUS-PR. GPU kernels are not bit-deterministic across runs, so tiny score
differences can appear; VUS-PR should agree to ~1e-3.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from industrial_tsad.configs import load_preset  # noqa: E402
from industrial_tsad.core.models import create_detector  # noqa: E402
from industrial_tsad.datasets import file_list, load  # noqa: E402
from industrial_tsad.evaluators.metrics import threshold_free_metrics  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", required=True)
    ap.add_argument("--checks", nargs="+", required=True)
    ap.add_argument("--list", default="TSB-AD-M-Eva.csv")
    ap.add_argument("--files", default="002", help="comma-separated file-name prefixes")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    files = file_list(a.list)
    pick = [(n, f) for n, f in enumerate(files) if f.split("_")[0] in a.files.split(",")]
    out = []
    for chk in a.checks:
        preset, run = chk.split(":")
        cfg = load_preset(preset)
        det = create_detector(cfg, a.device)
        for n, f in pick:
            s = load(f)
            seed = cfg.seed.base + (n if cfg.seed.per_series else 0)
            score = det.fit_score(s.x, s.train_end, seed=seed, dataset=s.dataset)
            sw = s.sliding_window
            row = {"check": chk, "file": f, "seed": seed}
            ref_dir = Path(a.reference) / run
            if cfg.type == "axonad":
                z = np.load(ref_dir / (f[:-4] + ".npz"))
                for k in ("mse", "jepa"):
                    row[f"max_abs_{k}"] = float(np.abs(det.parts[k][:len(z[k])] - z[k]).max())
                ref = (z["mse"] + z["jepa"]).astype(np.float64)
            else:
                ref = np.load(ref_dir / "scores" / (f[:-4] + ".npy"))
                row["max_abs_score"] = float(np.abs(np.asarray(score)[:len(ref)] - ref).max())
            n_ = min(len(score), len(ref), len(s.label))
            row["vus_new"] = threshold_free_metrics(np.asarray(score)[:n_], s.label[:n_], sw)["VUS-PR"]
            row["vus_ref"] = threshold_free_metrics(ref[:n_], s.label[:n_], sw)["VUS-PR"]
            row["vus_abs_diff"] = abs(row["vus_new"] - row["vus_ref"])
            print(json.dumps(row), flush=True)
            out.append(row)
    worst = max(r["vus_abs_diff"] for r in out)
    print(f"[done] {len(out)} checks, worst |VUS-PR diff| {worst:.2e}")


if __name__ == "__main__":
    main()
