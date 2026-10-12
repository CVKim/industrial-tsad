"""Run a detector preset over a TSB-AD file list (several processes may share one run: claim files).

  python local/run_benchmark.py --preset paano_final --benchmark M --list TSB-AD-M-Tuning.csv --tag paano_final_tun
  python local/run_benchmark.py --preset src/industrial_tsad/presets/mscad.yaml --benchmark U --list TSB-AD-U-Eva.csv --tag mscad_u_eva
Two workers per 10 GB GPU at most (more thrash under Windows WDDM paging): set CUDA_VISIBLE_DEVICES per process.
"""
import argparse

import _bootstrap  # noqa: F401
from industrial_tsad.configs import BenchmarkRunConfig, load_preset
from industrial_tsad.runners import run_benchmark


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", required=True, help="preset name (industrial_tsad/presets) or YAML path")
    ap.add_argument("--benchmark", choices=["M", "U"], default="M")
    ap.add_argument("--list", default=None, help="official list name or csv path (default TSB-AD-<B>-Tuning.csv)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default=None)
    ap.add_argument("--no_claim", action="store_true", help="single worker without claim files")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = BenchmarkRunConfig(detector=load_preset(a.preset), benchmark=a.benchmark,
                             file_list=a.list or f"TSB-AD-{a.benchmark}-Tuning.csv", tag=a.tag, out_root=a.out_root,
                             claim=not a.no_claim)
    d = run_benchmark(cfg, a.device, log=lambda m: print(m, flush=True))
    if not d.empty:
        print(f"[done] {len(d)} series  VUS-PR {d['VUS-PR'].mean():.4f}  dataset mean {d.groupby('dataset')['VUS-PR'].mean().mean():.4f}")


if __name__ == "__main__":
    main()
