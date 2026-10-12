"""Paired comparison of two runs (or a run and a published method) on the series both have.

  python local/compare.py --a results/tsbad_M/paano_final_eva --b published:AxonAD --benchmark M
  python local/compare.py --a results/tsbad_U/fusion --b results/tsbad_U/paano_final_eva --benchmark U --looks 2
"""
import argparse
import json

import _bootstrap  # noqa: F401
from industrial_tsad.runners import compare


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True, help="run folder or published:<Method>")
    ap.add_argument("--b", required=True)
    ap.add_argument("--benchmark", choices=["M", "U"], default="M")
    ap.add_argument("--looks", type=int, default=1, help="number of comparisons on this eval set (Bonferroni)")
    ap.add_argument("--out", default=None, help="write the result as JSON")
    a = ap.parse_args()
    r = compare(a.a, a.b, a.benchmark, looks=a.looks)
    print(json.dumps({k: v for k, v in r.items() if k != "by_dataset"}, indent=1))
    if a.out:
        with open(a.out, "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1)


if __name__ == "__main__":
    main()
