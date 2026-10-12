"""Write a pre-registration record before running an evaluation (refuses to overwrite; use --append for later entries).

  python local/prereg.py --out results/prereg/fusion_u.json --question "..." --design "..." --selection "..." --claim "..."
  python local/prereg.py --out results/prereg/fusion_u.json --append '{"result": "..."}'
"""
import argparse
import json

import _bootstrap  # noqa: F401
from industrial_tsad.registry import append, write


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--question")
    ap.add_argument("--design")
    ap.add_argument("--selection")
    ap.add_argument("--claim")
    ap.add_argument("--append", default=None, help="JSON object appended to 'later'")
    a = ap.parse_args()
    if a.append:
        rec = append(a.out, json.loads(a.append))
    else:
        rec = write(a.out, a.question, a.design, a.selection, a.claim)
    print(json.dumps({k: v for k, v in rec.items() if k != "code_sha256"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
