"""Pre-registration records: write the question, the selection rule and the claim rule BEFORE a result exists, with the
SHA-256 of the code that will produce it (the whole industrial_tsad package by default), and append later entries (results,
documented deviations) to the same file instead of editing it."""
import datetime
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional

PACKAGE = Path(__file__).resolve().parents[1]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hashes(paths: Optional[Iterable] = None) -> dict:
    """SHA-256 of the given files, or of every .py / .yaml file of the package (relative names)"""
    if paths is None:
        paths = sorted(p for p in PACKAGE.rglob("*") if p.suffix in (".py", ".yaml") and "__pycache__" not in p.parts)
        return {str(p.relative_to(PACKAGE)).replace("\\", "/"): sha256(p) for p in paths}
    return {str(p): sha256(p) for p in paths}


def write(path, question: str, design: str, selection: str, claim: str, extra: Optional[dict] = None,
          code: Optional[Iterable] = None) -> dict:
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists: add an entry with append() instead of overwriting a registration")
    rec = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "question": question, "design": design,
           "selection": selection, "claim_rule": claim, **(extra or {}), "code_sha256": code_hashes(code), "later": []}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec


def append(path, entry: dict) -> dict:
    path = Path(path)
    rec = json.loads(path.read_text(encoding="utf-8"))
    rec.setdefault("later", []).append({"created": datetime.datetime.now().isoformat(timespec="seconds"), **entry})
    path.write_text(json.dumps(rec, indent=1, ensure_ascii=False), encoding="utf-8")
    return rec
