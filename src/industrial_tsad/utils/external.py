"""Import helpers for code that stays in its own (local) clone and is never vendored into this repository."""
import sys
import types

from .paths import get_paths


def ensure_tsbad():
    """put the AxonAD clone (it ships the TSB_AD package) on sys.path; returns its path"""
    root = get_paths().tsbad_code
    if not (root / "TSB_AD").exists():
        raise FileNotFoundError(f"TSB_AD not found under {root} (set ITSAD_TSBAD_CODE to a clone of "
                                "github.com/iis-esslingen/AxonAD)")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if "torchinfo" not in sys.modules:  # TSB-AD's TimesNet imports torchinfo only for a summary that is never called
        sys.modules["torchinfo"] = types.ModuleType("torchinfo")
    return root
