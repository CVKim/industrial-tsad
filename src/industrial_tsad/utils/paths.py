"""Locations of data, external code and results.

Resolution order for every entry: environment variable > `paths.local.json` at the repository root (machine-specific,
git-ignored) > repository-relative default.

ITSAD_TSBAD_DATA     folder holding TSB-AD-M/ and TSB-AD-U/ (csv series)          default <repo>/data/TSB-AD
                     and optionally leaderboard_results/ (per-series results missing from the clone below)
ITSAD_TSBAD_CODE     clone of github.com/iis-esslingen/AxonAD (ships the TSB_AD package, the official file lists under
                     Datasets/File_List and published per-series results under benchmark_exp/)
                                                                                default <repo>/external/AxonAD
ITSAD_CATCH_CODE     clone of github.com/decisionintelligence/CATCH               default <repo>/external/CATCH
ITSAD_KANAD_CODE     clone of github.com/issaccv/KAN-AD                           default <repo>/external/KAN-AD
ITSAD_RESULTS        where runs are written                                       default <repo>/results
"""
import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
LOCAL_FILE = REPO / "paths.local.json"

_DEFAULTS = {
    "ITSAD_TSBAD_DATA": REPO / "data" / "TSB-AD",
    "ITSAD_TSBAD_CODE": REPO / "external" / "AxonAD",
    "ITSAD_CATCH_CODE": REPO / "external" / "CATCH",
    "ITSAD_KANAD_CODE": REPO / "external" / "KAN-AD",
    "ITSAD_RESULTS": REPO / "results",
}


@dataclass(frozen=True)
class Paths:
    tsbad_data: Path
    tsbad_code: Path
    catch_code: Path
    kanad_code: Path
    results: Path

    @property
    def file_lists(self) -> Path:
        return self.tsbad_code / "Datasets" / "File_List"

    @property
    def published(self) -> Path:
        return self.tsbad_code / "benchmark_exp"

    @property
    def leaderboard_extra(self) -> Path:
        """leaderboard CSVs that the TSB-AD clone does not ship (e.g. Multi_AxonAD.csv from the upstream TSB-AD repo)"""
        return self.tsbad_data / "leaderboard_results"


def get_paths() -> Paths:
    local = json.loads(LOCAL_FILE.read_text(encoding="utf-8")) if LOCAL_FILE.exists() else {}
    v = {k: Path(os.environ.get(k) or local.get(k) or d) for k, d in _DEFAULTS.items()}
    return Paths(v["ITSAD_TSBAD_DATA"], v["ITSAD_TSBAD_CODE"], v["ITSAD_CATCH_CODE"], v["ITSAD_KANAD_CODE"],
                 v["ITSAD_RESULTS"])
