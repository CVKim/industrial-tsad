"""TSB-AD (Liu & Paparrizos, NeurIPS 2024 D&B): official file lists, series loading and published per-series results."""
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from ..utils.external import ensure_tsbad
from ..utils.paths import get_paths


@dataclass
class Series:
    name: str          # file name, e.g. 002_MSL_id_1_Sensor_tr_500_1st_900.csv
    dataset: str
    x: np.ndarray      # (T, C) float64
    label: np.ndarray  # (T,) int
    train_end: int     # the official training prefix [0, train_end)

    @property
    def sliding_window(self) -> int:
        return sliding_window(self.x)


def train_end_of(file_name: str) -> int:
    return int(file_name.split("_tr_")[1].split("_")[0])


def file_list(name: str) -> List[str]:
    """an official list name (TSB-AD-M-Eva.csv, TSB-AD-U-Tuning.csv, ...) or a path to a csv with a file_name column"""
    p = Path(name)
    if not p.is_absolute() and not p.exists():
        p = get_paths().file_lists / name
    return pd.read_csv(p)["file_name"].tolist()


def load(file_name: str, benchmark: str = "M") -> Series:
    df = pd.read_csv(get_paths().tsbad_data / f"TSB-AD-{benchmark}" / file_name)
    return Series(name=file_name, dataset=file_name.split("_")[1], x=df.iloc[:, :-1].values.astype(np.float64),
                  label=df.iloc[:, -1].values.astype(int), train_end=train_end_of(file_name))


def sliding_window(x: np.ndarray) -> int:
    """TSB-AD's metric window: the dominant period of the first channel (find_length_rank, rank 1)"""
    ensure_tsbad()
    from TSB_AD.utils.slidingWindows import find_length_rank
    return find_length_rank(np.asarray(x)[:, 0].reshape(-1, 1), rank=1)


_META = {"ts_len", "anomaly_len", "num_anomaly", "avg_anomaly_len", "anomaly_ratio", "point_anomaly", "seq_anomaly"}


def published_table(benchmark: str = "M") -> pd.DataFrame:
    """every published per-series VUS-PR (index = file name, one column per method): the TSB-AD merged table plus the
    leaderboard CSVs in <TSB-AD code>/benchmark_exp/leaderboard_results and <TSB-AD data>/leaderboard_results.
    Leaderboard formats differ per contributor: a ``VUS-PR`` column is named after the file (Multi_AxonAD.csv ->
    AxonAD), otherwise every non-metadata column is a method (Multi_TSPulse.csv -> "TSPulse (ZS)", "TSPulse (FT)")."""
    p = get_paths()
    tag, prefix = ("multi", "Multi_") if benchmark == "M" else ("uni", "Uni_")
    tab = pd.read_csv(p.published / "benchmark_eval_results" / f"{tag}_mergedTable_VUS-PR.csv").set_index("file")
    tab = tab[[c for c in tab.columns if c not in _META]]
    for folder in (p.published / "leaderboard_results", p.leaderboard_extra):
        for f in sorted(folder.glob(f"{prefix}*.csv")):
            t = pd.read_csv(f)
            t = t.set_index("file" if "file" in t.columns else ("filename" if "filename" in t.columns else t.columns[0]))
            if "VUS-PR" in t.columns:
                tab[f.stem[len(prefix):]] = t["VUS-PR"]
            else:
                for c in t.columns:
                    if c not in _META:
                        tab[c] = t[c]
    return tab


def published(method: str, benchmark: str = "M") -> pd.Series:
    """per-series VUS-PR of one published method (index = file name); names as in ``published_table``"""
    tab = published_table(benchmark)
    if method not in tab.columns:
        raise KeyError(f"no published per-series VUS-PR for {method!r} on TSB-AD-{benchmark}; available: {list(tab.columns)}")
    return tab[method].dropna()
