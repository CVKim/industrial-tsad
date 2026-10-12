"""Published per-series results: the merged table plus leaderboard CSVs in both contributor formats (no data needed)."""
import pandas as pd
import pytest

from industrial_tsad.datasets import tsbad
from industrial_tsad.utils.paths import Paths


def _write(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def test_published_table_reads_both_folders_and_formats(tmp_path, monkeypatch):
    files = ["001_GHL_id_1_Sensor_tr_100_1st_200.csv", "002_MSL_id_1_Sensor_tr_100_1st_150.csv"]
    code, data = tmp_path / "code", tmp_path / "data"
    bench = code / "benchmark_exp"
    _write(bench / "benchmark_eval_results" / "multi_mergedTable_VUS-PR.csv",
           pd.DataFrame({"file": files, "IForest": [0.1, 0.2], "ts_len": [300, 300]}))
    _write(bench / "leaderboard_results" / "Multi_TSPulse.csv",
           pd.DataFrame({"file": files, "TSPulse (ZS)": [0.3, 0.4], "TSPulse (FT)": [0.5, 0.6], "ts_len": [300, 300]}))
    _write(data / "leaderboard_results" / "Multi_AxonAD.csv", pd.DataFrame({"filename": files[:1], "VUS-PR": [0.7]}))
    monkeypatch.setattr(tsbad, "get_paths", lambda: Paths(data, code, tmp_path, tmp_path, tmp_path))

    tab = tsbad.published_table("M")
    assert list(tab.columns) == ["IForest", "TSPulse (ZS)", "TSPulse (FT)", "AxonAD"]
    assert tsbad.published("TSPulse (FT)", "M").tolist() == [0.5, 0.6]
    axon = tsbad.published("AxonAD", "M")
    assert axon.index.tolist() == files[:1] and axon.iloc[0] == 0.7
    with pytest.raises(KeyError):
        tsbad.published("NoSuchMethod", "M")
