import numpy as np
import pandas as pd

from industrial_tsad.evaluators.stats import DatasetRule, dataset_of, paired, select


def _series(values, datasets):
    idx = [f"{i:03d}_{d}_id_{i}_X_tr_100_1st_200.csv" for i, d in enumerate(datasets)]
    return pd.Series(values, index=idx)


def test_dataset_of():
    assert dataset_of("002_MSL_id_1_Sensor_tr_500_1st_900.csv") == "MSL"


def test_paired_detects_a_clear_gain():
    rng = np.random.default_rng(1)
    ds = ["A"] * 30 + ["B"] * 30
    b = _series(rng.uniform(0.2, 0.6, 60), ds)
    a = b + 0.05 + rng.normal(0, 0.01, 60)
    r = paired(a, b, rng=np.random.default_rng(0), n_boot=2000)
    assert r["n"] == 60 and abs(r["diff"] - 0.05) < 0.01
    assert r["series_CI95"][0] > 0 and r["claim"]
    assert r["datasets_up"] == 2 and r["datasets_down"] == 0


def test_paired_no_claim_for_noise_and_bonferroni_is_wider():
    rng = np.random.default_rng(2)
    ds = ["A", "B", "C"] * 20
    b = _series(rng.uniform(0, 1, 60), ds)
    a = b + rng.normal(0, 0.05, 60)
    r1 = paired(a, b, rng=np.random.default_rng(0), n_boot=2000)
    r3 = paired(a, b, rng=np.random.default_rng(0), n_boot=2000, looks=3)
    assert not r1["claim"]
    lo, hi = r3["series_CI_bonferroni"]
    assert lo <= r1["series_CI95"][0] and hi >= r1["series_CI95"][1]


def test_dataset_rule_and_select():
    ds = ["A", "B", "C", "D"]
    ref = _series([0.5, 0.5, 0.5, 0.5], ds)
    good = _series([0.6, 0.6, 0.5, 0.49], ds)   # 2 up, 0 down beyond 0.01
    bad = _series([0.7, 0.7, 0.7, 0.40], ds)    # one dataset drops by 0.10 > 0.05
    rule = DatasetRule()
    assert rule(good, ref)["pass"] and not rule(bad, ref)["pass"]
    s = select({"good": good, "bad": bad}, ref)
    assert s["selected"] == "good" and s["passed"]
    s2 = select({"bad": bad}, ref)
    assert s2["selected"] == "bad" and not s2["passed"]
