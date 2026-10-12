import pytest
from pydantic import ValidationError

from industrial_tsad.configs import PRESETS, detector_config, load_preset
from industrial_tsad.core.models import list_detectors


def test_every_preset_validates():
    names = sorted(p.stem for p in PRESETS.glob("*.yaml"))
    assert {"paano_official", "paano_final", "mscad", "axonad", "catch", "kanad", "timesnet"} <= set(names)
    for n in names:
        cfg = load_preset(n)
        assert cfg.type in list_detectors()


def test_final_preset_values():
    c = load_preset("paano_final")
    assert (c.patch, c.iters, c.ci, c.ci_amp, c.point, c.metric) == (96, 100, 1.0, 1.0, 0.5, "cosine")
    assert c.seed.base == 2027 and c.seed.per_series


def test_axonad_seed_policy_is_fixed_2024():
    c = load_preset("axonad")
    assert c.seed.base == 2024 and not c.seed.per_series and c.adapter is None


def test_unknown_keys_are_rejected():
    with pytest.raises(ValidationError):
        detector_config({"type": "paano", "patchh": 64})
    with pytest.raises(ValidationError):
        detector_config({"type": "nope"})
