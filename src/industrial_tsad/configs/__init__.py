from pathlib import Path

import yaml
from pydantic import TypeAdapter

from .base import BaseConfig  # noqa: F401
from .models import (AxonADConfig, CATCHConfig, DetectorConfig, KANADConfig, ModeAdapterConfig, MSCADConfig,  # noqa: F401
                     PaAnoConfig, SeedPolicy, TimesNetConfig)
from .run import BenchmarkRunConfig  # noqa: F401

PRESETS = Path(__file__).resolve().parents[1] / "presets"
_ADAPTER = TypeAdapter(DetectorConfig)


def detector_config(d: dict):
    """dict (e.g. parsed YAML) -> the matching detector config, validated"""
    return _ADAPTER.validate_python(d)


def load_preset(name_or_path: str):
    """a preset name from industrial_tsad/presets (e.g. 'paano_final') or a YAML path -> detector config"""
    p = Path(name_or_path)
    if not p.suffix:
        p = PRESETS / f"{name_or_path}.yaml"
    return detector_config(yaml.safe_load(p.read_text(encoding="utf-8")))
