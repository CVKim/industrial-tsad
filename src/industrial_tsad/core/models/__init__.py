"""Detector registry (factory registry): ``@DetectorRegistry.register("name")`` on a builder
``(config, device) -> BaseDetector``; builders import their model code lazily so optional dependencies stay optional."""
from typing import Callable, Dict, Optional

from .base import BaseDetector  # noqa: F401


class DetectorRegistry:
    _builders: Dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str):
        def deco(fn):
            if name in cls._builders:
                raise KeyError(f"detector {name!r} registered twice")
            cls._builders[name] = fn
            return fn
        return deco

    @classmethod
    def build(cls, config, device: Optional[str] = None) -> BaseDetector:
        from . import factories  # noqa: F401  (populates the registry)
        if config.type not in cls._builders:
            raise KeyError(f"unknown detector type {config.type!r}; known: {sorted(cls._builders)}")
        return cls._builders[config.type](config, device)

    @classmethod
    def names(cls):
        from . import factories  # noqa: F401
        return sorted(cls._builders)


def create_detector(config, device: Optional[str] = None) -> BaseDetector:
    """config object, dict or preset name -> detector"""
    if isinstance(config, str):
        from ...configs import load_preset
        config = load_preset(config)
    elif isinstance(config, dict):
        from ...configs import detector_config
        config = detector_config(config)
    return DetectorRegistry.build(config, device)


def list_detectors():
    return DetectorRegistry.names()
