"""industrial-tsad: time-series analysis and anomaly detection for industrial data.

Detectors share one interface (``BaseDetector.fit_score(x, train_end, seed)`` -> point scores), are built from pydantic
configs or YAML presets through a registry, and are evaluated with a reproducible TSB-AD protocol (threshold-free
metrics, paired bootstrap, dataset rule, pre-registration records).
"""
__version__ = "0.1.0"

from .core.models import DetectorRegistry, create_detector, list_detectors  # noqa: E402,F401
from .configs import load_preset  # noqa: E402,F401
