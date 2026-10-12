from . import DetectorRegistry


@DetectorRegistry.register("paano")
def _paano(config, device):
    from .paano.detector import PaAnoDetector
    return PaAnoDetector(config, device)


@DetectorRegistry.register("mscad")
def _mscad(config, device):
    from .mscad.detector import MSCADDetector
    return MSCADDetector(config, device)


@DetectorRegistry.register("axonad")
def _axonad(config, device):
    from .axonad.detector import AxonADDetector
    return AxonADDetector(config, device)


@DetectorRegistry.register("catch")
def _catch(config, device):
    from .catch.detector import CATCHDetector
    return CATCHDetector(config, device)


@DetectorRegistry.register("kanad")
def _kanad(config, device):
    from .kanad.detector import KANADDetector
    return KANADDetector(config, device)


@DetectorRegistry.register("timesnet")
def _timesnet(config, device):
    from .timesnet.detector import TimesNetDetector
    return TimesNetDetector(config, device)
