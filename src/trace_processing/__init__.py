from .trace_processor import TraceProcessor
from .log_generator import LogGenerator
from .naive_utc import NaiveUTC, to_naive_utc, from_epoch_naive_utc


def __getattr__(name):
    if name == "ObjectCentricEventlog":
        from .eventlog_conversion import ObjectCentricEventlog
        return ObjectCentricEventlog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'TraceProcessor',
    'LogGenerator',
    'ObjectCentricEventlog',
    'NaiveUTC',
    'to_naive_utc',
    'from_epoch_naive_utc',
]
