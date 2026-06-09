from .log_generator import LogGenerator
from .trace_processor import TraceProcessor


def __getattr__(name):
    if name == "ObjectCentricEventlog":
        from .eventlog_conversion import ObjectCentricEventlog

        return ObjectCentricEventlog
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "TraceProcessor",
    "LogGenerator",
    "ObjectCentricEventlog",
]
