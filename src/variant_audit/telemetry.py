"""Observability — OpenTelemetry traces + metrics (your platform-eng home turf).

Every graph node becomes a span; you record latency, token spend, retrieval
scores, and correction-loop counts as metrics. Exported via OTLP to the
collector -> Tempo/Prometheus -> Grafana (infra/ wires the backends).

TODO(week-3): implement setup + the @traced_node decorator + metric instruments.
Degrade gracefully if no collector is running (don't crash the app).
"""

import functools


def setup_telemetry() -> None:
    """Idempotently init the OTel SDK (tracer + meter providers, OTLP exporters)."""
    raise NotImplementedError("TODO(week-3): configure TracerProvider/MeterProvider + OTLP")


def traced_node(name: str):
    """Decorator: wrap a graph node in a span + latency metric."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            # TODO(week-3): open a span named f"node.{name}", time it, record latency
            return fn(state, *args, **kwargs)

        return wrapper

    return decorator


# TODO(week-3): define metric instruments, e.g.
#   node_latency      (histogram)  per-node duration
#   llm_tokens        (counter)    tokens by direction + purpose
#   retrieval_score   (histogram)  top-1 similarity
#   correction_loops  (counter)    rewrites + regenerations
