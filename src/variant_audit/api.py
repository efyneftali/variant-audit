"""FastAPI service. Run: uvicorn variant_audit.api:app --reload --port 8000

Exposes the agent over HTTP so it's a real service (and so the eval harness /
other clients can hit it). Keep the endpoint thin — it just calls graph.ask().

TODO(day-3): stand up the skeleton (/healthz + a stub /classify).
TODO(day-6+): wire /classify to the real graph; add telemetry instrumentation.
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="variant-audit")


class ClassifyRequest(BaseModel):
    variant: str


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.post("/classify")
def classify_endpoint(req: ClassifyRequest) -> dict:
    """Run a variant through the agent and return classification + evidence + flags."""
    raise NotImplementedError("TODO(day-6): call graph.ask(req.variant) and shape the response")
