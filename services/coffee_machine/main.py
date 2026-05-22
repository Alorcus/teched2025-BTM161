from contextlib import asynccontextmanager
import logging
import os
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from .state import create_job, get_job
from .worker import run_worker

# Configure the coffee_shop.coffee_machine logger hierarchy to match the main program's format.
# When run standalone (uvicorn), this ensures logs are visible; when imported from the main
# program, the parent coffee_shop logger's handler takes precedence.
_coffee_machine_logger = logging.getLogger("coffee_shop.coffee_machine")
_coffee_machine_logger.setLevel(getattr(logging, os.environ.get("COFFEE_MACHINE_LOG_LEVEL", "INFO")))
if not _coffee_machine_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s — %(message)s"))
    _coffee_machine_logger.addHandler(_handler)

logger = _coffee_machine_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    thread = threading.Thread(target=run_worker, daemon=True)
    thread.start()
    logger.info("OCEL background worker started")
    yield


app = FastAPI(lifespan=lifespan)


# -------- Request schema --------
class BrewRequest(BaseModel):
    drink: str
    correlation_id: str


# -------- Endpoints --------

@app.post("/brew")
def brew(req: BrewRequest):
    logger.info("Brew requested: drink=%s, correlation_id=%s", req.drink, req.correlation_id)
    job = create_job(req.drink, req.correlation_id)

    return {
        "job_id": job["job_id"],
        "eta_seconds": job["duration"]
    }


@app.get("/jobs/{job_id}")
def job_status(job_id: str):
    job = get_job(job_id)

    if not job:
        logger.warning("Job status requested for unknown job: %s", job_id[:8])
        return JSONResponse({"error": "job not found"}, status_code=404)

    logger.debug("Job status: %s -> %s", job_id[:8], job["status"])
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"]
    }


@app.get("/healthz")
def health():
    return {"status": "ok"}
