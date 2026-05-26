import time
import uuid
import random
import os
import logging
from collections import defaultdict

from .logger import log_event

logger = logging.getLogger("coffee_shop.coffee_machine.state")

# ----------------------------
# Config
# ----------------------------
SEED = int(os.environ.get("COFFEE_MACHINE_SEED", "100"))
FAILURE_RATE = 0.2  # 20% failure rate

rng = random.Random(SEED)


def _generate_outcome() -> str:
    """Pre-roll one brew outcome consuming the same RNG calls as create_job."""
    rng.uniform(1, 3)  # duration
    return "FAIL" if rng.random() < FAILURE_RATE else "SUCC"


# ----------------------------
# In-memory stores
# ----------------------------
jobs = {}
job_events = defaultdict(list)  # job_id -> event list
machine_dirty = False
outcome_queue: list[str] = [_generate_outcome() for _ in range(4)]


# ----------------------------
# OCEL Event Emitter
# ----------------------------
def emit_event(job, activity: str, duration: float = None):
    timestamp = time.time()

    event = {
        "case_id": job["correlation_id"],   # OCEL case (process instance)
        "activity": activity,               # event type
        "timestamp": timestamp,             # OCEL time
        "duration": duration,

        # optional object attributes (for OCEL enrichment)
        "job_id": job["job_id"],
        "drink": job["drink"],
    }

    log_event(**event)

    job_events[job["job_id"]].append(event)
    logger.debug("Event emitted: %s for job %s (case %s)", activity, job["job_id"][:8], job["correlation_id"])

    return event


# ----------------------------
# Create Job (entry event)
# ----------------------------
def create_job(drink: str, correlation_id: str):
    global machine_dirty
    job_id = str(uuid.uuid4())

    outcome = outcome_queue.pop(0)
    outcome_queue.append(_generate_outcome())
    will_fail = outcome == "FAIL"
    duration = rng.uniform(1, 3)

    logger.debug("Job %s: outcome=%s, will_fail=%s", job_id[:8], outcome, will_fail)

    job = {
        "job_id": job_id,
        "drink": drink,
        "correlation_id": correlation_id,

        "status": "created",
        "created_at": time.time(),

        "duration": duration,
        "will_fail": will_fail,
        "contaminated": machine_dirty and not will_fail,

        "started_at": None,
        "finished_at": None,
        "logged_finished": False,
    }

    jobs[job_id] = job

    # OCEL lifecycle start
    emit_event(job, "user_prompt")
    logger.info("Job created: %s (drink=%s, duration=%.1fs, contaminated=%s)", job_id[:8], drink, duration, job["contaminated"])

    return job


# ----------------------------
# Status computation (pure function)
# ----------------------------
def compute_status(job):
    now = time.time()
    start = job["created_at"]
    duration = job["duration"]

    if now < start + duration:
        return "brewing"

    return "failed" if job["will_fail"] else "ready"


# ----------------------------
# Read model (GET = side-effect controlled)
# ----------------------------
def get_job(job_id: str):
    global machine_dirty
    job = jobs.get(job_id)
    if not job:
        return None

    status = compute_status(job)
    result = job.copy()
    result["status"] = status

    events = job_events[job_id]
    last_activity = events[-1]["activity"] if events else None

    # ----------------------------
    # Brewing transition event
    # ----------------------------
    if status == "brewing" and last_activity != "process_order":
        emit_event(job, "process_order", duration=job["duration"])

    # ----------------------------
    # Completion event
    # ----------------------------
    if status in ["ready", "failed"] and not job["logged_finished"]:
        emit_event(
            job,
            "brew_completed" if status == "ready" else "brew_failed",
            duration=job["duration"]
        )

        if status == "failed":
            machine_dirty = True
            logger.info("Machine marked dirty after job %s failure", job_id[:8])

        job["finished_at"] = job["created_at"] + job["duration"]
        job["logged_finished"] = True
        logger.info("Job %s finished: %s", job_id[:8], status)

    return result


# ----------------------------
# Debug helper
# ----------------------------
def get_job_events(job_id: str):
    return job_events.get(job_id, [])


def get_queue() -> list[str]:
    return list(outcome_queue)


# ----------------------------
# Machine cleaning
# ----------------------------
def clean_machine():
    global machine_dirty
    if machine_dirty:
        machine_dirty = False
        logger.info("Machine cleaned")
        return {"status": "cleaned"}
    return {"status": "already_clean"}


def reseed(new_seed: int):
    global rng, outcome_queue
    rng = random.Random(new_seed)
    outcome_queue = [_generate_outcome() for _ in range(4)]
    logger.info("Machine RNG reseeded with %d", new_seed)
    return {"status": "reseeded", "seed": new_seed}
