import time
import uuid
import random
import os
import logging
from collections import defaultdict

from .logger import log_event

logger = logging.getLogger("coffee_shop.coffee_machine.state")

SEED = int(os.environ.get("COFFEE_MACHINE_SEED", "100"))
FAILURE_RATE = 0.2

rng = random.Random(SEED)


def _generate_outcome() -> str:
    """Pre-roll one brew outcome consuming the same RNG calls as create_job."""
    rng.uniform(1, 3)  # duration
    return "FAIL" if rng.random() < FAILURE_RATE else "SUCC"


jobs = {}
job_events = defaultdict(list)
machine_dirty = False
outcome_queue: list[str] = [_generate_outcome() for _ in range(4)]


def emit_event(job, activity: str, duration: float = None):
    timestamp = time.time()

    event = {
        "case_id": job["correlation_id"],
        "activity": activity,
        "timestamp": timestamp,
        "duration": duration,

        "job_id": job["job_id"],
        "drink": job["drink"],
    }

    log_event(**event)

    job_events[job["job_id"]].append(event)
    logger.debug("Event emitted: %s for job %s (case %s)", activity, job["job_id"][:8], job["correlation_id"])

    return event


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

    emit_event(job, "job_created")
    logger.info("Job created: %s (drink=%s, duration=%.1fs, contaminated=%s)", job_id[:8], drink, duration, job["contaminated"])

    return job


def compute_status(job):
    now = time.time()
    start = job["created_at"]
    duration = job["duration"]

    if now < start + duration:
        return "brewing"

    return "failed" if job["will_fail"] else "ready"


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

    if status == "brewing" and last_activity != "process_order":
        emit_event(job, "process_order", duration=job["duration"])

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


def get_job_events(job_id: str):
    return job_events.get(job_id, [])


def get_queue() -> list[str]:
    return list(outcome_queue)


def clean_machine(correlation_id: str):
    global machine_dirty

    log_event(
        case_id=correlation_id,
        activity="clean_machine",
        timestamp=time.time(),
        duration=0.0,
    )

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
