import time
import uuid
import random
import os
from collections import defaultdict

from logger import log_event

# ----------------------------
# Config
# ----------------------------
SEED = int(time.time() * 1000) % 1000000  # Based on current time
FAILURE_RATE = 0.2  # 20% failure rate

rng = random.Random(SEED)

# ----------------------------
# In-memory stores
# ----------------------------
jobs = {}
job_events = defaultdict(list)  # job_id -> event list


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

    return event


# ----------------------------
# Create Job (entry event)
# ----------------------------
def create_job(drink: str, correlation_id: str):
    job_id = str(uuid.uuid4())

    duration = rng.uniform(10, 15)
    random_val = rng.random()
    will_fail = random_val < FAILURE_RATE

    print(f"[DEBUG] Job {job_id[:8]}: random={random_val:.4f}, will_fail={will_fail}, FAILURE_RATE={FAILURE_RATE}")

    job = {
        "job_id": job_id,
        "drink": drink,
        "correlation_id": correlation_id,

        "status": "created",
        "created_at": time.time(),

        "duration": duration,
        "will_fail": will_fail,

        "started_at": None,
        "finished_at": None,
        "logged_finished": False,
    }

    jobs[job_id] = job

    # OCEL lifecycle start
    emit_event(job, "user_prompt")

    return job


# ----------------------------
# Status computation (pure function)
# ----------------------------
def compute_status(job):
    now = time.time()
    start = job["created_at"]
    duration = job["duration"]
    elapsed = now - start  # Calculate elapsed time
    
    if now < start + duration:
        return "brewing"

    return "failed" if job["will_fail"] else "ready"


# ----------------------------
# Read model (GET = side-effect controlled)
# ----------------------------
def get_job(job_id: str):
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

        job["finished_at"] = job["created_at"] + job["duration"]
        job["logged_finished"] = True

    return result


# ----------------------------
# Debug helper
# ----------------------------
def get_job_events(job_id: str):
    return job_events.get(job_id, [])