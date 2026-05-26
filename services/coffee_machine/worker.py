import time
import logging
from .state import jobs, get_job

logger = logging.getLogger("coffee_shop.coffee_machine.worker")

POLL_INTERVAL = 1.0


def run_worker():
    """
    Background OCEL simulation engine.
    Continuously advances job states and triggers event emissions.
    """

    while True:
        for job_id in list(jobs.keys()):
            if jobs[job_id].get("logged_finished"):
                continue
            try:
                get_job(job_id)
            except Exception as e:
                logger.error("Error processing job %s: %s", job_id[:8], e)

        time.sleep(POLL_INTERVAL)
