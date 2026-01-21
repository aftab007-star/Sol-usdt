import logging
from datetime import datetime, timezone

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger(__name__)
scheduler = BackgroundScheduler(timezone="UTC")

def test_job():
    timestamp = datetime.now(timezone.utc).isoformat()
    logger.info("[SCHEDULER TEST] Fired at %s", timestamp)

def start_scheduler():
    scheduler.add_job(
        test_job,
        "interval",
        minutes=1,
        replace_existing=True,
        id="test_job",
    )
    if not scheduler.running:
        scheduler.start()
        logger.info("Scheduler started")
