import asyncio
import datetime
import logging

logger = logging.getLogger(__name__)

UTC = datetime.timezone.utc

def _now():
    return datetime.datetime.now(tz=UTC)

def seconds_until(target: datetime.datetime) -> float:
    return max((target - _now()).total_seconds(), 0)

def next_4h_close():
    now = _now()
    hour = ((now.hour // 4) + 1) * 4
    if hour >= 24:
        return now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
    return now.replace(hour=hour, minute=0, second=0, microsecond=0)

def next_daily_close():
    now = _now()
    return (now + datetime.timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)

def next_weekly_close():
    now = _now()
    days_ahead = (7 - now.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7
    return (now + datetime.timedelta(days=days_ahead)).replace(hour=0, minute=0, second=0, microsecond=0)

def next_monthly_close():
    now = _now()
    year = now.year + (now.month // 12)
    month = (now.month % 12) + 1
    return datetime.datetime(year, month, 1, tzinfo=UTC)

async def run_on_close(name: str, next_close_fn, callback):
    while True:
        next_run = next_close_fn()
        wait = seconds_until(next_run)
        logger.info(f"[Scheduler] {name} scheduled at {next_run.isoformat()}")
        await asyncio.sleep(wait)
        try:
            logger.info(f"[Scheduler] Executing {name}")
            await callback()
        except Exception:
            logger.exception(f"[Scheduler] {name} failed")
