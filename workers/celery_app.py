from celery import Celery
from celery.schedules import crontab

from app.config import get_settings

settings = get_settings()

celery_app = Celery(
    "sentinel",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=["workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    # Queue routing
    task_routes={
        "workers.tasks.evaluate_response": {"queue": "evaluation"},
        "workers.tasks.run_drift_detection": {"queue": "alerts"},
        "workers.tasks.check_rollback": {"queue": "alerts"},
    },
    # Scheduled tasks
    beat_schedule={
        "nightly-drift-detection": {
            "task": "workers.tasks.run_drift_detection",
            "schedule": crontab(hour=2, minute=0),  # 02:00 UTC daily
        },
        "hourly-rollback-check": {
            "task": "workers.tasks.check_rollback",
            "schedule": crontab(minute=0),  # Every hour
        },
    },
)
