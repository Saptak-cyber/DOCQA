"""
Celery configuration with multi-queue support.
Separate queues for different task types for better resource management.
"""

from celery import Celery
from app.config import get_settings

settings = get_settings()

# Create Celery app
celery_app = Celery(
    "docqa",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
)

# Configure Celery
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    
    # Task routing
    task_routes={
        "app.tasks.ingestion.*": {"queue": "ingestion"},
        "app.tasks.maintenance.*": {"queue": "maintenance"},
    },
)

# Import tasks so Celery discovers them
celery_app.autodiscover_tasks(["app.tasks"])
