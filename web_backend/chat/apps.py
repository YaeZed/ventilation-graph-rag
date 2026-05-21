"""Django app configuration for chat APIs."""

from django.apps import AppConfig


class ChatConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "chat"

    def ready(self):
        # Pipeline initialization is intentionally lazy. Creating Neo4j/Milvus
        # sockets during app loading makes management commands brittle, while
        # lazy initialization still keeps a singleton pipeline per process.
        from .pipeline_service import get_pipeline_service

        get_pipeline_service()

