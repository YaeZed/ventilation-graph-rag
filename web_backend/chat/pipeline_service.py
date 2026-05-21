"""Thread-safe lazy holder for the VentilationRAGPipeline."""

import logging
import sys
from pathlib import Path
from threading import RLock

from django.conf import settings


logger = logging.getLogger(__name__)

RAG_SYSTEM_DIR = settings.REPO_ROOT / "agent" / "rag_system"
AGENT_DIR = settings.REPO_ROOT / "agent"
for path in (str(AGENT_DIR), str(RAG_SYSTEM_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ventilation_rag_pipeline import VentilationRAGPipeline


class PipelineService:
    def __init__(self):
        self._lock = RLock()
        self._pipeline = None
        self._initializing = False

    def get_pipeline(self) -> VentilationRAGPipeline:
        if self._pipeline is not None:
            return self._pipeline

        with self._lock:
            if self._pipeline is None:
                if self._initializing:
                    raise RuntimeError("Pipeline is already initializing")
                self._initializing = True
                try:
                    logger.info("初始化 VentilationRAGPipeline for Django API")
                    pipeline = VentilationRAGPipeline(
                        force_rebuild_index=settings.VENTILATION_PIPELINE_FORCE_REBUILD
                    )
                    pipeline.initialize()
                    self._pipeline = pipeline
                finally:
                    self._initializing = False
        return self._pipeline

    def close(self) -> None:
        with self._lock:
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None


_service = PipelineService()


def get_pipeline_service() -> PipelineService:
    return _service

