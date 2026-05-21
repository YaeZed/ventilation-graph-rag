"""
Shared database connection manager for the ventilation Graph RAG system.

Neo4j and Milvus clients are thread-safe and maintain their own connection
pools, so the web layer and RAG modules should reuse a single manager instead
of creating duplicate drivers in every module.
"""

import logging
import os
from threading import RLock
from typing import Any, Optional

from neo4j import GraphDatabase
from pymilvus import MilvusClient

logger = logging.getLogger(__name__)


class ConnectionManager:
    """Singleton holder for Neo4j and Milvus connections."""

    _instance: Optional["ConnectionManager"] = None
    _instance_lock = RLock()

    def __new__(cls, *args, **kwargs):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, config: Optional[Any] = None):
        if getattr(self, "_initialized", False):
            if config is not None:
                self.configure(config)
            return

        self._lock = RLock()
        self._neo4j_driver = None
        self._milvus_client = None
        self._initialized = True

        self.neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.neo4j_user = os.getenv("NEO4J_USER", "neo4j")
        self.neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
        self.neo4j_database = os.getenv("NEO4J_DATABASE", "neo4j")

        self.milvus_host = os.getenv("MILVUS_HOST", "localhost")
        self.milvus_port = int(os.getenv("MILVUS_PORT", "19530"))
        self.milvus_uri = os.getenv("MILVUS_URI", f"http://{self.milvus_host}:{self.milvus_port}")

        if config is not None:
            self.configure(config)

    @classmethod
    def get_instance(cls, config: Optional[Any] = None) -> "ConnectionManager":
        """Return the shared manager instance and optionally refresh config."""
        return cls(config)

    def configure(self, config: Any) -> None:
        """Load connection settings from a config object without opening sockets."""
        with self._lock:
            neo4j_uri = getattr(config, "neo4j_uri", self.neo4j_uri)
            neo4j_user = getattr(config, "neo4j_user", self.neo4j_user)
            neo4j_password = getattr(config, "neo4j_password", self.neo4j_password)
            neo4j_database = getattr(config, "neo4j_database", self.neo4j_database)

            milvus_host = getattr(config, "milvus_host", self.milvus_host)
            milvus_port = int(getattr(config, "milvus_port", self.milvus_port))
            default_uri = f"http://{milvus_host}:{milvus_port}"
            milvus_uri = getattr(config, "milvus_uri", default_uri)

            neo4j_changed = (
                neo4j_uri,
                neo4j_user,
                neo4j_password,
                neo4j_database,
            ) != (
                self.neo4j_uri,
                self.neo4j_user,
                self.neo4j_password,
                self.neo4j_database,
            )
            milvus_changed = milvus_uri != self.milvus_uri

            if neo4j_changed and self._neo4j_driver is not None:
                self.close_neo4j()
            if milvus_changed and self._milvus_client is not None:
                self.close_milvus()

            self.neo4j_uri = neo4j_uri
            self.neo4j_user = neo4j_user
            self.neo4j_password = neo4j_password
            self.neo4j_database = neo4j_database

            self.milvus_host = milvus_host
            self.milvus_port = milvus_port
            self.milvus_uri = milvus_uri

    def get_neo4j_driver(self, verify: bool = False):
        """Create or return the shared Neo4j driver."""
        with self._lock:
            if self._neo4j_driver is None:
                self._neo4j_driver = GraphDatabase.driver(
                    self.neo4j_uri,
                    auth=(self.neo4j_user, self.neo4j_password),
                )
                logger.info("已创建共享 Neo4j driver: %s", self.neo4j_uri)

            if verify:
                with self._neo4j_driver.session(database=self.neo4j_database) as session:
                    session.run("RETURN 1 AS ok").single()
                logger.info("共享 Neo4j 连接测试成功")

            return self._neo4j_driver

    def get_milvus_client(self):
        """Create or return the shared Milvus client."""
        with self._lock:
            if self._milvus_client is None:
                self._milvus_client = MilvusClient(uri=self.milvus_uri)
                logger.info("已创建共享 Milvus client: %s", self.milvus_uri)
            return self._milvus_client

    def close_neo4j(self) -> None:
        with self._lock:
            if self._neo4j_driver is not None:
                self._neo4j_driver.close()
                self._neo4j_driver = None
                logger.info("共享 Neo4j driver 已关闭")

    def close_milvus(self) -> None:
        with self._lock:
            if self._milvus_client is not None:
                close = getattr(self._milvus_client, "close", None)
                if callable(close):
                    close()
                self._milvus_client = None
                logger.info("共享 Milvus client 已关闭")

    def close_all(self) -> None:
        """Release every managed connection."""
        self.close_neo4j()
        self.close_milvus()
