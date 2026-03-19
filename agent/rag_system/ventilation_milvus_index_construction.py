import logging
import time
from typing import List, Dict, Any, Optional

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class VentilationMilvusIndexConstruction:
    """矿井通风安全规程 Milvus 索引构建模块"""

    def __init__(self, 
                 host: str = "localhost", 
                 port: int = 19530,
                 collection_name: str = "ventilation_knowledge",
                 dimension: int = 512,
                 model_name: str = "BAAI/bge-small-zh-v1.5"):
        self.host = host
        self.port = port
        self.collection_name = collection_name
        self.dimension = dimension
        self.model_name = model_name

        self.client = None
        self.embeddings = None
        self.collection_created = False

        self._setup_client()
        self._setup_embeddings()

    def _safe_truncate(self, text: str, max_length: int) -> str:
        if text is None:
            return ""
        return str(text)[:max_length]

    def _setup_client(self):
        """初始化Milvus客户端"""
        try:
            self.client = MilvusClient(uri=f"http://{self.host}:{self.port}")
            logger.info(f"已连接到Milvus服务器: {self.host}:{self.port}")
        except Exception as e:
            logger.error(f"连接Milvus失败: {e}")
            raise
    
    def _setup_embeddings(self):
        """初始化嵌入模型"""
        logger.info(f"正在初始化嵌入模型: {self.model_name}")
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("嵌入模型初始化完成")

    def _create_collection_schema(self) -> CollectionSchema:
        """
        创建专属于通风规程的集合模式
        去掉了烹饪相关字段，增加了 article_title, param_count, req_count
        """
        fields = [
            # 基础标识符
            FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=150, is_primary=True),
            # 向量字段
            FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
            # 核心文本内容
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=15000),
            
            # 通风领域元数据
            FieldSchema(name="node_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="article_name", dtype=DataType.VARCHAR, max_length=300),
            FieldSchema(name="article_title", dtype=DataType.VARCHAR, max_length=500),
            FieldSchema(name="node_type", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=100),
            
            # 统计元数据（用于过滤）
            FieldSchema(name="param_count", dtype=DataType.INT64),
            FieldSchema(name="req_count", dtype=DataType.INT64),
            
            # RAG 管理元数据
            FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=50),
            FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=150),
            FieldSchema(name="parent_id", dtype=DataType.VARCHAR, max_length=100)
        ]
        
        schema = CollectionSchema(
            fields=fields,
            description="矿井通风安全规程向量集合"
        )
        return schema

    def create_collection(self, force_recreate: bool = False) -> bool:
        try:
            if self.client.has_collection(self.collection_name):
                if force_recreate:
                    logger.info(f"删除已存在的集合: {self.collection_name}")
                    self.client.drop_collection(self.collection_name)
                else:
                    logger.info(f"集合 {self.collection_name} 已存在")
                    self.collection_created = True
                    return True
            
            schema = self._create_collection_schema()
            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                metric_type="COSINE",
                consistency_level="Strong"
            )
            logger.info(f"成功创建通风规程集合: {self.collection_name}")
            self.collection_created = True
            return True
        except Exception as e:
            logger.error(f"创建集合失败: {e}")
            return False

    def create_index(self, M: int = 8, ef_construction: int = 64) -> bool:
        """
        为向量字段创建索引，可以独立调用以测试不同参数的性能
        这里可以写论文用
        Args:
            M: HNSW 节点的邻居数量 (4-64)
            ef_construction: 构建索引时的搜索范围 (8-512)
        """
        try:
            logger.info(f"正在创建 HNSW 索引 (M={M}, efConstruction={ef_construction})...")
            
            # 释放现有索引（如果有）
            self.client.release_collection(self.collection_name)
            # 删除旧索引（如果存在）
            self.client.drop_index(self.collection_name, "vector")
            
            # 准备并创建新索引
            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                index_type="HNSW",
                metric_type="COSINE",
                params={"M": M, "efConstruction": ef_construction}
            )
            self.client.create_index(self.collection_name, index_params)
            
            # 重新加载集合
            self.client.load_collection(self.collection_name)
            logger.info("集合已加载到内存")

            self.collection_created = True
            logger.info("索引创建并加载完成")
    
            return True
        except Exception as e:
            logger.error(f"创建索引失败: {e}")
            return False

            
    def build_vector_index(self, chunks: List[Document]) -> bool:
        """构建向量索引并插入通风数据"""
        logger.info(f"开始构建通风规程向量索引，共 {len(chunks)} 块")
        try:
            # 1. 创建集合（如果schema不兼容则强制重新创建）
            if not self.create_collection(force_recreate=True):
                return False

            texts = [chunk.page_content for chunk in chunks]
            vectors = self.embeddings.embed_documents(texts)
            
            entities = []
            for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
                meta = chunk.metadata
                entity = {
                    "id": self._safe_truncate(meta.get("chunk_id", f"v_chunk_{i}"), 150),
                    "vector": vector,
                    "text": self._safe_truncate(chunk.page_content, 15000),

                    "node_id": self._safe_truncate(meta.get("node_id", ""), 100),
                    "article_name": self._safe_truncate(meta.get("article_name", ""), 300),
                    "article_title": self._safe_truncate(meta.get("article_title", ""), 500),
                    "node_type": self._safe_truncate(meta.get("node_type", ""), 100),
                    "category": self._safe_truncate(meta.get("category", "通用"), 100),

                    "param_count": int(meta.get("param_count", 0)),
                    "req_count": int(meta.get("req_count", 0)),

                    "doc_type": self._safe_truncate(meta.get("doc_type", ""), 50),
                    "chunk_id": self._safe_truncate(meta.get("chunk_id", f"v_chunk_{i}"), 150),
                    "parent_id": self._safe_truncate(meta.get("parent_id", ""), 100)
                }
                entities.append(entity)
            
            batch_size = 100
            for i in range(0, len(entities), batch_size):
                self.client.insert(
                    collection_name=self.collection_name,
                    data=entities[i:i + batch_size]
                )
            
            # 使用默认参数创建索引
            return self.create_index()
            
        except Exception as e:
            logger.error(f"构建向量索引失败: {e}")
            return False

   
    def similarity_search(self, query: str, k: int = 5, filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """执行相似度搜索，支持复杂过滤条件"""
        if not self.collection_created:
            self.load_collection()

        try:
            # 向量化查询问题
            query_vector = self.embeddings.embed_query(query)
            
            # 构建过滤表达式（支持 IN 操作）
            filter_expr = ""
            if filters:
                conds = []
                for key, val in filters.items():
                    if isinstance(val, str):
                        conds.append(f'{key} == "{val}"')
                    elif isinstance(val, (int, float)):
                        conds.append(f'{key} == {val}')
                    elif isinstance(val, list):
                        # 支持 IN 操作
                        if all(isinstance(v, str) for v in val):
                            val_str = '", "'.join(val)
                            conds.append(f'{key} in ["{val_str}"]')
                        else:
                            val_str = ', '.join(map(str, val))
                            conds.append(f'{key} in [{val_str}]')
                filter_expr = " and ".join(conds)

            # 执行搜索
            results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                limit=k,
                output_fields=["text", "node_id", "article_name", "article_title", 
                                "node_type", "category", "param_count", "req_count"],
                filter=filter_expr if filter_expr else None
            )
            
            formatted = []
            if results and len(results) > 0:
                for hit in results[0]:
                    formatted.append({
                        "id": hit["id"],
                        "score": hit["distance"],
                        "text": hit["entity"]["text"],
                        "metadata": hit["entity"] # 直接返回 entity 包含的所有字段
                    })
            return formatted
        except Exception as e:
            logger.error(f"搜索失败: {e}")
            return []

    def load_collection(self):
        if self.client.has_collection(self.collection_name):
            self.client.load_collection(self.collection_name)
            self.collection_created = True

    def has_collection(self) -> bool:
        return self.client.has_collection(self.collection_name)

    def close(self):
        pass

    def __del__(self):
        self.close()