"""vector_store.py —— 存储层：Milvus / MongoDB / Elasticsearch / MySQL

四库分工：
    Milvus   语义召回（dense 由百炼 embedding 产出，sparse 由内置 BM25 Function 服务端生成）
    ES       关键词召回（BM25）
    MongoDB  原始档案库，整篇原文，doc_hash 关联
    MySQL    登记表，入库前判重 —— 让"重复处理一份简历"的代价降为一次主键查询

判重三道闸：MySQL 登记表（逻辑闸）→ Mongo 唯一索引（存储闸）→ Milvus/ES 幂等写入（写入闸）。
文件名按 PEP 8 用小写 vector_store.py，类名保持 VectorStore。
"""
from __future__ import annotations

import os

import pymysql
import requests
from dotenv import find_dotenv, load_dotenv
from elasticsearch import Elasticsearch
from openai import OpenAI
from pymilvus import (
    AnnSearchRequest,
    DataType,
    Function,
    FunctionType,
    MilvusClient,
    RRFRanker,
)
from pymongo import MongoClient

from document_processor import Chunk

load_dotenv(find_dotenv())

# ---------------------------------------------------------------- 四库命名
MILVUS_COLLECTION = "resume_chunks"
MONGO_COLLECTION = "resumes"
ES_INDEX = "resume_chunks"
MYSQL_TABLE = "processed_docs"

MONGO_DATABASE = os.getenv("MONGO_DATABASE", "smartrecruit")
MYSQL_DATABASE = os.getenv("MYSQL_DATABASE", "smartrecruit")
DENSE_DIM = int(os.getenv("EMBEDDING_DIM", "1024"))   # 必须与 .env 一致，别写死
EMBED_BATCH = 10     # 每批向量化条数；一次塞几百条会被网关拒绝
EMBED_MODEL = os.getenv("EMBEDDING_MODEL")
ANALYZER_PARAMS = {"type": "chinese"}   # pymilvus 内置中文分词，零配置

# MySQL 登记表：doc_hash 主键 = sha256 指纹（CHAR(64)，讲义 md5 是 32 别照抄），
# 与三库的 doc_hash 同源；char/chunk_count 为统计冗余；updated_at 由 ON UPDATE 自动维护。
MYSQL_DDL = f"""
CREATE TABLE IF NOT EXISTS `{MYSQL_DATABASE}`.`{MYSQL_TABLE}` (
    doc_hash    CHAR(64)     NOT NULL COMMENT 'sha256 内容指纹（64 位）',
    file_path   VARCHAR(512) NOT NULL COMMENT '源文件路径',
    char_count  INT          NOT NULL DEFAULT 0 COMMENT '清洗后正文字数',
    chunk_count INT          NOT NULL DEFAULT 0 COMMENT '切出的子块数',
    created_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT '首次登记时间',
    updated_at  DATETIME     NOT NULL DEFAULT CURRENT_TIMESTAMP
                             ON UPDATE CURRENT_TIMESTAMP COMMENT '最近处理时间',
    PRIMARY KEY (doc_hash),
    KEY idx_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
  COMMENT='简历去重登记表：出现过 = 已处理过'
"""

# 中文分词用内置 cjk（双字切 + 英文整词），零插件零重启；
# 对比默认 standard 的单字切分召回质量高得多，跨词噪音由 BM25 的 IDF 自然压低。
ES_ANALYZER = "cjk"

# ES mapping 必须显式定义，否则动态推断会把 content 推成 standard（中文退化成单字检索）。
ES_MAPPING = {
    "properties": {
        "content": {"type": "text", "analyzer": ES_ANALYZER},
        "metadata": {
            "properties": {
                "doc_hash": {"type": "keyword"},
                "parent_id": {"type": "keyword"},
                "file_path": {"type": "keyword"},
                "timestamp": {"type": "keyword"},
                # index=False：只进 _source 供回带，不建倒排索引（父块只用于回喂 LLM）
                "parent_content": {"type": "text", "index": False},
            }
        },
    }
}

# 检索/查询时需要带出的标量字段（parent_content 是回喂 LLM 的关键）
SCALAR_FIELDS = [
    "text",
    "parent_id",
    "parent_content",
    "doc_hash",
    "file_path",
    "timestamp",
]

# ---------------------------------------------------------------- 混合检索参数
#
# RRF（倒数排名融合）k=60：score = Σ 1/(k+rank)，只用排名不用分数，任何量纲都能融合
RRF_K = 60

# 融合前的召回池，要比最终 top_k 宽，否则精排没有重排余地
RECALL_K = 10

RERANK_MODEL = os.getenv("RERANK_MODEL")
RERANK_TOP_N = 5


def resolve_rerank_url() -> str:
    """rerank 的原生端点。

    实测：百炼 rerank 不走 OpenAI 兼容协议（/rerank 404），只有 DashScope
    原生路径可用。.env 给的是同一 host 的兼容端点，去掉 /compatible-mode/v1
    后接原生路径即可；显式配 RERANK_URL 可覆盖（换供应商不改代码）。
    """
    explicit = os.getenv("RERANK_URL")
    if explicit:
        return explicit
    base = (os.getenv("RERANK_BASE_URL") or "").rstrip("/")
    host = base.removesuffix("/compatible-mode/v1").rstrip("/")
    return f"{host}/api/v1/services/rerank/text-rerank/text-rerank"


RERANK_URL = resolve_rerank_url()

# ---------------------------------------------------------------------------
# Milvus 字段：id(PK) / text(enable_analyzer) / dense_vector(1024) /
#   sparse_vector(不填值，BM25 Function 生成) / parent_id / parent_content /
#   doc_hash(64，讲义 md5 是 32) / file_path / timestamp
# 索引：dense = FLAT + IP；sparse = SPARSE_INVERTED_INDEX + BM25
# ---------------------------------------------------------------------------


def rrf_fuse(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """多路召回按 RRF 合并成一路。

    不能加权求和：Milvus IP 是 0~1 余弦，ES BM25 无上界（实测 3.89），
    量纲不可比（讲义的 WeightedRanker 是库内融合、内部先归一化，与跨库相加不是一回事）。
    RRF 只用排名：score = Σ 1/(k+rank)。结果带 routes（来源路）与 rrf 分，便于调试。

    参数 rankings：每路是已按相关性降序的 hit 列表。
    """
    scores: dict[str, float] = {}
    best: dict[str, dict] = {}
    routes: dict[str, list[str]] = {}

    for route_idx, hits in enumerate(rankings):
        route_name = ROUTE_NAMES[route_idx] if route_idx < len(ROUTE_NAMES) else f"route{route_idx}"
        for rank, hit in enumerate(hits, start=1):
            key = hit["id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            best.setdefault(key, hit)          # 保留首次出现时的完整字段
            routes.setdefault(key, []).append(route_name)

    fused = []
    for key, score in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        row = dict(best[key])
        row["rrf"] = score
        row["routes"] = routes[key]
        fused.append(row)
    return fused


# 与 rrf_fuse 的 rankings 参数一一对应，仅用于标注来源
ROUTE_NAMES = ["milvus_hybrid", "es_bm25"]


class VectorStore:
    """四库读写层。构造时连上四个库并幂等地把集合/索引/表准备好。"""

    def __init__(
        self,
        collection_name: str = MILVUS_COLLECTION,
        dim: int = DENSE_DIM,
    ) -> None:
        self.collection_name = collection_name
        self.dim = dim
        self.client: MilvusClient | None = None
        self.emb_client: OpenAI | None = None
        self.mongo_client: MongoClient | None = None
        self.mongo_col = None
        self.es_client: Elasticsearch | None = None
        self.mysql_conn: pymysql.connections.Connection | None = None

        # 构造完即可用：先连上，再确保四库的容器都存在
        self._connect()
        self._ensure_collection()
        self._ensure_mongo()
        self._ensure_es()
        self._ensure_mysql()

    # ------------------------------------------------------------ 连接
    def _connect(self) -> None:
        """建立 Milvus / embedding / MongoDB / Elasticsearch / MySQL 五个客户端。"""
        host = os.getenv("MILVUS_HOST", "localhost")
        port = os.getenv("MILVUS_PORT", "19530")
        # uri 必须带 scheme，不带 http:// 会直接连不上
        self.client = MilvusClient(uri=f"http://{host}:{port}")

        # 百炼的 embedding 走 OpenAI 兼容协议，用裸 SDK 就够，不必套 langchain 包装类
        self.emb_client = OpenAI(
            api_key=os.getenv("EMBEDDING_API_KEY"),
            base_url=os.getenv("EMBEDDING_BASE_URL"),
        )

        # Mongo 用户建在 admin 库，必须 authSource=admin，否则 Authentication failed
        mongo_uri = (
            f"mongodb://{os.getenv('MONGO_USER')}:{os.getenv('MONGO_PASSWORD')}"
            f"@{os.getenv('MONGO_HOST')}:{os.getenv('MONGO_PORT')}/?authSource=admin"
        )
        self.mongo_client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
        self.mongo_col = self.mongo_client[MONGO_DATABASE][MONGO_COLLECTION]

        self.es_client = Elasticsearch(
            f"http://{os.getenv('ES_HOST')}:{os.getenv('ES_PORT')}"
        )

        # MySQL 故意不指定 database：库可能还不存在，先连服务再 CREATE DATABASE（见 _ensure_mysql）。
        # SQL 里统一写 `库名`.`表名`，重连后也不会报 No database selected。
        self.mysql_conn = self._new_mysql_conn()

    def _new_mysql_conn(self) -> pymysql.connections.Connection:
        """新建 MySQL 连接（不指定库）。autocommit=True：登记必须立即可见。"""
        return pymysql.connect(
            host=os.getenv("MYSQL_HOST", "localhost"),
            port=int(os.getenv("MYSQL_PORT", "3306")),
            user=os.getenv("MYSQL_USER", "root"),
            password=os.getenv("MYSQL_PASSWORD", ""),
            charset="utf8mb4",
            autocommit=True,
        )

    def _ensure_collection(self) -> None:
        """幂等地把 Milvus 集合准备好：已存在就跳过，不存在才建。"""
        if self.client.has_collection(self.collection_name):
            return

        schema = self.client.create_schema(auto_id=False, enable_dynamic_field=False)

        schema.add_field("id", DataType.VARCHAR, max_length=128, is_primary=True)
        schema.add_field(
            "text",
            DataType.VARCHAR,
            max_length=65535,
            enable_analyzer=True,          # 供 BM25 分词
            analyzer_params=ANALYZER_PARAMS,
        )
        schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=self.dim)
        schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)   # 只声明，不填值
        schema.add_field("parent_id", DataType.VARCHAR, max_length=128)
        schema.add_field("parent_content", DataType.VARCHAR, max_length=65535)
        schema.add_field("doc_hash", DataType.VARCHAR, max_length=64)
        schema.add_field("file_path", DataType.VARCHAR, max_length=512)
        schema.add_field("timestamp", DataType.VARCHAR, max_length=32)

        # 挂 BM25 函数：text → sparse_vector 由 Milvus 服务端自动算
        # 这是本项目替代讲义 BGE-M3 本地模型的关键，零模型下载
        schema.add_function(
            Function(
                name="bm25",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["sparse_vector"],
            )
        )

        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="dense_vector",
            index_type="FLAT",
            metric_type="IP",
        )
        index_params.add_index(
            field_name="sparse_vector",
            index_type="SPARSE_INVERTED_INDEX",
            metric_type="BM25",
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

    def _ensure_mongo(self) -> None:
        """doc_hash 唯一索引 —— 去重的最后一道闸。create_index 幂等。"""
        self.mongo_col.create_index("doc_hash", unique=True)

    def _ensure_es(self) -> None:
        """建 ES 索引（显式 mapping，见 ES_MAPPING）。单节点必须 replicas=0，否则一直 yellow。"""
        if self.es_client.indices.exists(index=ES_INDEX):
            return
        self.es_client.indices.create(
            index=ES_INDEX,
            settings={"number_of_shards": 1, "number_of_replicas": 0},
            mappings=ES_MAPPING,
        )

    def _ensure_mysql(self) -> None:
        """幂等建库建表。IF NOT EXISTS 原子，避免"先查后建"的并发竞态。"""
        with self._mysql().cursor() as cur:
            cur.execute(f"CREATE DATABASE IF NOT EXISTS `{MYSQL_DATABASE}`")
            cur.execute(MYSQL_DDL)

    def _mysql(self) -> pymysql.connections.Connection:
        """取可用连接：服务端默认 8 小时断空闲连接，用前 ping，断了重建。

        不用 ping(reconnect=True)：pymysql 1.2 已废弃且不再真正重连。
        """
        try:
            self.mysql_conn.ping()
        except pymysql.err.MySQLError:
            self.mysql_conn = self._new_mysql_conn()
        return self.mysql_conn

    # ------------------------------------------------------------ 向量化
    def _embed(self, texts: list[str]) -> list[list[float]]:
        """批量向量化。必须分批（EMBED_BATCH），一次塞几百条会被网关拒绝；别 for 循环单条。"""
        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH):
            batch = texts[start : start + EMBED_BATCH]
            resp = self.emb_client.embeddings.create(model=EMBED_MODEL, input=batch)
            vectors.extend(item.embedding for item in resp.data)
        return vectors

    # ------------------------------------------------------------ 写入
    def insert(self, chunks: list[Chunk], full_text: str) -> dict[str, int | bool]:
        """一份简历写进四个库：chunks → Milvus + ES；full_text → Mongo；chunks[0] → MySQL 登记。

        full_text 是整篇原文而非子块拼接（父块之间有重叠）。"该不该写"的判断
        在 rag_pipeline（先 is_processed 再 insert），本方法只管写。
        """
        if not chunks:
            return {"milvus": 0, "es": 0, "mongo": False, "mysql": False}

        return {
            "milvus": self._insert_milvus(chunks),
            "es": self._insert_es(chunks),
            "mongo": self._save_mongo(full_text, chunks[0]),
            "mysql": self.register(chunks[0], char_count=len(full_text),
                                   chunk_count=len(chunks)),
        }

    def _insert_milvus(self, chunks: list[Chunk]) -> int:
        """写入 Milvus。用 upsert 而非 insert：insert 不做主键去重，重复入库会写出多份同 id 记录。

        查条数别用 get_collection_stats()（物理行数，upsert 旧版本 GC 前仍计入，虚高），
        权威口径是 query 的 count(*)。
        """
        vectors = self._embed([c.text for c in chunks])
        rows = [
            {
                "id": c.id,
                "text": c.text,
                "dense_vector": vector,
                # 注意：不放 sparse_vector —— 传了会报错，BM25 函数自己算
                "parent_id": c.parent_id,
                "parent_content": c.parent_content,
                "doc_hash": c.doc_hash,
                "file_path": c.file_path,
                "timestamp": c.timestamp,
            }
            for c, vector in zip(chunks, vectors)
        ]

        self.client.upsert(collection_name=self.collection_name, data=rows)
        self.client.flush(self.collection_name)   # 不 flush，紧接着 search 可能查不到
        return len(rows)

    def _insert_es(self, chunks: list[Chunk]) -> int:
        """bulk 批量写 ES。_id 复用子块 id，同 id 覆盖即天然幂等（与 Milvus upsert 对齐）。"""
        operations = []
        for c in chunks:
            operations.append({"index": {"_index": ES_INDEX, "_id": c.id}})
            operations.append(
                {
                    "content": c.text,
                    "metadata": {
                        "doc_hash": c.doc_hash,
                        "parent_id": c.parent_id,
                        "file_path": c.file_path,
                        "timestamp": c.timestamp,
                        "parent_content": c.parent_content,
                    },
                }
            )

        # refresh=True：ES 默认近实时（约 1 秒后才可搜），不刷新就紧接着 search 会漏
        resp = self.es_client.bulk(operations=operations, refresh=True)
        if resp.body.get("errors"):
            failed = [
                item for item in resp.body["items"] if item.get("index", {}).get("error")
            ]
            raise RuntimeError(f"ES bulk 部分失败（{len(failed)} 条）：{failed[:2]}")
        return len(chunks)

    def _save_mongo(self, full_text: str, chunk: Chunk) -> bool:
        """整篇原文存 Mongo，True=新建 / False=覆盖。upsert 幂等且能区分两种情况。"""
        result = self.mongo_col.update_one(
            {"doc_hash": chunk.doc_hash},
            {
                "$set": {
                    "content": full_text,
                    "file_path": chunk.file_path,
                    "timestamp": chunk.timestamp,
                    "char_count": len(full_text),
                }
            },
            upsert=True,
        )
        return result.upserted_id is not None

    # ------------------------------------------------------------ 判重（MySQL）
    def is_processed(self, doc_hash: str) -> bool:
        """判重的「读」阶段。必须与 register（写）分开：边扫边登记会把同批
        同内容文件误判成"历史上已处理过"（练习 5 的坑）。
        """
        sql = f"SELECT 1 FROM `{MYSQL_DATABASE}`.`{MYSQL_TABLE}` WHERE doc_hash = %s LIMIT 1"
        with self._mysql().cursor() as cur:
            cur.execute(sql, (doc_hash,))
            return cur.fetchone() is not None

    def register(
        self,
        chunk: Chunk,
        char_count: int = 0,
        chunk_count: int = 0,
    ) -> bool:
        """判重的「写」阶段，True=首次登记 / False=此前已登记。

        ON DUPLICATE KEY UPDATE 的 rowcount：新插入=1 / 已存在无变更=0 / 有变更=2。
        比 INSERT IGNORE 好在只吞主键冲突，数据截断等真错误照样抛。
        """
        sql = (
            f"INSERT INTO `{MYSQL_DATABASE}`.`{MYSQL_TABLE}` "
            "(doc_hash, file_path, char_count, chunk_count) "
            "VALUES (%s, %s, %s, %s) "
            "ON DUPLICATE KEY UPDATE "
            "file_path = VALUES(file_path), "
            "char_count = VALUES(char_count), "
            "chunk_count = VALUES(chunk_count)"
        )
        with self._mysql().cursor() as cur:
            cur.execute(
                sql,
                (chunk.doc_hash, chunk.file_path, char_count, chunk_count),
            )
            return cur.rowcount == 1     # 1 = 新插入；0 = 已存在（无变更）

    def count_registered(self) -> int:
        """登记表里有多少份简历。"""
        sql = f"SELECT COUNT(*) FROM `{MYSQL_DATABASE}`.`{MYSQL_TABLE}`"
        with self._mysql().cursor() as cur:
            cur.execute(sql)
            return int(cur.fetchone()[0])

    # ------------------------------------------------------------ 检索
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Milvus dense 向量检索（语义召回）。返回命中子块 + 其所属父块全文。"""
        self.client.load_collection(self.collection_name)
        query_vector = self._embed([query])[0]

        hits = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            anns_field="dense_vector",
            limit=top_k,
            output_fields=SCALAR_FIELDS,
            search_params={"metric_type": "IP"},
        )

        # hits 形如 [[Hit, Hit, ...]]；Hit 是 dict 子类，
        # hit["parent_content"] 会自动落到 entity 里取，不必手写 hit["entity"][...]
        results = []
        for hit in hits[0]:
            row = {field: hit[field] for field in SCALAR_FIELDS}
            row["id"] = hit["id"]
            row["score"] = hit["distance"]     # IP 度量，越大越相似
            results.append(row)
        return results

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        """ES 关键词检索。返回结构与 search() 对齐以便 RRF 融合；BM25 分无上界，与 IP 分不可相加。"""
        resp = self.es_client.search(
            index=ES_INDEX,
            query={"multi_match": {"query": query, "fields": ["content"]}},
            size=top_k,
        )

        results = []
        for hit in resp["hits"]["hits"]:
            source = hit["_source"]
            results.append(
                {
                    "id": hit["_id"],
                    "score": hit["_score"],
                    "text": source["content"],
                    **source["metadata"],     # 展开 doc_hash / parent_id / parent_content ...
                }
            )
        return results

    def hybrid_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Milvus 库内混合检索：dense + sparse 两路，服务端 RRFRanker 融合。

        替代讲义 BGE-M3 本地模型：sparse 由 BM25 Function 服务端生成，零模型下载。
        注意稀疏通道 data 传原始文本而非向量（Milvus 服务端现算），传向量报错。
        """
        self.client.load_collection(self.collection_name)

        reqs = [
            AnnSearchRequest(
                data=[self._embed([query])[0]],
                anns_field="dense_vector",
                param={"metric_type": "IP"},
                limit=top_k,
            ),
            AnnSearchRequest(
                data=[query],                       # ← 文本，不是向量
                anns_field="sparse_vector",
                param={"metric_type": "BM25"},
                limit=top_k,
            ),
        ]

        hits = self.client.hybrid_search(
            collection_name=self.collection_name,
            reqs=reqs,
            ranker=RRFRanker(k=RRF_K),
            limit=top_k,
            output_fields=SCALAR_FIELDS,
        )

        results = []
        for hit in hits[0]:
            row = {field: hit[field] for field in SCALAR_FIELDS}
            row["id"] = hit["id"]
            row["score"] = hit["distance"]     # 库内 RRF 分，量纲已统一
            results.append(row)
        return results

    def search_all(self, query: str, top_k: int = 5, recall_k: int = RECALL_K) -> list[dict]:
        """跨库混合检索：Milvus 库内混合 + ES BM25 → 应用层 RRF
        （Milvus 无法参与 ES 排序，跨库融合只能在应用层做）。"""
        milvus_hits = self.hybrid_search(query, top_k=recall_k)
        es_hits = self.search_bm25(query, top_k=recall_k)
        return rrf_fuse([milvus_hits, es_hits])[:top_k]

    def rerank(self, query: str, hits: list[dict], top_n: int = RERANK_TOP_N) -> list[dict]:
        """百炼 Cross-Encoder 精排。

        召回是双塔打分（快而糙），精排 query+doc 拼接打分（慢而准），
        所以流程是"宽召回 → 精排"。端点必须走原生 DashScope 路径。
        """
        if not hits:
            return []

        resp = requests.post(
            RERANK_URL,
            headers={
                "Authorization": f"Bearer {os.getenv('RERANK_API_KEY')}",
                "Content-Type": "application/json",
            },
            json={
                "model": RERANK_MODEL,
                "query": query,
                "documents": [h["text"] for h in hits],
                "top_n": min(top_n, len(hits)),
            },
            timeout=60,
        )
        resp.raise_for_status()

        ranked = []
        for item in resp.json()["output"]["results"]:
            row = dict(hits[item["index"]])       # 按返回的 index 回填原 hit
            row["rerank_score"] = item["relevance_score"]
            ranked.append(row)
        # 百炼返回时已是降序，但显式排一次不依赖服务端的排序承诺
        ranked.sort(key=lambda r: r["rerank_score"], reverse=True)
        return ranked

    def hybrid_search_with_rerank(
        self,
        query: str,
        top_k: int = RERANK_TOP_N,
        recall_k: int = RECALL_K,
    ) -> list[dict]:
        """检索层一站式入口：三路召回（Milvus dense/sparse + ES BM25）→ 库内 RRF → 跨库 RRF → 精排。"""
        candidates = self.search_all(query, top_k=recall_k, recall_k=recall_k)
        return self.rerank(query, candidates, top_n=top_k)

    def get_full_text(self, doc_hash: str) -> str:
        """从 MongoDB 取回整篇原文；不存在返回空串。"""
        doc = self.mongo_col.find_one({"doc_hash": doc_hash}, {"content": 1})
        return doc["content"] if doc else ""

    def get_metadata_by_hash(self, doc_hash: str) -> list[dict]:
        """按 doc_hash 从 Milvus 取回该文档的全部子块元数据（不返回向量）。"""
        self.client.load_collection(self.collection_name)
        return self.client.query(
            collection_name=self.collection_name,
            filter=f'doc_hash == "{doc_hash}"',
            output_fields=SCALAR_FIELDS,
        )


if __name__ == "__main__":
    # 自检入口：建四库 → 四库写入 → 判重演示 → 四层检索逐级验证
    from pathlib import Path

    from document_processor import DocumentProcessor

    store = VectorStore()

    info = store.client.describe_collection(store.collection_name)
    print(f"[Milvus] 集合 {store.collection_name} 就绪，{len(info.get('fields', []))} 个字段")
    print(f"[Mongo]  库 {MONGO_DATABASE}.{MONGO_COLLECTION} 就绪（doc_hash 唯一索引）")
    print(f"[ES]     索引 {ES_INDEX} 就绪，分析器 = {ES_ANALYZER}")
    print(f"[MySQL]  表 {MYSQL_DATABASE}.{MYSQL_TABLE} 就绪，已登记 {store.count_registered()} 份")

    pdf = Path(__file__).resolve().parent.parent / "个人简历测试.pdf"
    text, chunks = DocumentProcessor().process_with_text(str(pdf))
    doc_hash = chunks[0].doc_hash

    # ---------------------------------------------------------- 判重演示
    # 关键：先读后写，两个阶段分开。不能边 insert 边判断。
    already = store.is_processed(doc_hash)
    print(f"\n[判重] doc_hash={doc_hash[:12]} → {'历史上已处理过' if already else '新文档'}")

    print(f"\n切块 {len(chunks)} 个（全文 {len(text)} 字），写入四库 ...")
    counts = store.insert(chunks, full_text=text)
    print(
        f"  Milvus {counts['milvus']} 条 / ES {counts['es']} 条 / "
        f"Mongo {'新建原文' if counts['mongo'] else '更新原文'} / "
        f"MySQL {'首次登记' if counts['mysql'] else '已登记（幂等）'}"
    )

    # 再登记一次，验证幂等：counts['mysql'] 应为 False，行数不应增加
    before = store.count_registered()
    store.register(chunks[0], char_count=len(text), chunk_count=len(chunks))
    after = store.count_registered()
    print(f"  重复登记一次 → 行数 {before} → {after}（应相等，幂等 ✅）")

    query = "有没有用过 Elasticsearch 和 Milvus 做检索的候选人？"

    print(f"\n[1/4 Milvus 纯语义] {query}")
    for rank, hit in enumerate(store.search(query, top_k=3), start=1):
        print(f"  #{rank} score={hit['score']:.4f}  {hit['text'][:36].strip()}...")

    print(f"\n[2/4 ES 纯关键词] {query}")
    for rank, hit in enumerate(store.search_bm25(query, top_k=3), start=1):
        print(f"  #{rank} score={hit['score']:.4f}  {hit['text'][:36].strip()}...")

    print(f"\n[3/4 Milvus 库内混合（dense+BM25 服务端 RRF）]")
    for rank, hit in enumerate(store.hybrid_search(query, top_k=3), start=1):
        print(f"  #{rank} rrf={hit['score']:.5f}  {hit['text'][:36].strip()}...")

    print(f"\n[4/4 跨库融合（Milvus 混合 + ES → 应用层 RRF）]")
    for rank, hit in enumerate(store.search_all(query, top_k=4), start=1):
        print(
            f"  #{rank} rrf={hit['rrf']:.5f}  [{'+'.join(hit['routes'])}]  "
            f"{hit['text'][:30].strip()}..."
        )

    print(f"\n[精排] 对前 {RECALL_K} 条候选做 Cross-Encoder rerank（{RERANK_MODEL}）")
    for rank, hit in enumerate(store.hybrid_search_with_rerank(query, top_k=3), start=1):
        print(
            f"  #{rank} rerank={hit['rerank_score']:.4f}（融合分 {hit['rrf']:.5f}）"
            f"  {hit['text'][:30].strip()}..."
        )

    full = store.get_full_text(doc_hash)
    print(f"\n[Mongo 原文回溯] doc_hash={doc_hash[:12]} → 取回 {len(full)} 字")
