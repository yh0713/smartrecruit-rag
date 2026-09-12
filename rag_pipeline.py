"""rag_pipeline.py —— 编排层：只决定调用顺序与跳过什么，不含业务细节。

两条主流程：
    写入：PDF → 指纹 → 判重 → 切块 → 四库落库
    查询：岗位要求 → 三路召回 → RRF → 精排 → LLM 结构化评估
"""
from __future__ import annotations

from pathlib import Path

from chain import (
    CONTEXT_TOP_K,
    MatchReport,
    build_evaluator,
    format_context,
)
from document_processor import DocumentProcessor
from vector_store import (
    ES_INDEX,
    MONGO_COLLECTION,
    VectorStore,
)


class RAGPipeline:
    """智能简历推荐系统的流程编排器。构造一次，反复使用。"""

    def __init__(self, store: VectorStore | None = None) -> None:
        self.processor = DocumentProcessor()
        self.store = store or VectorStore()
        # 只构建"评估"那一段（不含检索），检索由本类自己控制，
        # 这样一次 match 只检索一次，命中的原文还能直接给界面展示
        self.evaluator = build_evaluator()

    # ================================================================ 写入
    def ingest(self, pdf_path: str | Path) -> dict:
        """单份简历入库。顺序不能改：先判重，命中历史直接返回 —— 省掉 embedding 与 rerank 的钱。"""
        text, chunks = self.processor.process_with_text(str(pdf_path))
        if not chunks:
            return {"file": Path(pdf_path).name, "status": "失败",
                    "detail": "未抽出任何文本（可能是扫描件图片）"}

        doc_hash = chunks[0].doc_hash

        # —— 读阶段：先查历史 ——
        if self.store.is_processed(doc_hash):
            return {"file": Path(pdf_path).name, "status": "已处理，跳过",
                    "doc_hash": doc_hash, "chunks": 0}

        # —— 写阶段 ——
        counts = self.store.insert(chunks, full_text=text)
        return {
            "file": Path(pdf_path).name,
            "status": "入库" if counts["mysql"] else "同批重复，已覆盖",
            "doc_hash": doc_hash,
            "chunks": counts["milvus"],
            "chars": len(text),
        }

    def ingest_directory(self, directory: str | Path, pattern: str = "*.pdf") -> list[dict]:
        """批量入库一个目录。必须两阶段：先读完所有指纹，再开始写。

        边扫边登记会让同批内容相同的第二个文件被误判成"历史上已处理过"
        （练习 5 踩过的坑）—— "历史命中"该整份跳过，"本批内重复"仍要如实记账。
        """
        files = sorted(Path(directory).glob(pattern))

        # ---------- 第一阶段：只读，算指纹 + 查历史 ----------
        plan: list[dict] = []
        for path in files:
            try:
                text, chunks = self.processor.process_with_text(str(path))
            except Exception as exc:          # 单个文件坏掉不该整批中断
                plan.append({"path": path, "error": f"{type(exc).__name__}: {exc}"})
                continue

            if not chunks:
                plan.append({"path": path, "error": "未抽出任何文本（可能是扫描件图片）"})
                continue

            doc_hash = chunks[0].doc_hash
            plan.append(
                {
                    "path": path,
                    "text": text,
                    "chunks": chunks,
                    "doc_hash": doc_hash,
                    # 此刻本批次还没有任何写入，所以这个结果只反映"历史"
                    "known": self.store.is_processed(doc_hash),
                }
            )

        # ---------- 第二阶段：只写 ----------
        results: list[dict] = []
        for item in plan:
            name = item["path"].name

            if "error" in item:
                results.append({"file": name, "status": "失败", "detail": item["error"]})
                continue

            if item["known"]:
                results.append({"file": name, "status": "已处理，跳过",
                                "doc_hash": item["doc_hash"], "chunks": 0})
                continue

            counts = self.store.insert(item["chunks"], full_text=item["text"])
            results.append(
                {
                    "file": name,
                    # mysql=False 意味着本批里已经写过同内容的另一份
                    "status": "入库" if counts["mysql"] else "同批重复，已覆盖",
                    "doc_hash": item["doc_hash"],
                    "chunks": counts["milvus"],
                    "chars": len(item["text"]),
                }
            )

        return results

    # ================================================================ 查询
    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """只检索，不调 LLM。用于界面上的"召回预览"。"""
        return self.store.hybrid_search_with_rerank(query, top_k=top_k)

    def match(self, job_description: str, top_k: int = CONTEXT_TOP_K) -> dict:
        """核心方法：岗位要求 → 匹配报告。

        返回 dict 携带 hits，供界面展示"结论基于哪些原文"。
        检索只做一次：先自己检索，再把拼好的上下文交给 evaluator。
        """
        hits = self.store.hybrid_search_with_rerank(job_description, top_k=top_k)
        context = format_context(hits)

        report: MatchReport = self.evaluator.invoke(
            {"job_description": job_description, "context": context}
        )

        return {
            "report": report,
            "hits": hits,
            "context_chars": len(context),
            "context": context,
        }

    # ================================================================ 状态
    def stats(self) -> dict:
        """四库数据量。Milvus 必须用 count(*)：get_collection_stats() 是物理行数，
        upsert 旧版本在后台 GC 前仍计入（虚高）。
        """
        self.store.client.load_collection(self.store.collection_name)
        milvus_rows = self.store.client.query(
            collection_name=self.store.collection_name,
            filter="",
            output_fields=["count(*)"],
        )
        return {
            "milvus_chunks": milvus_rows[0]["count(*)"],
            "es_chunks": self.store.es_client.count(index=ES_INDEX)["count"],
            "mongo_resumes": self.store.mongo_col.count_documents({}),
            "mysql_registered": self.store.count_registered(),
        }

    def list_documents(self) -> list[dict]:
        """已入库简历清单。从 Mongo 读：MySQL 只有指纹统计，Mongo 才有界面要的元数据。"""
        cursor = self.store.mongo_col.find(
            {},
            {"_id": 0, "doc_hash": 1, "file_path": 1, "timestamp": 1, "char_count": 1},
        ).sort("timestamp", -1)
        return list(cursor)


if __name__ == "__main__":
    import sys

    pipeline = RAGPipeline()

    print("=" * 70)
    print("一、当前四库状态")
    print("=" * 70)
    for key, value in pipeline.stats().items():
        print(f"  {key:<20} {value}")

    print()
    print("=" * 70)
    print("二、批量入库演示（同一个目录跑两遍，验证增量判重）")
    print("=" * 70)
    test_dir = Path(__file__).resolve().parent / "data" / "testdata"
    if test_dir.exists():
        for round_no in (1, 2):
            print(f"\n--- 第 {round_no} 遍 ---")
            for row in pipeline.ingest_directory(test_dir):
                extra = row.get("detail") or f"{row.get('chunks', 0)} 块"
                print(f"  {row['file']:<24} {row['status']:<14} {extra}")
    else:
        print(f"  （跳过：{test_dir} 不存在）")

    print()
    print("=" * 70)
    print("三、批量入库单文件（幂等验证）")
    print("=" * 70)
    pdf = Path(__file__).resolve().parent.parent / "个人简历测试.pdf"
    for round_no in (1, 2):
        row = pipeline.ingest(pdf)
        print(f"  第 {round_no} 次：{row['status']}")

    print()
    print("=" * 70)
    print("四、已入库简历清单")
    print("=" * 70)
    for doc in pipeline.list_documents():
        print(f"  {doc['doc_hash'][:12]}  {doc['char_count']:>5} 字  "
              f"{Path(doc['file_path']).name}")

    if "--match" in sys.argv:
        print()
        print("=" * 70)
        print("五、岗位匹配（会真实调用 LLM）")
        print("=" * 70)
        jd = "招聘 RAG 工程师：3 年 Python 后端经验，熟悉 LangChain、Milvus、Elasticsearch"
        result = pipeline.match(jd)
        print(f"上下文 {result['context_chars']} 字，命中 {len(result['hits'])} 条")
        for rank, cand in enumerate(result["report"].candidates, start=1):
            print(f"  #{rank} {cand.candidate_name}  {cand.match_score}/100  "
                  f"{cand.recommendation}")
