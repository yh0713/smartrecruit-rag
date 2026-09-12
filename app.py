"""app.py —— 界面层：简历入库 + 岗位匹配

只管交互与展示，业务逻辑全在 RAGPipeline。
启动：streamlit run app.py
"""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag_pipeline import RAGPipeline

UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
TEST_DIR = Path(__file__).resolve().parent / "data" / "testdata"

BADGE = {"强烈推荐": "✅", "可以考虑": "⚠️", "不推荐": "❌"}

DEFAULT_JD = """岗位：RAG / 大模型应用工程师（后端方向）

职责：
1. 负责检索增强生成（RAG）系统的设计与开发，包括文档解析、切块、向量化与检索链路
2. 负责向量数据库与搜索引擎的选型与调优（Milvus / Elasticsearch）
3. 参与大模型应用的后端服务开发（FastAPI）

要求：
1. 3 年以上 Python 后端开发经验
2. 熟悉 LangChain 等大模型应用框架，有 RAG 项目落地经验
3. 熟悉 Milvus、Elasticsearch 等检索组件的原理与调优
4. 熟悉 MySQL、Redis、Docker 等常用后端组件"""


@st.cache_resource(show_spinner="正在连接四库 ...")
def get_pipeline() -> RAGPipeline:
    """进程级单例：Streamlit 每次交互都重跑脚本，不能反复重连四个库。"""
    return RAGPipeline()


# ---------------------------------------------------------------- 展示组件
def render_candidate(rank: int, cand) -> None:
    """单个候选人的评估卡片。"""
    with st.container(border=True):
        left, right = st.columns([3, 1])
        left.subheader(f"#{rank}　{cand.candidate_name}")
        right.metric("匹配度", f"{cand.match_score}/100")

        badge = BADGE.get(cand.recommendation, "")
        st.progress(cand.match_score / 100, text=f"{badge} {cand.recommendation}")

        c1, c2, c3 = st.columns(3)
        years = f"{cand.experience_years} 年" if cand.experience_years else "未体现"
        c1.markdown(f"**工作年限**\n\n{years}")
        c2.markdown("**已具备**\n\n" + ("、".join(cand.matched_skills) or "—"))
        c3.markdown("**缺失**\n\n" + ("、".join(cand.missing_skills) or "—"))

        if cand.concerns:
            st.warning("风险点：" + "；".join(cand.concerns))
        st.markdown(f"**评语**　{cand.summary}")

        if cand.evidence:
            with st.expander(f"证据原文（{len(cand.evidence)} 条）"):
                for ev in cand.evidence:
                    st.markdown(f"> {ev}")


def render_sources(hits: list[dict]) -> None:
    """展示结论的依据来源 —— 没有来源的 AI 结论不可信。"""
    with st.expander(f"检索依据（命中 {len(hits)} 个父块）"):
        for i, hit in enumerate(hits, start=1):
            score = hit.get("rerank_score", hit.get("rrf", 0.0))
            routes = "+".join(hit.get("routes", [])) or "—"
            st.markdown(
                f"**#{i}** `{Path(hit['file_path']).name}`　"
                f"精排分 `{score:.4f}`　召回路径 `{routes}`"
            )
            st.caption(hit["text"].strip()[:220])
            st.divider()


def ingest_rows(rows: list[dict]) -> None:
    """入库/匹配结果统一渲染成表格。"""
    st.dataframe(
        [
            {
                "文件": r["file"],
                "状态": r["status"],
                "块数": r.get("chunks", 0),
                "备注": r.get("detail", ""),
            }
            for r in rows
        ],
        hide_index=True,
    )


# ---------------------------------------------------------------- 侧边栏
def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        st.header("知识库")

        stats = pipeline.stats()
        a, b = st.columns(2)
        a.metric("简历", stats["mongo_resumes"])
        b.metric("文本块", stats["milvus_chunks"])
        st.caption(
            f"ES {stats['es_chunks']} 块 ｜ MySQL 登记 {stats['mysql_registered']} 份"
        )

        st.divider()
        st.subheader("入库")

        # 上一轮入库的结果（rerun 会清空界面，放 session_state 里带过来）
        flash = st.session_state.pop("flash", None)
        if flash is not None:
            ingest_rows(flash)

        uploads = st.file_uploader(
            "上传简历 PDF", type=["pdf"], accept_multiple_files=True
        )
        if uploads and st.button("写入知识库", type="primary"):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            rows = []
            for f in uploads:
                dest = UPLOAD_DIR / f.name
                dest.write_bytes(f.getbuffer())
                rows.append(pipeline.ingest(dest))
            st.session_state["flash"] = rows
            st.rerun()

        directory = st.text_input("目录批量入库", value=str(TEST_DIR))
        if st.button("扫描该目录"):
            if not Path(directory).is_dir():
                st.error("目录不存在")
            else:
                st.session_state["flash"] = pipeline.ingest_directory(directory)
                st.rerun()

        st.divider()
        with st.expander(f"已入库简历（{stats['mongo_resumes']}）"):
            for doc in pipeline.list_documents():
                st.caption(
                    f"{Path(doc['file_path']).name}　{doc['char_count']} 字　"
                    f"{doc['timestamp'][:19]}"
                )


# ---------------------------------------------------------------- 主区域
def main() -> None:
    st.title("📄 智能简历推荐系统")
    st.caption("Milvus 语义 + ES 关键词 + RRF 融合 + Cross-Encoder 精排 → LLM 结构化评估")

    pipeline = get_pipeline()
    render_sidebar(pipeline)

    tab_match, tab_recall = st.tabs(["岗位匹配", "召回预览"])

    with tab_match:
        with st.form("match_form"):
            jd = st.text_area("岗位要求", value=DEFAULT_JD, height=240)
            match_top_k = st.slider("评估候选人数", 1, 10, 5)
            submitted = st.form_submit_button("开始匹配", type="primary")

        if submitted:
            if not jd.strip():
                st.error("请填写岗位要求")
            else:
                with st.spinner("三路召回 → RRF → 精排 → 模型评估 ..."):
                    try:
                        result = pipeline.match(jd, top_k=match_top_k)
                    except Exception as exc:
                        st.error(f"匹配失败：{type(exc).__name__}: {exc}")
                    else:
                        report = result["report"]
                        if not report.candidates:
                            st.info("未检索到候选人档案，请先在左侧入库简历。")
                        for rank, cand in enumerate(report.candidates, start=1):
                            render_candidate(rank, cand)
                        st.caption(
                            f"上下文 {result['context_chars']} 字 ｜ "
                            f"命中 {len(result['hits'])} 个父块"
                        )
                        render_sources(result["hits"])

    with tab_recall:
        query = st.text_input("查询词", value="熟悉 Milvus 和 Elasticsearch 的候选人")
        recall_top_k = st.slider("返回条数", 1, 10, 5, key="recall_top_k")
        if st.button("检索"):
            if not query.strip():
                st.error("请填写查询词")
            else:
                with st.spinner("三路召回 → RRF → 精排 ..."):
                    hits = pipeline.search(query, top_k=recall_top_k)
                st.dataframe(
                    [
                        {
                            "排名": i,
                            "精排分": round(hit.get("rerank_score", 0.0), 4),
                            "召回路径": "+".join(hit.get("routes", [])) or "—",
                            "来源": Path(hit["file_path"]).name,
                            "片段": hit["text"].strip()[:60],
                        }
                        for i, hit in enumerate(hits, start=1)
                    ],
                    hide_index=True,
                )


main()
