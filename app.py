"""app.py —— 界面层：简历入库 + 岗位匹配

只管交互与展示，业务逻辑全在 RAGPipeline。
启动：streamlit run app.py
"""
from __future__ import annotations

import html
import time
from pathlib import Path

import streamlit as st

from rag_pipeline import RAGPipeline

UPLOAD_DIR = Path(__file__).resolve().parent / "data" / "uploads"
TEST_DIR = Path(__file__).resolve().parent / "data" / "testdata"

# 推荐结论 → (药丸样式, 展示文案)
PILL = {
    "强烈推荐": "ok",
    "可以考虑": "mid",
    "不推荐": "bad",
}

STACK = ["Milvus 语义", "ES BM25", "RRF 融合", "Cross-Encoder 精排", "LLM 结构化评估"]

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

# ---------------------------------------------------------------- 样式
CSS = """
:root{
  --bg:#0A0E14; --surface:#111926; --border:rgba(255,255,255,.08);
  --border-2:rgba(255,255,255,.15); --text:#E6EDF3; --muted:#8B98AA; --dim:#5D6B7E;
  --c1:#22D3EE; --c2:#8B5CF6;
}
html, body, [class*="st-"], input, textarea, button{
  font-family:-apple-system,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
}
.stApp{
  background:
    radial-gradient(880px 480px at 10% -10%, rgba(34,211,238,.10), transparent 60%),
    radial-gradient(760px 460px at 95% -2%, rgba(139,92,246,.13), transparent 58%),
    var(--bg);
  color:var(--text);
}
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stDecoration"], #MainMenu, footer{ display:none; }
/* 隐藏右上角 Deploy 与工具栏动作，避免演示截图里出现平台按钮 */
[data-testid="stAppDeployButton"], [data-testid="stToolbarActions"]{ display:none !important; }
.block-container{ padding:2.4rem 2.2rem 4rem; max-width:1280px; }

/* ---------- 侧边栏 ---------- */
[data-testid="stSidebar"]{
  background:linear-gradient(180deg,#0C121C 0%,#0A0E14 100%);
  border-right:1px solid var(--border);
}
[data-testid="stSidebar"] .block-container{ padding-top:1.6rem; }

/* ---------- 原生控件重绘 ---------- */
.stButton>button, [data-testid="stFormSubmitButton"] button{
  background:rgba(255,255,255,.045); border:1px solid var(--border-2); color:var(--text);
  border-radius:10px; font-weight:600; padding:.5rem 1rem; transition:all .18s ease;
}
.stButton>button:hover, [data-testid="stFormSubmitButton"] button:hover{
  border-color:rgba(34,211,238,.55); color:#fff; transform:translateY(-1px);
  box-shadow:0 0 0 3px rgba(34,211,238,.12);
}
/* 主按钮：kind 有 primary / primaryFormSubmit 等变体，用前缀匹配 */
button[kind^="primary"], [data-testid^="stBaseButton-primary"]{
  background:linear-gradient(135deg,var(--c1),var(--c2)); border:none; color:#061018;
}
button[kind^="primary"]:hover, [data-testid^="stBaseButton-primary"]:hover{
  color:#061018; box-shadow:0 8px 26px -10px rgba(34,211,238,.65); transform:translateY(-1px);
}

[data-baseweb="textarea"], [data-baseweb="base-input"]{
  background:#0E1520 !important; border:1px solid var(--border) !important;
  border-radius:10px !important; transition:border-color .18s ease, box-shadow .18s ease;
}
[data-baseweb="textarea"]:focus-within, [data-baseweb="base-input"]:focus-within{
  border-color:rgba(34,211,238,.6) !important; box-shadow:0 0 0 3px rgba(34,211,238,.12);
}
[data-baseweb="textarea"] textarea, [data-baseweb="base-input"] input{
  background:transparent !important; color:var(--text) !important;
}
[data-testid="stFileUploaderDropzone"]{
  background:rgba(255,255,255,.02); border:1px dashed var(--border-2); border-radius:12px;
  padding:1rem;
}
[data-testid="stFileUploaderDropzone"]:hover{ border-color:rgba(34,211,238,.45); }

[data-testid="stForm"]{
  background:rgba(255,255,255,.022); border:1px solid var(--border);
  border-radius:16px; padding:1.3rem 1.4rem .5rem;
}
[data-testid="stExpander"] details{
  background:rgba(255,255,255,.02); border:1px solid var(--border); border-radius:12px;
}
[data-testid="stExpander"] summary{ color:var(--muted); font-weight:600; font-size:.86rem; }
[data-testid="stExpander"] summary:hover{ color:var(--c1); }

[data-testid="stTabs"] [role="tablist"]{ gap:.4rem; border-bottom:1px solid var(--border); }
[data-testid="stTabs"] button[role="tab"]{
  background:transparent; border:none; color:var(--muted); font-weight:600;
  padding:.6rem 1.15rem; font-size:.92rem;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]{ color:var(--text); }
[data-testid="stTabs"] [data-baseweb="tab-highlight"]{
  background:linear-gradient(90deg,var(--c1),var(--c2)) !important; height:2px;
}
[data-testid="stTabs"] [data-baseweb="tab-border"]{ background:transparent !important; }

[data-testid="stSlider"] [role="slider"]{
  background:linear-gradient(135deg,var(--c1),var(--c2)); border:none;
  box-shadow:0 0 0 4px rgba(34,211,238,.15);
}
[data-testid="stSlider"] [data-testid="stTickBar"]{ background:transparent; }
label, .stSlider label p, .stTextArea label p, .stTextInput label p{
  color:var(--muted) !important; font-size:.82rem !important; font-weight:600;
}

/* ---------- 顶部品牌区 ---------- */
.sr-hero{ padding:0 0 1.2rem; }
.sr-title{
  font-size:2.05rem; font-weight:800; letter-spacing:-.02em; margin:0; line-height:1.2;
  background:linear-gradient(100deg,#EAF6FF 8%,#7DE3F4 46%,#A78BFA 92%);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}
.sr-sub{ color:var(--muted); font-size:.9rem; margin-top:.5rem; }
.sr-chips{ display:flex; flex-wrap:wrap; gap:.4rem; margin-top:.9rem; }
.sr-chip{
  font-size:.73rem; letter-spacing:.02em; color:#9FE8F5; padding:.22rem .6rem;
  border-radius:999px; background:rgba(34,211,238,.09); border:1px solid rgba(34,211,238,.22);
}
.sr-chip.v{ color:#C4B5FD; background:rgba(139,92,246,.10); border-color:rgba(139,92,246,.26); }

/* ---------- 指标卡 ---------- */
.sr-kpis{ display:grid; grid-template-columns:1fr 1fr; gap:.55rem; }
.sr-kpi{
  background:linear-gradient(160deg,rgba(255,255,255,.055),rgba(255,255,255,.018));
  border:1px solid var(--border); border-radius:12px; padding:.7rem .8rem;
}
.sr-kpi .v{ font-size:1.32rem; font-weight:700; font-variant-numeric:tabular-nums; }
.sr-kpi .l{ font-size:.72rem; color:var(--muted); margin-top:.15rem; }
.sr-kpi.hl .v{
  background:linear-gradient(135deg,#7DE3F4,#A78BFA);
  -webkit-background-clip:text; background-clip:text; color:transparent;
}

/* ---------- 候选人卡片 ---------- */
.sr-card{
  position:relative; overflow:hidden; margin-bottom:.8rem; padding:1rem 1.15rem;
  border:1px solid var(--border); border-radius:16px;
  background:linear-gradient(165deg,rgba(255,255,255,.055),rgba(255,255,255,.016));
}
.sr-card::before{
  content:""; position:absolute; top:0; bottom:0; left:0; width:3px;
  background:linear-gradient(180deg,var(--c1),var(--c2));
}
.sr-card.top{
  border-color:rgba(34,211,238,.32);
  box-shadow:0 14px 36px -20px rgba(34,211,238,.6);
}
.sr-card:hover{ border-color:var(--border-2); }
.sr-head{ display:flex; align-items:flex-start; justify-content:space-between; gap:1rem; }
.sr-name{ font-size:1.1rem; font-weight:700; }
.sr-rank{
  display:inline-block; margin-right:.5rem; padding:.1rem .42rem; border-radius:6px;
  font-size:.72rem; font-weight:700; color:#061018;
  background:linear-gradient(135deg,var(--c1),var(--c2)); vertical-align:1px;
}
.sr-score{
  font-size:1.7rem; font-weight:800; line-height:1.1; font-variant-numeric:tabular-nums;
  background:linear-gradient(135deg,#7DE3F4,#A78BFA);
  -webkit-background-clip:text; background-clip:text; color:transparent; white-space:nowrap;
}
.sr-score span{
  font-size:.78rem; font-weight:600; margin-left:.18rem;
  -webkit-text-fill-color:var(--dim); color:var(--dim);
}
.sr-bar{ height:6px; border-radius:999px; background:rgba(255,255,255,.07); margin:.75rem 0 .1rem; }
.sr-bar>i{
  display:block; height:100%; border-radius:999px;
  background:linear-gradient(90deg,var(--c1),var(--c2));
}
.sr-pill{
  display:inline-block; margin-top:.5rem; padding:.14rem .55rem; border-radius:999px;
  font-size:.73rem; font-weight:700;
}
.sr-pill.ok{ color:#6EE7B7; background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.3); }
.sr-pill.mid{ color:#FCD34D; background:rgba(251,191,36,.12); border:1px solid rgba(251,191,36,.3); }
.sr-pill.bad{ color:#FCA5A5; background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.3); }

.sr-meta{
  display:grid; grid-template-columns:96px minmax(0,1fr) minmax(0,1.05fr);
  gap:1.5rem; margin-top:.9rem; align-items:start;
}
.sr-meta .k{ font-size:.7rem; color:var(--dim); letter-spacing:.07em; text-transform:uppercase; }
.sr-meta .v{ font-size:.9rem; margin-top:.22rem; }
.sr-tags{ display:flex; flex-wrap:wrap; gap:.3rem; margin-top:.3rem; }
/* 模型可能给出很长的技能描述，允许标签内换行，避免撑破栅格 */
.sr-tag{ font-size:.73rem; padding:.15rem .5rem; border-radius:6px; max-width:100%; overflow-wrap:anywhere; }
.sr-tag.y{ color:#93E6C8; background:rgba(52,211,153,.10); border:1px solid rgba(52,211,153,.22); }
.sr-tag.n{ color:#9AA8B8; background:rgba(255,255,255,.04); border:1px solid var(--border); }
.sr-tag.miss{ color:#E9B4B4; background:rgba(248,113,113,.08); border-color:rgba(248,113,113,.22); }

.sr-alert{
  margin-top:.85rem; padding:.5rem .7rem; border-radius:0 8px 8px 0; font-size:.84rem;
  color:#F6D6A2; background:rgba(251,191,36,.07); border-left:3px solid rgba(251,191,36,.7);
}
.sr-sum{
  margin-top:.85rem; padding-left:.75rem; font-size:.89rem; line-height:1.7; color:#C2CEDB;
  border-left:2px solid var(--border-2);
}
.sr-ev{
  margin-bottom:.45rem; padding:.5rem .7rem; border-radius:0 8px 8px 0; line-height:1.65;
  font-size:.82rem; color:#98A6B6; background:rgba(255,255,255,.025);
  border-left:2px solid rgba(34,211,238,.35);
}

/* ---------- 结果表格 ---------- */
.sr-tbl{
  width:100%; border-collapse:separate; border-spacing:0; font-size:.85rem;
  border:1px solid var(--border); border-radius:12px; overflow:hidden;
}
.sr-tbl th{
  text-align:left; padding:.55rem .7rem; font-size:.7rem; font-weight:600;
  letter-spacing:.06em; text-transform:uppercase; color:var(--muted);
  background:rgba(255,255,255,.035); white-space:nowrap;
}
.sr-tbl td{
  padding:.5rem .7rem; border-top:1px solid var(--border); color:#C2CEDB; vertical-align:top;
}
.sr-tbl th:first-child, .sr-tbl td:first-child{ width:1%; white-space:nowrap; }
.sr-tbl tr:hover td{ background:rgba(255,255,255,.02); }
.sr-tbl .num{ font-variant-numeric:tabular-nums; color:#9FE8F5; font-weight:600; }
.sr-tbl .file{ color:var(--text); }
.sr-tbl .path{ color:#C4B5FD; font-size:.78rem; }

/* ---------- 侧边栏内小构件 ---------- */
.sr-sec{
  display:flex; align-items:center; gap:.45rem; margin:.2rem 0 .7rem;
  font-size:.78rem; font-weight:700; letter-spacing:.1em; text-transform:uppercase; color:var(--dim);
}
.sr-sec::after{ content:""; flex:1; height:1px; background:var(--border); }
.sr-doc{
  display:flex; justify-content:space-between; gap:.5rem; padding:.4rem 0;
  font-size:.78rem; color:#A9B6C4; border-bottom:1px solid rgba(255,255,255,.045);
}
.sr-doc .t{ color:var(--dim); font-variant-numeric:tabular-nums; }
.sr-note{ font-size:.76rem; color:var(--dim); line-height:1.6; }
"""


def esc(value) -> str:
    """所有外部文本（简历内容、模型输出）进 HTML 前必须转义。"""
    return html.escape(str(value if value is not None else ""), quote=True)


def paint(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


@st.cache_resource(show_spinner="正在连接四库 ...")
def get_pipeline() -> RAGPipeline:
    """进程级单例：Streamlit 每次交互都重跑脚本，不能反复重连四个库。"""
    return RAGPipeline()


# ---------------------------------------------------------------- 展示组件
def render_hero() -> None:
    chips = "".join(
        f'<span class="sr-chip{" v" if i % 2 else ""}">{esc(s)}</span>'
        for i, s in enumerate(STACK)
    )
    paint(
        '<div class="sr-hero">'
        '<h1 class="sr-title">智能简历推荐系统</h1>'
        '<div class="sr-sub">多路召回 · 混合排序 · 结构化评估 —— 从简历库到候选人排序的完整链路</div>'
        f'<div class="sr-chips">{chips}</div>'
        "</div>"
    )


def tag_list(items: list[str], cls: str) -> str:
    return "".join(f'<span class="sr-tag {cls}">{esc(s)}</span>' for s in items)


def candidate_card(rank: int, cand, is_top: bool) -> str:
    """把一条评估结论渲染成卡片（模型输出一律转义）。"""
    score = max(0, min(100, int(cand.match_score or 0)))
    kind = PILL.get(cand.recommendation, "mid")
    years = f"{cand.experience_years} 年" if cand.experience_years else "未体现"
    has = tag_list(cand.matched_skills, "y") or '<span class="sr-tag n">—</span>'
    miss = tag_list(cand.missing_skills, "miss") or '<span class="sr-tag n">无</span>'
    concern = (
        f'<div class="sr-alert"><b>风险点</b>　{esc("；".join(cand.concerns))}</div>'
        if cand.concerns
        else ""
    )
    return (
        f'<div class="sr-card{" top" if is_top else ""}">'
        '<div class="sr-head"><div>'
        f'<div class="sr-name"><span class="sr-rank">#{rank}</span>{esc(cand.candidate_name)}</div>'
        f'<span class="sr-pill {kind}">{esc(cand.recommendation or "—")}</span>'
        "</div>"
        f'<div class="sr-score">{score}<span>/100</span></div>'
        "</div>"
        f'<div class="sr-bar"><i style="width:{score}%"></i></div>'
        '<div class="sr-meta">'
        f'<div><div class="k">工作年限</div><div class="v">{esc(years)}</div></div>'
        f'<div><div class="k">已具备</div><div class="sr-tags">{has}</div></div>'
        f'<div><div class="k">缺失</div><div class="sr-tags">{miss}</div></div>'
        "</div>"
        f"{concern}"
        f'<div class="sr-sum">{esc(cand.summary)}</div>'
        "</div>"
    )


def render_candidate(rank: int, cand, is_top: bool) -> None:
    paint(candidate_card(rank, cand, is_top))
    if cand.evidence:
        with st.expander(f"证据原文 · {len(cand.evidence)} 条"):
            for ev in cand.evidence:
                paint(f'<div class="sr-ev">{esc(ev)}</div>')


def table(headers: list[str], rows: list[list[str]]) -> str:
    """rows 里的单元格需调用方自行转义（部分单元格要放标签 HTML）。"""
    th = "".join(f"<th>{esc(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="sr-tbl"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def render_sources(hits: list[dict]) -> None:
    """展示结论的依据来源 —— 没有来源的 AI 结论不可信。"""
    with st.expander(f"检索依据 · 命中 {len(hits)} 个父块"):
        rows = []
        for i, hit in enumerate(hits, start=1):
            score = hit.get("rerank_score", hit.get("rrf", 0.0))
            routes = "+".join(hit.get("routes", [])) or "—"
            rows.append(
                [
                    f'<span class="num">#{i}</span>',
                    f'<span class="file">{esc(Path(hit["file_path"]).name)}</span>',
                    f'<span class="num">{score:.4f}</span>',
                    f'<span class="path">{esc(routes)}</span>',
                    f'{esc(hit["text"].strip()[:160])}…',
                ]
            )
        paint(table(["#", "来源", "精排分", "召回路径", "片段"], rows))


def ingest_rows(rows: list[dict]) -> None:
    """入库结果统一渲染成表。"""
    data = [
        [
            f'<span class="file">{esc(r["file"])}</span>',
            esc(r["status"]),
            f'<span class="num">{r.get("chunks", 0)}</span>',
            esc(r.get("detail", "")),
        ]
        for r in rows
    ]
    paint(table(["文件", "状态", "块数", "备注"], data))


# ---------------------------------------------------------------- 侧边栏
def render_sidebar(pipeline: RAGPipeline) -> None:
    with st.sidebar:
        stats = pipeline.stats()
        kpis = [
            ("简历档案", stats["mongo_resumes"], True),
            ("文本块", stats["milvus_chunks"], False),
            ("ES 索引", stats["es_chunks"], False),
            ("MySQL 登记", stats["mysql_registered"], False),
        ]
        paint(
            '<div class="sr-sec">知识库概览</div><div class="sr-kpis">'
            + "".join(
                f'<div class="sr-kpi{" hl" if hl else ""}">'
                f'<div class="v">{esc(v)}</div><div class="l">{esc(k)}</div></div>'
                for k, v, hl in kpis
            )
            + "</div>"
        )

        paint('<div class="sr-sec">入库</div>')

        # 上一轮入库的结果（rerun 会清空界面，放 session_state 里带过来）
        flash = st.session_state.pop("flash", None)
        if flash is not None:
            ingest_rows(flash)

        uploads = st.file_uploader("上传简历 PDF", type=["pdf"], accept_multiple_files=True)
        if uploads and st.button("写入知识库", type="primary", use_container_width=True):
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            results = []
            for f in uploads:
                dest = UPLOAD_DIR / f.name
                dest.write_bytes(f.getbuffer())
                results.append(pipeline.ingest(dest))
            st.session_state["flash"] = results
            st.rerun()

        directory = st.text_input("目录批量入库", value=str(TEST_DIR))
        if st.button("扫描该目录", use_container_width=True):
            if not Path(directory).is_dir():
                st.error("目录不存在")
            else:
                st.session_state["flash"] = pipeline.ingest_directory(directory)
                st.rerun()

        docs = pipeline.list_documents()
        with st.expander(f"已入库简历 · {len(docs)}"):
            if not docs:
                paint('<div class="sr-note">还没有简历入库</div>')
            for doc in docs:
                paint(
                    '<div class="sr-doc">'
                    f'<span>{esc(Path(doc["file_path"]).name)}</span>'
                    f'<span class="t">{esc(doc["char_count"])} 字 · {esc(doc["timestamp"][11:19])}</span>'
                    "</div>"
                )


# ---------------------------------------------------------------- 主区域
def render_match_tab(pipeline: RAGPipeline) -> None:
    with st.form("match_form"):
        jd = st.text_area("岗位要求", value=DEFAULT_JD, height=230, label_visibility="collapsed")
        left, right = st.columns([3, 1])
        match_top_k = left.slider("评估候选人数", 1, 10, 5)
        submitted = right.form_submit_button("开始匹配", type="primary", use_container_width=True)

    if not submitted:
        return
    if not jd.strip():
        st.error("请填写岗位要求")
        return

    with st.spinner("三路召回 → RRF → 精排 → 模型评估 ..."):
        started = time.perf_counter()
        try:
            result = pipeline.match(jd, top_k=match_top_k)
        except Exception as exc:
            st.error(f"匹配失败：{type(exc).__name__}: {exc}")
            return
        elapsed = time.perf_counter() - started

    report = result["report"]
    if not report.candidates:
        st.info("未检索到候选人档案，请先在左侧入库简历。")
        return

    top = report.candidates[0].match_score or 0
    paint(
        '<div class="sr-chips" style="margin:0 0 1rem">'
        f'<span class="sr-chip">候选人 {len(report.candidates)}</span>'
        f'<span class="sr-chip v">最高 {int(top)} 分</span>'
        f'<span class="sr-chip">命中父块 {len(result["hits"])}</span>'
        f'<span class="sr-chip v">上下文 {result["context_chars"]} 字</span>'
        f'<span class="sr-chip">耗时 {elapsed:.1f}s</span>'
        "</div>"
    )
    for rank, cand in enumerate(report.candidates, start=1):
        render_candidate(rank, cand, is_top=rank == 1)
    render_sources(result["hits"])


def render_recall_tab(pipeline: RAGPipeline) -> None:
    query = st.text_input("查询词", value="熟悉 Milvus 和 Elasticsearch 的候选人")
    recall_top_k = st.slider("返回条数", 1, 10, 5, key="recall_top_k")
    if not st.button("检索", type="primary"):
        return
    if not query.strip():
        st.error("请填写查询词")
        return

    with st.spinner("三路召回 → RRF → 精排 ..."):
        hits = pipeline.search(query, top_k=recall_top_k)

    rows = [
        [
            f'<span class="num">#{i}</span>',
            f'<span class="num">{hit.get("rerank_score", 0.0):.4f}</span>',
            f'<span class="path">{esc("+".join(hit.get("routes", [])) or "—")}</span>',
            f'<span class="file">{esc(Path(hit["file_path"]).name)}</span>',
            esc(hit["text"].strip()[:70]),
        ]
        for i, hit in enumerate(hits, start=1)
    ]
    paint(table(["排名", "精排分", "召回路径", "来源", "片段"], rows))


def main() -> None:
    st.set_page_config(page_title="智能简历推荐系统", page_icon="📄", layout="wide")
    paint(f"<style>{CSS}</style>")
    render_hero()

    pipeline = get_pipeline()
    render_sidebar(pipeline)

    tab_match, tab_recall = st.tabs(["岗位匹配", "召回预览"])
    with tab_match:
        render_match_tab(pipeline)
    with tab_recall:
        render_recall_tab(pipeline)


main()
