"""chain.py —— RAG 链：Prompt + LLM + 结构化输出（LangChain 的主战场）

与讲义差异：讲义用 JsonOutputParser 手工解析 JSON 字符串；本项目用
with_structured_output() 走 function calling，schema 约束在服务端完成
（阶段二练习 4 已验证的路线）。
"""
from __future__ import annotations

import os
from typing import Literal

from dotenv import find_dotenv, load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from vector_store import VectorStore

load_dotenv(find_dotenv())

# ---------------------------------------------------------------- 链参数
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
CONTEXT_TOP_K = 5        # 最终喂给 LLM 的父块数量
CONTEXT_RECALL_K = 10    # 精排前的召回池大小（宽召回 → 精排）
MAX_EVIDENCE = 6         # 每个候选人最多要求几条原文证据


# ---------------------------------------------------------------- 结构化输出契约
#
# Field 的 description 会转成 JSON Schema 字段说明，等于提示词的一部分（练习 4 的结论）
class CandidateMatch(BaseModel):
    """一个候选人的匹配评估。评估对象是"人"：检索命中的是子块，上层按 doc_hash 聚成候选人。"""

    candidate_name: str = Field(
        description="候选人姓名。原文里找不到就填 '未知'，禁止猜测"
    )
    match_score: int = Field(
        ge=0,
        le=100,
        description="岗位匹配度 0-100 的整数。60 分=基本达标，80 分以上=高度匹配",
    )
    experience_years: float | None = Field(
        default=None,
        description="相关工作年限（年）。原文没有明确年份就填 null，不要推算",
    )
    matched_skills: list[str] = Field(
        default_factory=list,
        description="岗位要求中该候选人**确实具备**的技能，逐项对照原文提取",
    )
    missing_skills: list[str] = Field(
        default_factory=list,
        description="岗位要求中该候选人**未体现**的技能。宁缺毋滥，不要为了凑数硬填",
    )
    evidence: list[str] = Field(
        default_factory=list,
        description=(
            "支撑评分的原文摘录，必须**逐字来自候选人档案**，不得改写、不得编造。"
            f"最多 {MAX_EVIDENCE} 条"
        ),
    )
    concerns: list[str] = Field(
        default_factory=list,
        description="风险点或需要面试确认的地方，如履历空档、技术栈偏旧、项目深度不足",
    )
    recommendation: Literal["强烈推荐", "可以考虑", "不推荐"] = Field(
        description="基于岗位要求的最终建议，三选一"
    )
    summary: str = Field(description="不超过 80 字的总体评语")


class MatchReport(BaseModel):
    """一次检索到的**全部候选人**的评估结果。"""

    candidates: list[CandidateMatch] = Field(
        description="逐个候选人的评估。顺序按匹配度从高到低"
    )


# ---------------------------------------------------------------- 提示词
SYSTEM_PROMPT = """你是一位有 10 年经验的资深技术招聘专家，擅长从简历中精准判断候选人与岗位的匹配度。

工作纪律（必须遵守）：
1. **只依据候选人档案作答**。档案里没有的信息一律视为"未体现"，不猜测、不脑补、不补全。
2. evidence 字段必须是档案原文的**逐字摘录**。编造证据比不写证据更严重。
3. match_score 要严格：岗位要求的核心技能缺失就该扣分，不要给出讨好性的高分。
4. 若档案信息明显不足（如只有姓名和联系方式），应给出低分并在 concerns 里说明原因。
5. 一律使用简体中文作答。"""

HUMAN_PROMPT = """## 岗位要求
{job_description}

## 候选人档案
{context}

## 任务
请逐个评估上述每一位候选人与该岗位的匹配度，按匹配度从高到低返回结果。"""


# ---------------------------------------------------------------- 上下文拼装
def group_by_document(hits: list[dict]) -> dict[str, list[dict]]:
    """按 doc_hash 聚合 —— 一个 doc_hash 一位候选人。不分组 LLM 会把同一份简历的多个子块当成多个人。"""
    grouped: dict[str, list[dict]] = {}
    for hit in hits:
        grouped.setdefault(hit["doc_hash"], []).append(hit)
    return grouped


def format_context(hits: list[dict], max_chars_per_parent: int = 1200) -> str:
    """把检索结果拼成按候选人分组的档案文本。

    - 喂 parent_content（父块）而非 text（子块）：子块负责被检索命中，父块负责语义完整。
    - 按 parent_id 去重：同一父块的多个子块命中时不重复拼接，避免白烧 token、评分虚高。
    """
    grouped = group_by_document(hits)
    blocks: list[str] = []

    for idx, (doc_hash, doc_hits) in enumerate(grouped.items(), start=1):
        seen_parents: set[str] = set()
        parts: list[str] = []
        for hit in doc_hits:
            pid = hit["parent_id"]
            if pid in seen_parents:
                continue
            seen_parents.add(pid)
            content = hit["parent_content"][:max_chars_per_parent]
            parts.append(content)

        file_name = os.path.basename(doc_hits[0]["file_path"])
        blocks.append(
            f"### 候选人 {idx}（档案ID {doc_hash[:12]}，来源 {file_name}）\n"
            + "\n---\n".join(parts)
        )

    return "\n\n".join(blocks) if blocks else "（检索未命中任何候选人档案）"


def retrieve_and_format_context(
    query: str,
    store: VectorStore,
    top_k: int = CONTEXT_TOP_K,
    recall_k: int = CONTEXT_RECALL_K,
) -> str:
    """检索 → 格式化。检索走 hybrid_search_with_rerank：三路召回 → 库内 RRF → 跨库 RRF → 精排。"""
    hits = store.hybrid_search_with_rerank(query, top_k=top_k, recall_k=recall_k)
    return format_context(hits)


# ---------------------------------------------------------------- 构建链
def build_llm(
    model: str | None = None,
    temperature: float | None = None,
) -> ChatOpenAI:
    """按 .env 构建 ChatOpenAI。

    三个坑（阶段二都实测过）：参数名是 base_url；网关对 403/429 不自动重试
    （链上挂了 with_retry）；思考型模型拒绝强制 tool_choice，会令
    with_structured_output 报 400，.env 的 LLM_MODEL 是实测支持工具调用的。
    """
    return ChatOpenAI(
        model=model or os.getenv("LLM_MODEL"),
        api_key=os.getenv("LLM_API_KEY"),
        base_url=os.getenv("LLM_BASE_URL"),
        temperature=LLM_TEMPERATURE if temperature is None else temperature,
    )


def build_prompt() -> ChatPromptTemplate:
    """system + human 两段式提示词。"""
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", HUMAN_PROMPT),
        ]
    )


def build_evaluator(
    llm: ChatOpenAI | None = None,
    temperature: float | None = None,
):
    """构建「prompt → LLM → Pydantic」段，**不含检索**。

    调用方通常已自己检索过、且要展示命中来源；检索封在链内会白跑第二次
    embedding + rerank。with_retry：网关对瞬时错误不自动重试。
    """
    prompt = build_prompt()
    structured_llm = (
        (llm or build_llm(temperature=temperature))
        .with_structured_output(MatchReport)
        .with_retry(stop_after_attempt=3)
    )
    return prompt | structured_llm


def build_chain(
    store: VectorStore | None = None,
    llm: ChatOpenAI | None = None,
):
    """完整 LCEL 链（含检索）：岗位要求 → MatchReport。

    第一段字典把输入分流：job_description 原样透传，context 走检索。
    需要自己控制检索（如展示命中来源）时改用 build_evaluator，
    见 rag_pipeline.RAGPipeline.match。
    """
    store = store or VectorStore()

    # 普通 Python 函数不是 LCEL 组件，用 RunnableLambda 包一层才能接进管道
    retrieve = RunnableLambda(
        lambda job_description: retrieve_and_format_context(job_description, store)
    )

    return (
        {"job_description": RunnablePassthrough(), "context": retrieve}
        | build_evaluator(llm)
    )


if __name__ == "__main__":
    # 自检入口：走一遍「岗位要求 → 检索 → 结构化匹配报告」
    SAMPLE_JD = """岗位：RAG / 大模型应用工程师（后端方向）

职责：
1. 负责检索增强生成（RAG）系统的设计与开发，包括文档解析、切块、向量化与检索链路
2. 负责向量数据库与搜索引擎的选型与调优（Milvus / Elasticsearch）
3. 参与大模型应用的后端服务开发（FastAPI）

要求：
1. 3 年以上 Python 后端开发经验
2. 熟悉 LangChain 等大模型应用框架，有 RAG 项目落地经验
3. 熟悉 Milvus、Elasticsearch 等检索组件的原理与调优
4. 熟悉 MySQL、Redis、Docker 等常用后端组件
5. 有大模型结构化输出（function calling / JSON Schema）实践经验者优先"""

    store = VectorStore()
    chain = build_chain(store=store)

    print("=" * 70)
    print("检索 + 上下文拼装（不调 LLM，先看喂进去的是什么）")
    print("=" * 70)
    ctx = retrieve_and_format_context(SAMPLE_JD, store)
    print(ctx[:700] + ("\n...(截断)" if len(ctx) > 700 else ""))
    print(f"\n上下文总长 {len(ctx)} 字")

    print()
    print("=" * 70)
    print("调用 LLM 做结构化匹配评估")
    print("=" * 70)
    report: MatchReport = chain.invoke(SAMPLE_JD)

    for rank, cand in enumerate(report.candidates, start=1):
        print(f"\n#{rank}  {cand.candidate_name}　　匹配度 {cand.match_score}/100"
              f"　　{cand.recommendation}")
        print(f"    年限   : {cand.experience_years}")
        print(f"    已具备 : {'、'.join(cand.matched_skills) or '—'}")
        print(f"    缺失   : {'、'.join(cand.missing_skills) or '—'}")
        print(f"    风险点 : {'；'.join(cand.concerns) or '—'}")
        print(f"    评语   : {cand.summary}")
        for ev in cand.evidence[:3]:
            print(f"    证据   : {ev[:60]}")
