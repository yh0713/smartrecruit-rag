import json
import os
import re
import warnings
from datetime import date
from typing import Annotated, Literal

from dotenv import load_dotenv, find_dotenv

# ⚠️ 警告过滤器必须在 import 之前装好。
# 放到 import 之后，probe_03_split 导入时发出的 DeprecationWarning
# 已经被默认过滤器丢掉，simplefilter 就白设了。
warnings.simplefilter("always", DeprecationWarning)

from probe_03_split import load_pdf  # noqa: E402
from langchain_openai import ChatOpenAI  # noqa: E402
from pydantic import BaseModel, ConfigDict, Field, field_validator  # noqa: E402

PDF_PATH = r"D:\CodeLearning\Heima\12_基于人力资源RAG简历智能推荐系统\个人简历测试.pdf"

TODAY = date.today()


# ==========================================================
# 一、抽取模型：只放"原文里确实写着"的信息
#    规则：字段的每个值都必须能在简历原文里找得到，一个字都不用推导。
# ==========================================================
class Project(BaseModel):
    name: Annotated[str, Field(description="项目名称")]
    duties: Annotated[list[str], Field(
        default_factory=list,
        description="具体承担的职责，每条一句话",
    )]
    highlights: Annotated[list[str], Field(
        default_factory=list,
        description="项目的量化成果，每条一句话，必须原文有据",
    )]
    period: Annotated[str | None, Field(
        default=None,
        description="起止时间，原文照录如 '2023.08 - 2023.12'；原文未给就写 null",
    )]


class Education(BaseModel):
    school: Annotated[str, Field(description="学校名称")]
    degree: Annotated[
        Literal["博士", "硕士", "本科", "其他"] | None,
        Field(default=None, description="学历层次"),
    ]
    major: Annotated[str | None, Field(default=None, description="专业")]
    gpa: Annotated[float | None, Field(default=None, description="GPA")]
    period: Annotated[str | None, Field(
        default=None,
        description="起止时间，原文照录如 '2016.09 - 2020.06'；原文未给就写 null",
    )]


class WorkExperience(BaseModel):
    company: Annotated[str, Field(description="公司名称")]
    position: Annotated[str, Field(description="职位名称，原文照录")]
    duties: Annotated[list[str], Field(
        default_factory=list,
        description="岗位职责，每条一句话",
    )]
    period: Annotated[str | None, Field(
        default=None,
        description="起止时间，原文照录如 '2023.07 - 至今'；原文未给就写 null",
    )]


class ResumeExtract(BaseModel):
    """LLM 抽取层：所有字段都能在原文里找到，不含任何需要计算的派生值"""

    model_config = ConfigDict(str_strip_whitespace=True)

    name: Annotated[str, Field(description="候选人姓名")]
    phone: Annotated[str | None, Field(default=None, description="手机号，原文照录")]
    email: Annotated[str | None, Field(default=None, description="邮箱")]
    age: Annotated[int | None, Field(default=None, description="年龄，单位岁")]
    skills: Annotated[list[str], Field(
        default_factory=list,
        description="技能关键词，去重后的短词",
    )]
    educations: Annotated[list[Education], Field(default_factory=list)]
    work_experiences: Annotated[list[WorkExperience], Field(default_factory=list)]
    projects: Annotated[list[Project], Field(default_factory=list)]

    @field_validator("phone", mode="before")
    @classmethod
    def clean_phone(cls, v):
        """清洗而非拒绝：统一号码格式，让下游拿到的都是纯数字"""
        if isinstance(v, str):
            return v.replace(" ", "").replace("-", "")
        return v


# ==========================================================
# 二、业务模型：抽取结果 + 代码算出来的派生字段
# ==========================================================
class Resume(ResumeExtract):
    """注意：years 不在送给模型的 Schema 里，它由 calc_years() 回填"""

    years: float | None = Field(default=None, description="工作年限（年），由代码计算")


# ==========================================================
# 三、派生字段计算：只认 period 字符串，不依赖模型判断
# ==========================================================
_RANGE_SEP = re.compile(r"\s*[-–—~]\s*")
_NOW_WORDS = ("至今", "现在", "今", "present", "now", "current")


def _parse_ym(text: str) -> date | None:
    """把 '2023.07' / '2023-7' / '2023年7月' 解析成 date；拿不到年份就返回 None"""
    m = re.search(r"(\d{4})\D*(\d{1,2})?", text)
    if not m:
        return None
    year = int(m.group(1))
    month = int(m.group(2)) if m.group(2) else 1
    if not 1 <= month <= 12:
        month = 1
    return date(year, month, 1)


def parse_period(period: str | None) -> tuple[date, date] | None:
    """把 '2023.07 - 至今' 解析成 (起, 止)；解析不出返回 None"""
    if not period:
        return None
    parts = [p for p in _RANGE_SEP.split(period) if p.strip()]
    if not parts:
        return None

    start = _parse_ym(parts[0])
    if start is None:
        return None

    if len(parts) >= 2:
        tail = parts[-1].strip().lower()
        end = TODAY if any(w in tail for w in _NOW_WORDS) else _parse_ym(parts[-1])
    else:
        end = None  # 只给了一个时间点，当作"仍在进行"

    return start, end or TODAY


def _months(start: date, end: date) -> int:
    return (end.year - start.year) * 12 + (end.month - start.month)


def calc_years(
    experiences: list[WorkExperience],
    exclude_intern: bool = True,
) -> float | None:
    """按「时间并集」累计工作年限 —— 多段经历重叠时不会重复计算"""
    spans: list[tuple[date, date]] = []
    for w in experiences:
        if exclude_intern and "实习" in (w.position or ""):
            continue
        r = parse_period(w.period)
        if r:
            spans.append(r)

    if not spans:
        return None

    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])

    total = sum(_months(s, e) for s, e in merged)
    return round(total / 12, 1)


# ==========================================================
# 四、主流程
# ==========================================================
def build_llm() -> ChatOpenAI:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("没有读到LLM_API_KEY，检查.env是否在项目根目录")
    return ChatOpenAI(
        model=os.getenv("LLM_MODEL", "qwen-plus"),
        base_url=os.getenv("LLM_BASE_URL"),
        api_key=api_key,
        temperature=0.0,
    )


if __name__ == '__main__':
    load_dotenv(find_dotenv())

    # 1) 模型实际收到的"合同"：只有抽取字段，没有 years
    print(json.dumps(ResumeExtract.model_json_schema(), ensure_ascii=False, indent=2)[:1200])

    # 2) 加载 PDF 转纯文本
    pages = load_pdf(PDF_PATH)
    text = "\n".join(d.page_content for d in pages if d.page_content)

    # 3) 抽取
    # ⚠️ 这个网关会随机对一部分请求误报 403 AccessDenied.Unpurchased
    #    （同一个 key、同一个模型，相隔几秒：一次成功、一次 403），
    #    所以给链路挂上重试。注意：如果真是额度耗尽，5 次都会失败，不会被掩盖。
    extractor = (
        build_llm()
        .with_structured_output(ResumeExtract)
        .with_retry(stop_after_attempt=5, wait_exponential_jitter=True)
    )
    data = extractor.invoke("从下面这段简历文本中抽取结构化信息，没提到的字段留空，不要编造：\n\n" + text)

    # 4) 派生字段由代码算，不问模型
    resume = Resume(
        **data.model_dump(),
        years=calc_years(data.work_experiences, exclude_intern=True),
    )
    years_with_intern = calc_years(data.work_experiences, exclude_intern=False)

    # 5) 结果
    print("\n返回类型:", type(resume).__name__)
    print("姓名        ：", resume.name)
    print("手机号       ：", resume.phone)
    print("年龄        ：", resume.age)
    print("学历        ：", resume.educations[0].degree if resume.educations else None)
    print(f"工作年限    ： {resume.years} 年（不含实习）；含实习 {years_with_intern} 年")

    print("\n工作经历（原文抽取）：")
    for w in resume.work_experiences:
        print(f"  {w.period}  {w.company}  {w.position}  [{len(w.duties)} 条职责]")

    print("\n项目经历（原文抽取）：")
    for p in resume.projects:
        print(f"  {p.name}（{p.period}）{len(p.duties)} 条职责 / {len(p.highlights)} 条成果")

    print("\n技能数量    ：", len(resume.skills))
    print("技能        ：", ",".join(resume.skills))

    print("\nmodel_dump():")
    print(json.dumps(resume.model_dump(), ensure_ascii=False, indent=2))
