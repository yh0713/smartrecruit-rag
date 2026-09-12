"""document_processor.py —— PDF → 归一化 → 层级切块 → list[Chunk]

纯文本处理，不连库、不调 LLM。Chunk 是后续所有模块的数据契约。
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------------------- 切块参数
PARENT_SIZE = 1000      # 父块：命中后回喂给 LLM 的"完整上下文"
CHILD_SIZE = 400        # 子块：真正被向量化、参与检索的粒度
OVERLAP = 50            # 子块重叠，防止句子被切在边界上丢语义

# 中文优先的分隔符，从"语义最强"到"最弱"逐级回退
SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]

# 简历 PDF 尾部的模板说明文字：从首个命中处截断（比 re.sub 安全，模式写宽会误删正文）
NOISE_PATTERNS = [
    r"使用方法[:：]",
    r"如果你想要[，,]\s*我还可以帮你",
]

# 任意空白但**除换行**（含全角空格 \u3000、不换行空格 \xa0 等）→ 折叠成一个半角空格
_INLINE_WS = re.compile(r"[^\S\n]+")
# 3 个及以上连续换行 → 2 个
_MULTI_NL = re.compile(r"\n{3,}")


@dataclass
class Chunk:
    """一个子块 + 回溯父块所需的全部信息。"""

    id: str              # f"{doc_hash}_p{pi}_c{ci}"，全局唯一，直接当 Milvus 主键
    text: str            # 子块正文（会被向量化）
    parent_id: str       # f"{doc_hash}_p{pi}"
    parent_content: str  # 父块全文（命中后回喂 LLM 的"完整上下文"）
    doc_hash: str        # sha256(归一化全文)，64 位十六进制
    file_path: str       # 来源 PDF 路径
    timestamp: str       # ISO 格式；同一批切块共用一个时间戳


class DocumentProcessor:
    """把一份简历 PDF 加工成可入库的 Chunk 列表。"""

    def __init__(
        self,
        parent_size: int = PARENT_SIZE,
        child_size: int = CHILD_SIZE,
        overlap: int = OVERLAP,
    ) -> None:
        self.parent_size = parent_size
        self.child_size = child_size
        self.overlap = overlap

        # splitter 只是配置，构造时建好，让 split() 保持干净
        self.parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=0,          # 父块之间不重叠，避免同一段话被回喂两次
            separators=SEPARATORS,
        )
        self.child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=overlap,    # 子块之间重叠，保住被切断的句子
            separators=SEPARATORS,
        )

    # ------------------------------------------------------------ 静态工具
    @staticmethod
    def normalize(text: str) -> str:
        """统一换行与空白 —— 指纹稳定的前提：同一简历经不同导出器，字节可能完全不同。"""
        text = text.replace("\r\n", "\n").replace("\r", "\n")   # 统一换行符
        text = _INLINE_WS.sub(" ", text)                        # 折叠行内空白
        text = _MULTI_NL.sub("\n\n", text)                      # 折叠连续空行
        return "\n".join(line.strip() for line in text.split("\n")).strip()

    @staticmethod
    def strip_noise(text: str) -> str:
        """从最靠前的噪音标记处截断，丢弃其后全部（噪音固定出现在文末）。"""
        cut = len(text)
        for pattern in NOISE_PATTERNS:
            match = re.search(pattern, text)
            if match:
                cut = min(cut, match.start())
        return text[:cut]

    @staticmethod
    def fingerprint(text: str) -> str:
        """sha256(归一化文本)，64 位。Milvus 的 doc_hash 字段 max_length 必须是 64（讲义 md5 是 32）。"""
        normalized = DocumentProcessor.normalize(text)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()

    # ------------------------------------------------------------ 主流程
    def load(self, pdf_path: str) -> str:
        """读 PDF → 拼成整篇 → 去噪音 → 归一化。

        ① 先拼成整篇 str 再切：直接 split_documents() 会按页边界硬断父块。
        ② 拼页用 "\\n" 而非 "\\n\\n"：按 "\\n\\n" 拼时整篇被当作 [第1页, 第2页] 两个片段，
           第1页超长触发递归切分，而递归尾块不再与后续片段合并 —— 页边界仍是隐形墙；
           用 "\\n" 拼则全文退化为行序列，不触发递归，合并贯穿全文。
        """
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"{pdf_path} 文件不存在")

        pages = PyPDFLoader(str(path)).load()
        raw = "\n".join(page.page_content for page in pages)
        return self.normalize(self.strip_noise(raw))

    def split(self, text: str, doc_hash: str, file_path: str) -> list[Chunk]:
        """两层切块：父块 1000 / 子块 400 / overlap 50。入参为整篇文本，用 split_text()。"""
        timestamp = datetime.now().isoformat(timespec="seconds")   # 整批共用一个时间戳
        chunks: list[Chunk] = []

        parent_texts = self.parent_splitter.split_text(text)
        for pi, parent_text in enumerate(parent_texts):
            parent_id = f"{doc_hash}_p{pi}"
            for ci, child_text in enumerate(self.child_splitter.split_text(parent_text)):
                chunks.append(
                    Chunk(
                        id=f"{doc_hash}_p{pi}_c{ci}",
                        text=child_text,
                        parent_id=parent_id,
                        parent_content=parent_text,
                        doc_hash=doc_hash,
                        file_path=file_path,
                        timestamp=timestamp,
                    )
                )
        return chunks

    def process(self, pdf_path: str) -> list[Chunk]:
        """一站式入口：load → fingerprint → split。供 VectorStore 调用。"""
        return self.process_with_text(pdf_path)[1]

    def process_with_text(self, pdf_path: str) -> tuple[str, list[Chunk]]:
        """同 process，但把整篇文本一并返回 —— Mongo 存原文需要 text 与 chunks 同时可得，
        一次处理避免 PDF 被读两遍。
        """
        text = self.load(pdf_path)
        doc_hash = self.fingerprint(text)
        return text, self.split(text, doc_hash, str(pdf_path))


if __name__ == "__main__":
    # 临时自检入口：整个流水线接通后可以删掉
    target = Path(__file__).resolve().parent.parent / "个人简历测试.pdf"

    processor = DocumentProcessor()
    text = processor.load(str(target))
    chunks = processor.process(str(target))
    parent_ids = {c.parent_id for c in chunks}

    print(f"清洗后正文字数 : {len(text)}")
    print(f"doc_hash       : {chunks[0].doc_hash}")
    print(f"父块 {len(parent_ids)} 个 / 子块 {len(chunks)} 个")
    print(f"平均子块长度   : {sum(len(c.text) for c in chunks) / len(chunks):.0f} 字\n")

    first = chunks[0]
    print(f"[首个父块] {first.parent_id}   长度 {len(first.parent_content)}")
    print(f"  被切成 {sum(1 for c in chunks if c.parent_id == first.parent_id)} 个子块")
    print(f"  {first.parent_content[:80]}...\n")
    print(f"[首个子块] {first.id}   长度 {len(first.text)}")
    print(f"  {first.text[:80]}...")
