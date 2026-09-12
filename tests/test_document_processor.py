"""document_processor 单元测试（纯文本处理，不连库、不调外部服务）。"""
import re

import pytest

from document_processor import PARENT_SIZE, CHILD_SIZE, Chunk, DocumentProcessor


@pytest.fixture(scope="module")
def processor():
    return DocumentProcessor()


# ---------------------------------------------------------------- normalize
def test_normalize_unifies_line_endings_and_whitespace():
    raw = "姓名：张三\r\n技能：\u3000Python　 \n\n\n\n经验：RAG\xa0应用\r"
    out = DocumentProcessor.normalize(raw)
    assert "\r" not in out and "\u3000" not in out and "\xa0" not in out
    assert "\n\n\n" not in out          # 3 个以上连续换行折叠为 2 个
    assert "技能： Python" in out       # 行内空白折叠成半角空格


def test_normalize_makes_fingerprint_stable(processor):
    a = "技能：Python\n经验：RAG"
    b = "  技能：Python \r\n 经验：RAG  "
    assert processor.fingerprint(a) == processor.fingerprint(b)


def test_fingerprint_is_sha256_hex(processor):
    fp = processor.fingerprint("任意文本")
    assert re.fullmatch(r"[0-9a-f]{64}", fp)


# ---------------------------------------------------------------- strip_noise
def test_strip_noise_truncates_at_earliest_marker():
    text = "正文A 使用方法：xxx 如果你想要，我还可以帮你 yyy"
    assert DocumentProcessor.strip_noise(text) == "正文A "


def test_strip_noise_keeps_clean_text():
    text = "一份没有任何噪音的正文"
    assert DocumentProcessor.strip_noise(text) == text


# ---------------------------------------------------------------- split
def test_split_chunk_contract(processor):
    """Chunk 数据契约：id / parent_id / parent_content / doc_hash 的一致性。"""
    chunks = processor.split("字" * 1300, "h" * 64, "x.pdf")
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.doc_hash == "h" * 64
        assert c.id == f"{c.parent_id}_c{c.id.rsplit('_c', 1)[1]}"
        assert len(c.text) <= CHILD_SIZE
        assert len(c.parent_content) <= PARENT_SIZE
    # 同一父块的所有子块共享同一份 parent_content
    by_parent = {}
    for c in chunks:
        by_parent.setdefault(c.parent_id, set()).add(c.parent_content)
    assert all(len(v) == 1 for v in by_parent.values())


def test_split_children_overlap(processor):
    """相邻子块有 overlap，被切断的句子不会整句丢失。"""
    chunks = processor.split("字" * 1200, "h" * 64, "x.pdf")
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.text[:20] in prev.text


# ------------------------------------------------- 跨页屏障回归（load 拼页符的选择）
def _page(tag: str, n_lines: int, chars_per_line: int = 140) -> str:
    """构造一页：每行 140 字、行间单换行，行首有唯一标记便于断言。"""
    lines = [f"{tag}{i:02d}" + "文" * (chars_per_line - 4) for i in range(n_lines)]
    return "\n".join(lines)


def test_page_join_newline_merges_across_pages(processor):
    """拼页用 \\n 而非 \\n\\n 的回归测试。

    \\n\\n 拼页时整篇顶层是 [第1页, 第2页]，第1页超长触发递归切分，
    递归尾块不与第2页合并 → 页边界是隐形墙；\\n 拼页则贪心合并贯穿全文。
    """
    page1 = _page("L", 10)   # 约 1409 字，超 1000 → 触发递归切分
    page2 = _page("P", 3)    # 短页
    tail_of_page1, first_of_page2 = "L09", "P00"

    wrong = processor.parent_splitter.split_text(page1 + "\n\n" + page2)
    right = processor.parent_splitter.split_text(page1 + "\n" + page2)

    assert not any(tail_of_page1 in p and first_of_page2 in p for p in wrong)
    assert any(tail_of_page1 in p and first_of_page2 in p for p in right)
