"""vector_store 纯函数单元测试（只测函数逻辑，不连任何数据库 / 外部服务）。"""
import pytest

from vector_store import rrf_fuse, resolve_rerank_url


def _hit(id_: str, score: float = 1.0) -> dict:
    return {"id": id_, "score": score, "text": f"text-{id_}"}


# ---------------------------------------------------------------- rrf_fuse
def test_rrf_fuse_doc_in_both_routes_wins():
    """两路都命中的文档排名高于任一单路第一名 —— RRF 的核心价值。"""
    milvus = [_hit("a"), _hit("b"), _hit("c")]
    es = [_hit("b"), _hit("d")]
    fused = rrf_fuse([milvus, es], k=60)

    assert fused[0]["id"] == "b"
    assert fused[0]["rrf"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0]["rrf"] > 1 / 61      # 高于单路 rank1 的 a


def test_rrf_fuse_routes_recorded():
    fused = rrf_fuse([[_hit("a")], [_hit("a"), _hit("b")]])
    by_id = {row["id"]: row for row in fused}
    assert by_id["a"]["routes"] == ["milvus_hybrid", "es_bm25"]
    assert by_id["b"]["routes"] == ["es_bm25"]


def test_rrf_fuse_keeps_first_hit_fields():
    """同一 id 两路字段不同时，保留首次出现（更高排名那路）的字段。"""
    fused = rrf_fuse([[_hit("a", score=0.9)], [_hit("a", score=3.5)]])
    assert fused[0]["score"] == 0.9
    assert fused[0]["rrf"] == pytest.approx(2 / 61)


def test_rrf_fuse_empty():
    assert rrf_fuse([]) == []


# ---------------------------------------------------------------- resolve_rerank_url
def test_resolve_rerank_url_from_compatible_base(monkeypatch):
    monkeypatch.setenv("RERANK_BASE_URL", "https://x.example/compatible-mode/v1")
    monkeypatch.delenv("RERANK_URL", raising=False)
    assert resolve_rerank_url() == "https://x.example/api/v1/services/rerank/text-rerank/text-rerank"


def test_resolve_rerank_url_explicit_override(monkeypatch):
    monkeypatch.setenv("RERANK_URL", "https://y.example/rerank")
    assert resolve_rerank_url() == "https://y.example/rerank"
