"""练习 5：文档指纹与去重（纯标准库 + pypdf，不依赖 LLM）

三条能力：
  1. 内容指纹判重  —— 归一化正文取 sha256，抓"字节完全相同"的文档
  2. 增量判重      —— 登记表落盘 JSON，第二次跑同一批文件直接判"已处理"
  3. 近重复检测    —— hash 抓不到"改了几个字"的版本，改用相似度比对

用法：
    python test_dedup.py            # 正常跑（登记表累加）
    python test_dedup.py --reset    # 先清空登记表再跑
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import re
import sys
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path

from probe_03_split import load_pdf

EXPLORE_DIR = Path(__file__).resolve().parent        # Smartrecruit\explore
CODE_DIR = EXPLORE_DIR.parent                        # Smartrecruit\
PROJECT_ROOT = CODE_DIR.parent                       # 项目根（放测试 PDF 的地方）
DATA_DIR = CODE_DIR / "data"                         # 已进 .gitignore，运行时产物都放这
REGISTRY_PATH = DATA_DIR / "hash_registry.json"
SCAN_DIRS = [PROJECT_ROOT, DATA_DIR / "testdata"]

SUPPORTED_SUFFIX = (".pdf", ".txt")
NEAR_DUP_THRESHOLD = 0.95
TOP_N = 5

_WS = re.compile(r"[ \t\u3000]+")
_MULTI_NL = re.compile(r"\n{3,}")


# ---------------- 1. 归一化：把"人眼相同"变成"字节相同" ----------------
def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")   # 统一换行
    text = _WS.sub(" ", text)                               # 折叠行内空白（含全角空格）
    text = _MULTI_NL.sub("\n\n", text)                      # 折叠连续空行
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def fingerprint(text: str, algo: str = "sha256") -> str:
    """对归一化后的正文取指纹 —— 注意不是对文件字节取"""
    digest = hashlib.new(algo)
    digest.update(normalize(text).encode("utf-8"))
    return digest.hexdigest()


# ---------------- 2. 登记表：模拟 MySQL 里那张 hash 表 ----------------
@dataclass
class HashRegistry:
    table: dict[str, list[str]] = field(default_factory=dict)

    def add(self, fp: str, source: str) -> bool:
        """登记一条来源；返回它是否是该指纹的第一个来源"""
        bucket = self.table.setdefault(fp, [])
        if source in bucket:          # 同一文件重复登记不算新来源
            return False
        bucket.append(source)
        return len(bucket) == 1

    def duplicates(self) -> dict[str, list[str]]:
        return {fp: sources for fp, sources in self.table.items() if len(sources) > 1}

    def dump(self, path: str | Path = REGISTRY_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)   # 输出产物先保证目录存在
        path.write_text(
            json.dumps(self.table, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path = REGISTRY_PATH) -> HashRegistry:
        path = Path(path)
        if not path.exists():
            return cls()
        return cls(table=json.loads(path.read_text(encoding="utf-8")))


# ---------------- 读取文件 → 整篇正文 ----------------
def read_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        # load_pdf 会把"共 X 页"打到 stdout，这里临时吞掉，保证本脚本输出干净
        with contextlib.redirect_stdout(io.StringIO()):
            pages = load_pdf(str(path))
        return "\n".join(page.page_content for page in pages)
    return path.read_text(encoding="utf-8", errors="ignore")


def collect_files(dirs: list[Path]) -> list[Path]:
    found: list[Path] = []
    seen: set[str] = set()
    for directory in dirs:
        if not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_SUFFIX:
                continue
            key = str(path.resolve())          # 两个目录可能扫到同一个文件
            if key not in seen:
                seen.add(key)
                found.append(path)
    return found


def source_name(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pad(text: str, width: int = 34) -> str:
    """按显示宽度补齐：中日韩字符占 2 列，否则中英混排必然错位"""
    shown = sum(2 if ord(ch) > 0x2E80 else 1 for ch in text)
    return text + " " * max(1, width - shown)


# ---------------- 3. 近重复：hash 抓不到，只能靠相似度 ----------------
def similarity(a: str, b: str) -> float:
    # autojunk 必须显式关掉：默认只在序列长度 >= 200 时生效，
    # 它会把"出现频率 > 1% 的元素"当噪声丢掉 —— 中文长文本下会让 ratio 严重失真
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def report_near_duplicates(texts: dict[str, str], threshold: float = NEAR_DUP_THRESHOLD) -> None:
    """texts: 指纹 -> 正文（传进唯一文档即可）"""
    items = list(texts.items())
    if len(items) < 2:
        print("\n近重复检测：唯一文档不足 2 篇，跳过")
        return

    pairs = [
        (similarity(items[i][1], items[j][1]), items[i][0], items[j][0])
        for i in range(len(items))
        for j in range(i + 1, len(items))
    ]
    pairs.sort(reverse=True)

    print(f"\n近重复检测（阈值 {threshold}，只列最相似的 {min(TOP_N, len(pairs))} 对）：")
    for ratio, fp_a, fp_b in pairs[:TOP_N]:
        flag = "疑似近重复 ⚠" if ratio >= threshold else "视为不同文档"
        print(f"  {ratio:.4f}  {fp_a[:8]} vs {fp_b[:8]}  {flag}")


# ---------------- 主流程 ----------------
def main() -> None:
    if "--reset" in sys.argv:
        REGISTRY_PATH.unlink(missing_ok=True)
        print(f"已清除登记表：{REGISTRY_PATH}\n")

    files = collect_files(SCAN_DIRS)
    if not files:
        print("没有扫到可处理的文件，扫描目录：")
        for directory in SCAN_DIRS:
            print(f"  {directory}")
        return

    registry = HashRegistry.load(REGISTRY_PATH)
    # 关键：先把"上一轮就存在"的指纹快照下来。
    # 否则本轮一边扫描一边登记，后面的文件去查 is_processed 会被刚写进去的记录污染，
    # 把"本次扫描内重复"误判成"历史上已处理过"。
    pre_existing = set(registry.table)

    print("扫描目录：" + " , ".join(str(d) for d in SCAN_DIRS))
    print(f"登记表历史指纹 {len(pre_existing)} 条 / 近重复阈值 {NEAR_DUP_THRESHOLD}\n")

    texts: dict[str, str] = {}
    new_count = in_run_dup = history_hits = 0
    for path in files:
        name = source_name(path)
        try:
            text = normalize(read_text(path))
        except Exception as exc:                       # 单个文件坏掉不该整批中断
            print(f"{pad(path.name)} 读取失败：{type(exc).__name__}: {exc}")
            continue

        fp = fingerprint(text)
        known = fp in pre_existing
        first_source = registry.add(fp, name)

        if not known and first_source:
            tag, new_count = "新文档 → 入库", new_count + 1
        elif not known:
            tag, in_run_dup = "本次重复 → 跳过", in_run_dup + 1
        elif first_source:
            tag = "同内容新文件 → 仅登记来源"
        else:
            tag, history_hits = "已处理过 → 跳过", history_hits + 1
        print(f"{pad(path.name)} {fp[:12]}  {tag}")
        texts.setdefault(fp, text)

    unique = len(registry.table)
    dup_groups = registry.duplicates()
    print(f"\n扫描 {len(files)} 个文件 / 唯一指纹 {unique} 个 / 重复组 {len(dup_groups)} 组")
    print(f"新文档 {new_count} 个 / 本次内重复 {in_run_dup} 个 / 命中历史登记表 {history_hits} 个")
    for fp, sources in dup_groups.items():
        print(f"  {fp[:12]} ← {sources}")

    report_near_duplicates(texts)

    registry.dump()
    print(f"\n登记表已写入 {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
