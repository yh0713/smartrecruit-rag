"""把仓库根目录加进 sys.path，让 tests 能直接 import 根目录模块。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
