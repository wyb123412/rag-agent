"""让 pytest 能找到项目根目录下的 rag 包。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
