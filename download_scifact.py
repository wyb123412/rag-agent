"""下载 BEIR scifact 数据集到 data/beir/scifact，供 beir_eval.py 评测。

来源：BEIR 官方服务器（UKP TU Darmstadt）。国内直连 HuggingFace 常被墙，
官方服务器通常可直连；zip 内已是 corpus.jsonl / queries.jsonl / qrels/test.tsv
的标准格式，无需再转换。

用法：python download_scifact.py
"""
import sys
import zipfile
from pathlib import Path

import requests

# Windows 下强制 UTF-8 输出，避免中文/emoji 乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
BEIR_DIR = Path(__file__).parent / "data" / "beir"
DEST = BEIR_DIR / "scifact"


def _extract_all(zip_path: Path):
    """安全解压到 data/beir/：逐个校验成员路径，防止 zip-slip（路径穿越）。"""
    beir_root = BEIR_DIR.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.infolist():
            dest = (BEIR_DIR / member.filename).resolve()
            if not dest.is_relative_to(beir_root):
                print(f"⚠️  跳过可疑路径：{member.filename}")
                continue
            zf.extract(member, BEIR_DIR)
    # 若解压后多出一层顶层目录（如 scifact/），直接就是目标；若文件散在根目录，手动规整
    if (DEST / "corpus.jsonl").exists():
        return
    # 文件可能直接落在 data/beir/ 根目录
    for name in ("corpus.jsonl", "queries.jsonl"):
        src = BEIR_DIR / name
        if src.exists():
            src.replace(DEST / name)
    src_qrels = BEIR_DIR / "qrels"
    if src_qrels.exists() and not (DEST / "qrels").exists():
        src_qrels.replace(DEST / "qrels")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    if (DEST / "corpus.jsonl").exists() and (DEST / "qrels").exists():
        print(f"✅ scifact 已存在：{DEST}")
        return

    print(f"⬇️  下载 {URL} …")
    zip_path = BEIR_DIR / "scifact.zip"
    with requests.get(URL, stream=True, timeout=180) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 1024):
                f.write(chunk)
    print(f"   已下载 {zip_path.stat().st_size / 1e6:.2f} MB，解压中…")

    _extract_all(zip_path)
    zip_path.unlink()

    if (DEST / "corpus.jsonl").exists() and (DEST / "qrels").exists():
        print("✅ 完成，目录结构：")
        for p in sorted(DEST.rglob("*")):
            print("   ", p.relative_to(DEST))
    else:
        print("⚠️  解压后未找到预期文件，请检查 data/beir/ 目录。")


if __name__ == "__main__":
    main()
