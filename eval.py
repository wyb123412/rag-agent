"""评测脚本：对多种检索模式做消融，输出「切片级」Recall@k / MRR，并对最终模式的答案做生成质量打分。

黄金标准有两级：
- gold_text  ：答案所在切片里出现的标志性短语（切片级，衡量「召回的是不是对的那一段」）
- gold_source：答案所在文件名（文件级，衡量「对不对文件」，更宽松）

用法：python eval.py
产出：eval_report.md（消融表 + 难例子集 + 逐题生成质量）
"""
import json
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出，避免中文乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rag.config import get_llm_client, get_embed_client, LLM_MODEL, CHROMA_DIR
from rag.store import get_collection
from rag.hybrid import HybridSearcher
from rag.gen_eval import (judge_faithfulness, judge_answer_relevancy,
                          judge_context_precision, parse_binary, parse_int_list)

EVAL_SET = Path(__file__).parent / "data" / "eval_set.json"
K = 5
MODES = ["vector", "hybrid", "hybrid+rerank", "hyde+hybrid+rerank", "multi+hybrid+rerank"]

ANSWER_SYSTEM = """你只能根据下面【参考资料】回答。若资料里有答案，简洁回答并标注 [1][2] 来源；若资料不足以回答，就说"根据我的知识库无法回答这个问题"，不要编造。"""


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))["questions"]


def _retrieve_for_mode(searcher, llm, question: str, mode: str, k: int) -> dict:
    if mode == "hyde+hybrid+rerank":
        from rag.rewrite import hyde_search
        return hyde_search(searcher, llm, searcher.embed, question, mode="hybrid+rerank", top_k=k)
    if mode == "multi+hybrid+rerank":
        from rag.rewrite import multi_search
        return multi_search(searcher, llm, question, mode="hybrid+rerank", top_k=k)
    return searcher.search(question, mode=mode, top_k=k)


def _hit_index(documents: list[str], gold_text: str) -> int | None:
    """返回第一个含 gold_text 的切片下标（0 起）；无 gold_text 或无命中返回 None。"""
    if not gold_text:
        return None
    for i, d in enumerate(documents):
        if gold_text in d:
            return i
    return None


def compute_metrics(questions: list[dict], results: list[dict], ks: tuple = (1, 3, 5)) -> dict:
    """基于一次性 top-K 检索结果，切出前 k 名计算 Recall@k（切片级 + 文件级）与 MRR@k。"""
    n = len(questions)
    out = {}
    for k in ks:
        hit = mrr = doc_hit = 0.0
        for q, r in zip(questions, results):
            docs = r["documents"][:k]
            metas = r["metadatas"][:k]
            idx = _hit_index(docs, q.get("gold_text", ""))
            if idx is not None:
                hit += 1
                mrr += 1.0 / (idx + 1)
            sources = [Path(m["source"]).name for m in metas]
            if q["gold_source"] in sources:
                doc_hit += 1
        out[f"recall@{k}"] = hit / n          # 切片级：命中黄金切片
        out[f"mrr@{k}"] = mrr / n
        out[f"recall_doc@{k}"] = doc_hit / n  # 文件级：命中黄金文件
    return out


def build_prompt(question: str, chunks: list[str]) -> str:
    refs = "\n\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return f"【参考资料】\n{refs}\n\n【问题】\n{question}"


def generate_answer(llm, searcher: HybridSearcher, question: str):
    r = searcher.search(question, mode="hybrid+rerank", top_k=K)
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": build_prompt(question, r["documents"])},
        ],
        temperature=0.3,
    )
    return resp.choices[0].message.content, r["documents"]


def _fmt_mode_line(tag: str, m: dict) -> str:
    return (f"[{tag:22s}] 切片Recall@1={m['recall@1']:.2%} @3={m['recall@3']:.2%} @5={m['recall@5']:.2%}  "
            f"MRR@5={m['mrr@5']:.3f}  文件Recall@5={m['recall_doc@5']:.2%}")


def main():
    llm = get_llm_client()
    embed = get_embed_client()
    collection = get_collection(CHROMA_DIR)
    searcher = HybridSearcher(collection, embed)
    questions = load_eval_set()
    hard = [q for q in questions if q.get("difficulty") == "hard"]

    print(f"评测集：{len(questions)} 条（其中难例 {len(hard)} 条），top_k={K}\n")

    rows, hard_rows = {}, {}
    for mode in MODES:
        # 每个模式只检索一次（top_k=K），再切出前 1/3/5 名分别算指标
        results = [_retrieve_for_mode(searcher, llm, q["question"], mode, K) for q in questions]
        rows[mode] = compute_metrics(questions, results)
        print(_fmt_mode_line(mode, rows[mode]))
        if hard:
            hard_results = [r for q, r in zip(questions, results) if q.get("difficulty") == "hard"]
            hard_rows[mode] = compute_metrics(hard, hard_results)
            print(" " * 24 + f"└─ 难例子集：切片Recall@1={hard_rows[mode]['recall@1']:.2%} @5={hard_rows[mode]['recall@5']:.2%}  MRR@5={hard_rows[mode]['mrr@5']:.3f}")

    # 生成质量（RAGAS 风格三指标）：只在最终模式（hybrid+rerank）上评测
    f_ok = ar_ok = cp_ok = 0
    details = []
    for q in questions:
        answer, chunks = generate_answer(llm, searcher, q["question"])
        f = parse_binary(judge_faithfulness(llm, q["question"], answer, chunks), "忠实")
        ar = parse_binary(judge_answer_relevancy(llm, q["question"], answer), "相关")
        cp_list = parse_int_list(judge_context_precision(llm, q["question"], chunks), len(chunks))
        cp = (sum(cp_list) / len(cp_list)) if cp_list else 0.0
        f_ok += f
        ar_ok += ar
        cp_ok += cp
        details.append((q["question"], "忠实" if f else "不忠实",
                        "相关" if ar else "不相关", f"{cp:.0%}"))

    n = len(questions)
    faithfulness, relevancy, precision = f_ok / n, ar_ok / n, cp_ok / n
    print(f"\n生成质量（RAGAS 风格，hybrid+rerank）：忠实度 {faithfulness:.2%} · 答案相关性 {relevancy:.2%} · 上下文精度 {precision:.2%}")

    # 写报告
    lines = [
        "# RAG 评测报告\n\n",
        f"- 评测集：{len(questions)} 条问题（其中难例 {len(hard)} 条，top_k={K}）\n",
        f"- 生成质量：忠实度 {faithfulness:.2%} · 答案相关性 {relevancy:.2%} · 上下文精度 {precision:.2%}\n\n",
        "## 消融表（切片级 Recall —— 命中黄金切片）\n\n",
        "| 模式 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 文件级 Recall@5 |\n|---|---|---|---|---|---|\n",
    ]
    for mode in MODES:
        m = rows[mode]
        lines.append(f"| {mode} | {m['recall@1']:.2%} | {m['recall@3']:.2%} | {m['recall@5']:.2%} | {m['mrr@5']:.3f} | {m['recall_doc@5']:.2%} |\n")
    if hard:
        lines.append("\n## 难例子集（改写/干扰题，更能拉开模式差距）\n\n| 模式 | Recall@1 | Recall@5 | MRR@5 |\n|---|---|---|---|\n")
        for mode in MODES:
            m = hard_rows[mode]
            lines.append(f"| {mode} | {m['recall@1']:.2%} | {m['recall@5']:.2%} | {m['mrr@5']:.3f} |\n")
    lines.append("\n## 逐题生成质量\n\n| 问题 | 忠实度 | 答案相关性 | 上下文精度 |\n|---|---|---|---|\n")
    for q_text, fv, arv, cpv in details:
        lines.append(f"| {q_text} | {fv} | {arv} | {cpv} |\n")
    Path(__file__).parent.joinpath("eval_report.md").write_text("".join(lines), encoding="utf-8")
    print("\n报告已写入 eval_report.md")


if __name__ == "__main__":
    main()
