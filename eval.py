"""评测脚本：对三种检索模式做消融，输出 Hit@5 / Recall@5 / MRR，并对最终模式的答案做忠实度打分。

用法：python eval.py
产出：eval_report.md（消融表 + 逐题忠实度）
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

EVAL_SET = Path(__file__).parent / "data" / "eval_set.json"
K = 5

ANSWER_SYSTEM = """你只能根据下面【参考资料】回答。若资料里有答案，简洁回答并标注 [1][2] 来源；若资料不足以回答，就说"根据我的知识库无法回答这个问题"，不要编造。"""

JUDGE_SYSTEM = """你是 RAG 评测员。判断「回答」是否忠实于「参考资料」：有没有编造资料里没有的内容。
只输出两个词之一：忠实 或 不忠实。"""


def load_eval_set() -> list[dict]:
    return json.loads(EVAL_SET.read_text(encoding="utf-8"))["questions"]


def retrieval_metrics(searcher: HybridSearcher, questions: list[dict], mode: str, ks: tuple = (1, 3, 5)) -> dict:
    """计算多个 k 下的 Recall@k 与 MRR@k。单 gold 时 Recall@k == Hit@k。"""
    n = len(questions)
    out = {}
    for k in ks:
        hits, mrr = 0, 0.0
        for q in questions:
            r = searcher.search(q["question"], mode=mode, top_k=k)
            sources = [Path(m["source"]).name for m in r["metadatas"]]
            gold = q["gold_source"]
            if gold in sources:
                hits += 1
                mrr += 1.0 / (sources.index(gold) + 1)
        out[f"recall@{k}"] = hits / n
        out[f"mrr@{k}"] = mrr / n
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


def judge_faithful(llm, question: str, answer: str, chunks: list[str]) -> str:
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": f"参考资料：\n{'\n'.join(chunks)}\n\n问题：{question}\n\n回答：{answer}"},
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def main():
    llm = get_llm_client()
    embed = get_embed_client()
    collection = get_collection(CHROMA_DIR)
    searcher = HybridSearcher(collection, embed)
    questions = load_eval_set()

    print(f"评测集：{len(questions)} 条问题，top_k={K}\n")

    rows = {}
    for mode in ["vector", "hybrid", "hybrid+rerank"]:
        m = retrieval_metrics(searcher, questions, mode)
        rows[mode] = m
        print(f"[{mode:14s}] Recall@1={m['recall@1']:.2%}  Recall@3={m['recall@3']:.2%}  Recall@5={m['recall@5']:.2%}  MRR@5={m['mrr@5']:.3f}")

    # 忠实度：只在最终模式（hybrid+rerank）上评测
    faithful = 0
    details = []
    for q in questions:
        answer, chunks = generate_answer(llm, searcher, q["question"])
        verdict = judge_faithful(llm, q["question"], answer, chunks)
        ok = "忠实" in verdict
        faithful += ok
        details.append((q["question"], verdict))

    faithfulness = faithful / len(questions)
    print(f"\n忠实度（hybrid+rerank）：{faithfulness:.2%}")

    # 写报告
    lines = [
        "# RAG 评测报告\n\n",
        f"- 评测集：{len(questions)} 条问题（top_k={K}）\n",
        f"- 答案忠实度：{faithfulness:.2%}\n\n",
        "## 消融表\n\n| 模式 | Recall@1 | Recall@3 | Recall@5 | MRR@5 |\n|---|---|---|---|---|\n",
    ]
    for mode in ["vector", "hybrid", "hybrid+rerank"]:
        m = rows[mode]
        lines.append(f"| {mode} | {m['recall@1']:.2%} | {m['recall@3']:.2%} | {m['recall@5']:.2%} | {m['mrr@5']:.3f} |\n")
    lines.append("\n## 逐题忠实度\n\n| 问题 | 判定 |\n|---|---|\n")
    for q_text, verdict in details:
        lines.append(f"| {q_text} | {verdict} |\n")
    Path(__file__).parent.joinpath("eval_report.md").write_text("".join(lines), encoding="utf-8")
    print("\n报告已写入 eval_report.md")


if __name__ == "__main__":
    main()
