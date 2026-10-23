"""生成质量评测（RAGAS 风格，手写 LLM-as-Judge，零额外依赖）。

在「检索 → 生成」链路里，检索指标（Recall/MRR/nDCG）只说明「有没有召回对」，
这里的三个指标说明「召回到之后，答得对不对」：
- 忠实度（faithfulness）       回答有没有编造资料里没有的内容
- 答案相关性（answer relevancy）回答有没有切题
- 上下文精度（context precision）召回的切片里有几条真正相关
"""
import json

from .config import LLM_MODEL

_FAITHFUL_SYSTEM = ("你是 RAG 评测员。判断「回答」是否忠实于「参考资料」：有没有编造资料里没有的内容。"
                    "只输出两个词之一：忠实 或 不忠实。")
_RELEVANCY_SYSTEM = ("你是 RAG 评测员。判断「回答」是否直接、切题地回应了「问题」。"
                     "只输出两个词之一：相关 或 不相关。")
_PRECISION_SYSTEM = ("你是检索评测员。逐条判断每条【资料】是否与【问题】相关（包含能回答问题的信息）。"
                     "输出一个 JSON 数组，长度与资料条数一致，1 表示相关、0 表示不相关，如 [1,0,1]。只输出数组。")


def _chat(llm, system: str, user: str, temperature: float = 0.0) -> str:
    resp = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=temperature,
    )
    return (resp.choices[0].message.content or "").strip()


def judge_faithfulness(llm, question: str, answer: str, chunks: list[str]) -> str:
    refs = "\n\n".join(chunks)
    return _chat(llm, _FAITHFUL_SYSTEM, f"参考资料：\n{refs}\n\n问题：{question}\n\n回答：{answer}")


def judge_answer_relevancy(llm, question: str, answer: str) -> str:
    return _chat(llm, _RELEVANCY_SYSTEM, f"问题：{question}\n\n回答：{answer}")


def judge_context_precision(llm, question: str, chunks: list[str]) -> str:
    refs = "\n".join(f"[{i + 1}] {c}" for i, c in enumerate(chunks))
    return _chat(llm, _PRECISION_SYSTEM, f"问题：{question}\n\n资料：\n{refs}")


def parse_binary(text: str, positive: str) -> bool:
    """判定二元词（如 相关/忠实）：命中 positive 且未命中其否定式（不相关/不忠实）。"""
    return positive in text and ("不" + positive) not in text


def parse_int_list(text: str, n: int) -> list[int]:
    """解析 LLM 返回的 0/1 数组；解析失败返回空列表，长度不足补 0、超出截断。"""
    s = text.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        s = s[s.find("\n") + 1:] if "\n" in s else s
        s = s.rstrip("`").strip()
    try:
        arr = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("["), s.rfind("]")
        if a == -1 or b <= a:
            return []
        try:
            arr = json.loads(s[a:b + 1])
        except json.JSONDecodeError:
            return []
    if not isinstance(arr, list):
        return []
    out = [1 if x else 0 for x in arr]
    return (out + [0] * n)[:n]
