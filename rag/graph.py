"""手写轻量知识图谱（GraphRAG-lite）：用 LLM 抽取「实体-关系」三元组构建邻接表，
检索时做实体匹配 + 一跳邻居扩展，产出图谱上下文与向量/BM25 结果融合。"""
import json
import logging
from pathlib import Path

from .config import LLM_MODEL

logger = logging.getLogger(__name__)

_EXTRACT_SYSTEM = """你是信息抽取助手。从下面文本中抽取命名实体和它们之间的关系。
严格输出 JSON，格式：{"entities": ["实体1", "实体2"], "relations": [["实体1", "关系", "实体2"]]}
规则：实体是文本中出现的人名/地名/技术概念/组织/专有名词；只保留文本中明确提到的关系；没有就输出空列表。只输出 JSON，不要解释。"""


def parse_extraction_json(raw: str) -> dict:
    """健壮解析 LLM 输出的 JSON（去掉 markdown code fence，失败兜底为空）。"""
    s = raw.strip()
    if s.startswith("```"):
        s = s.lstrip("`")
        s = s[s.find("\n") + 1:] if "\n" in s else s
        s = s.rstrip("`").strip()
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a == -1 or b <= a:
            return {"entities": [], "relations": []}
        try:
            data = json.loads(s[a:b + 1])
        except json.JSONDecodeError:
            return {"entities": [], "relations": []}
    return {
        "entities": [str(x) for x in data.get("entities", [])],
        "relations": [[str(y) for y in r] for r in data.get("relations", []) if len(r) >= 3],
    }


def extract_entities_relations(llm, chunk: str) -> dict:
    """对一段文本抽取实体与关系三元组，失败返回空。"""
    try:
        resp = llm.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "system", "content": _EXTRACT_SYSTEM}, {"role": "user", "content": chunk}],
            temperature=0.0,
        )
        return parse_extraction_json(resp.choices[0].message.content or "")
    except Exception as e:
        logger.warning("实体关系抽取失败：%s", e)
        return {"entities": [], "relations": []}


def build_graph(chunks: list[str], llm=None, extract_fn=None) -> dict:
    """对每个切片抽取三元组，聚合成 {entities: {实体: [chunk_id...]}, triples: [(头,关系,尾,chunk_id)]}。"""
    extract = extract_fn or (lambda c: extract_entities_relations(llm, c))
    entities: dict[str, set] = {}
    triples: list[tuple] = []
    for cid, chunk in enumerate(chunks):
        d = extract(chunk)
        for e in d["entities"]:
            entities.setdefault(e, set()).add(cid)
        for head, rel, tail in d["relations"]:
            triples.append((head, rel, tail, cid))
    return {"entities": {k: sorted(v) for k, v in entities.items()}, "triples": triples}


def save_graph(path, graph: dict):
    Path(path).write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")


def load_graph(path) -> dict | None:
    p = Path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _neighbors(graph: dict, entity: str) -> list[tuple[str, str]]:
    """一跳邻居：(关系, 相邻实体)。"""
    out = []
    for head, rel, tail, _cid in graph["triples"]:
        if head == entity:
            out.append((rel, tail))
        elif tail == entity:
            out.append((rel, head))
    return out


def graph_context(graph: dict, query_entities: list[str], top_k: int = 8) -> str:
    """纯函数：实体匹配 + 一跳邻居扩展，返回图谱上下文文本。"""
    entities = graph.get("entities", {})
    matched = [e for q in query_entities for e in entities if q and (q == e or q in e or e in q)]
    matched = list(dict.fromkeys(matched))
    if not matched:
        return ""

    rel_lines = []
    for e in matched:
        for rel, other in _neighbors(graph, e):
            rel_lines.append(f"- {e} --{rel}--> {other}")
    if not rel_lines:
        return ""

    return "\n".join(["【知识图谱上下文】相关实体：" + "、".join(matched)] + rel_lines[:top_k * 2])


def graph_retrieve(llm, graph: dict, question: str, top_k: int = 8) -> str:
    """从问题抽实体，再做图谱检索；无图谱返回空串。"""
    if not graph:
        return ""
    d = extract_entities_relations(llm, question)
    return graph_context(graph, d["entities"], top_k=top_k)
