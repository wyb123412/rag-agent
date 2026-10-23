"""Agent 工具：定义工具 JSON schema，并实现「检索 / 文档摘要 / 字数统计」三个工具。

工具通过「名称 → 处理函数」分发。处理函数接收 ctx（携带 llm / embed / collection 实例）
和模型传入的参数，返回一段字符串结果（会被当作 tool 消息喂回给模型）。
"""
from pathlib import Path

from .config import DATA_DIR, LLM_MODEL, QUERY_REWRITE
from .load import load_documents
from .graph import graph_retrieve
from .rewrite import hyde_search, multi_search


def list_document_names() -> list[str]:
    """返回 data/samples 下所有文档的文件名（basename），用于注入工具描述。

    只列文件名、不做解析/OCR，避免每次启动都重新识别一遍图片。
    """
    from .load import list_document_paths
    return [Path(p).name for p in list_document_paths(DATA_DIR)]


def _find_document(filename: str) -> dict | None:
    """按 basename 精确匹配，失败再退化为子串匹配；找不到返回 None。"""
    docs = load_documents(DATA_DIR)
    for d in docs:
        if Path(d["path"]).name == filename:
            return d
    matches = [d for d in docs if filename.lower() in Path(d["path"]).name.lower()]
    return matches[0] if len(matches) == 1 else None


def _retrieve(ctx: dict, query: str) -> dict:
    """按 QUERY_REWRITE 配置分派检索方式（hyde / multi / 默认 hybrid+rerank）。"""
    searcher = ctx["searcher"]
    if QUERY_REWRITE == "hyde":
        return hyde_search(searcher, ctx["llm"], searcher.embed, query)
    if QUERY_REWRITE == "multi":
        return multi_search(searcher, ctx["llm"], query)
    return searcher.search(query, mode="hybrid+rerank")


def _search_knowledge_base(ctx: dict, args: dict) -> str:
    """工具1：检索（可查询改写）+ 知识图谱上下文，带编号和来源返回。"""
    query = (args.get("query") or "").strip()
    if not query:
        return "错误：search_knowledge_base 需要非空的 query 参数。"
    r = _retrieve(ctx, query)
    # 记录来源，供最终回答展示引用
    ctx["sources"].extend(m.get("source", "") for m in r["metadatas"])

    parts = []
    if ctx.get("graph"):
        gctx = graph_retrieve(ctx["llm"], ctx["graph"], query)
        if gctx:
            parts.append(gctx)

    lines = []
    for i, (doc, meta) in enumerate(zip(r["documents"], r["metadatas"]), 1):
        src = Path(meta.get("source", "?")).name
        lines.append(f"[{i}]（来源：{src}）{doc}")
    if lines:
        parts.append("\n\n".join(lines))
    return "\n\n".join(parts) if parts else "知识库里没有检索到相关内容。"


def _count_words(ctx: dict, args: dict) -> str:
    """工具2：统计文档字数（去掉空白后的字符数）。不传 filename 统计全部。"""
    filename = (args.get("filename") or "").strip()
    docs = load_documents(DATA_DIR)
    if filename:
        doc = _find_document(filename)
        if doc is None:
            return f"未找到名为「{filename}」的文档。可用文档：{'、'.join(list_document_names())}"
        docs = [doc]

    lines, total = [], 0
    for d in docs:
        n = len("".join(d["text"].split()))  # 去掉所有空白后的字符数，即「字数」
        total += n
        lines.append(f"- {Path(d['path']).name}：{n} 字")
    if filename:
        return lines[0]
    lines.append(f"合计：{total} 字")
    return "\n".join(lines)


def _summarize_document(ctx: dict, args: dict) -> str:
    """工具3：对指定文档生成中文要点摘要（内部再调一次 LLM）。"""
    filename = (args.get("filename") or "").strip()
    if not filename:
        return "错误：summarize_document 需要 filename 参数。可用文档：" + "、".join(list_document_names())
    doc = _find_document(filename)
    if doc is None:
        return f"未找到名为「{filename}」的文档。可用文档：{'、'.join(list_document_names())}"

    name = Path(doc["path"]).name
    resp = ctx["llm"].chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system",
             "content": "你是文档摘要助手。用中文把文档要点分条概括，简洁、忠实原文、不添油加醋，总长不超过 150 字。"},
            {"role": "user", "content": f"请摘要下面这份文档《{name}》：\n\n{doc['text']}"},
        ],
        temperature=0.3,
    )
    return f"《{name}》摘要：\n{resp.choices[0].message.content}"


def build_tool_schemas() -> list[dict]:
    """构造 OpenAI 风格的 tools 定义，并把可用文档名注入到描述里。"""
    names = list_document_names()
    doc_list = "、".join(names) or "（暂无文档）"
    example = names[0] if names else "示例文档.md"
    return [
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "在个人知识库中检索与问题最相关的内容。回答知识类问题时先调用它，再基于返回结果作答。",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "检索关键词或问题"}},
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "summarize_document",
                "description": f"对指定文档生成中文摘要。可用文档：{doc_list}",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": f"要摘要的文档文件名，如 {example}"}},
                    "required": ["filename"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "count_words",
                "description": f"统计文档字数。不传 filename 时统计所有文档；传 filename 统计指定文档。可用文档：{doc_list}",
                "parameters": {
                    "type": "object",
                    "properties": {"filename": {"type": "string", "description": "文档文件名；留空则统计全部"}},
                },
            },
        },
    ]


_TOOLS = {
    "search_knowledge_base": _search_knowledge_base,
    "summarize_document": _summarize_document,
    "count_words": _count_words,
}


def execute_tool(name: str, args: dict, ctx: dict) -> str:
    """执行一个工具并返回字符串结果。未知工具 / 执行异常都转成友好文字，避免中断循环。"""
    handler = _TOOLS.get(name)
    if handler is None:
        return f"未知工具：{name}"
    try:
        return handler(ctx, args or {})
    except Exception as e:
        return f"工具 {name} 执行出错：{e}"
