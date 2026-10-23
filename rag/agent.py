"""Agent：用 function calling 让模型自主决定「检索 / 摘要 / 字数统计」，ReAct 循环直到给出最终答案。

支持多轮对话：用滑动窗口保留最近对话，超出的旧轮次压缩成摘要，避免上下文无限膨胀。
支持流式输出：answer_stream 边生成边 yield token；知识问答自动附带图谱上下文（若已构建 graph.json）。
"""
import json

from .config import get_llm_client, get_embed_client, LLM_MODEL, CHROMA_DIR, GRAPH_PATH
from .store import get_collection
from .hybrid import HybridSearcher
from .graph import load_graph
from .tools import build_tool_schemas, execute_tool

SYSTEM_PROMPT = """你是一个个人知识库问答助手，具备工具调用能力。
规则：
1. 知识类问题：先调用 search_knowledge_base 检索，再根据返回内容作答，并在句末用 [1][2] 这类编号标注引用来源；检索不到相关内容就回答"根据我的知识库无法回答这个问题"，不要编造、不要补充资料里没有的内容。
2. 用户要求"总结/摘要"某文档：调用 summarize_document。
3. 用户要求"统计字数/多少字"：调用 count_words。
4. 与知识库无关的闲聊（如"你好"）：直接简短回答，不必调用工具。
5. 全程用中文，回答简洁清晰；可结合前面对话上下文理解指代（如"它"指什么）。"""


class Agent:
    MAX_STEPS = 6          # 单轮最多几次「工具调用 → 再回答」
    WINDOW = 8             # 滑动窗口：最近保留的对话消息条数
    SUMMARY_THRESHOLD = 14  # 历史超过这个条数，就把最旧的压成摘要

    def __init__(self):
        self.llm = get_llm_client()
        self.embed = get_embed_client()
        self.collection = get_collection(CHROMA_DIR)
        self.searcher = HybridSearcher(self.collection, self.embed)
        self.graph = load_graph(GRAPH_PATH)
        self.tools = build_tool_schemas()
        self.history: list[dict] = []   # 多轮对话记忆（user/assistant 成对）
        self.summary = ""               # 被压缩掉的旧轮次摘要
        self.last_answer = ""           # 最近一次回答（流式结束后可用）
        self.last_sources: list[str] = []  # 最近一次回答用到的来源文件
        self.last_trace = None          # 最近一次检索链路的耗时追踪（Tracer 或 None）

    def refresh(self):
        """重新入库后调用：重新拿 collection 并重建检索器与图谱。"""
        self.collection = get_collection(CHROMA_DIR)
        self.searcher = HybridSearcher(self.collection, self.embed)
        self.graph = load_graph(GRAPH_PATH)

    def reset(self):
        """清空多轮对话记忆。"""
        self.history = []
        self.summary = ""

    def _build_messages(self, question: str) -> list[dict]:
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        if self.summary:
            messages.append({"role": "system", "content": f"[历史对话摘要]\n{self.summary}"})
        messages.extend(self.history[-self.WINDOW:])
        messages.append({"role": "user", "content": question})
        return messages

    def _make_ctx(self) -> dict:
        return {"llm": self.llm, "searcher": self.searcher, "graph": self.graph, "sources": []}

    def answer(self, question: str) -> tuple[str, list[str]]:
        """回答一个问题，返回 (答案文本, 用到的来源文件列表)。"""
        messages = self._build_messages(question)
        ctx = self._make_ctx()

        for _ in range(self.MAX_STEPS):
            resp = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
            )
            msg = resp.choices[0].message

            # 没有工具调用 → 模型已给出最终答案
            if not msg.tool_calls:
                answer = msg.content or ""
                self._remember(question, answer)
                self.last_answer, self.last_sources = answer, ctx["sources"]
                self.last_trace = ctx.get("trace")
                return answer, ctx["sources"]

            # 回填 assistant 的 tool_calls 消息
            messages.append(self._tool_calls_message(msg))
            for tc in msg.tool_calls:
                args = self._parse_args(tc.function.arguments)
                result = execute_tool(tc.function.name, args, ctx)
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        answer = "抱歉，处理步骤过多，请换个问法再试。"
        self._remember(question, answer)
        self.last_answer, self.last_sources = answer, ctx["sources"]
        self.last_trace = ctx.get("trace")
        return answer, ctx["sources"]

    def answer_stream(self, question: str):
        """流式回答：逐个 yield 最终答案的 token；工具调用的中间步骤不对外输出。

        完成后可通过 self.last_sources 拿到来源（在 for 循环结束后读取）。
        """
        messages = self._build_messages(question)
        ctx = self._make_ctx()

        for _ in range(self.MAX_STEPS):
            stream = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                tools=self.tools,
                tool_choice="auto",
                temperature=0.3,
                stream=True,
            )
            content_parts: list[str] = []
            calls: dict[int, dict] = {}
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    content_parts.append(delta.content)
                    yield delta.content
                for tc in (delta.tool_calls or []):
                    c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function and tc.function.name:
                        c["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        c["args"] += tc.function.arguments

            # 本步没有工具调用 → 已流式输出最终答案
            if not calls:
                answer = "".join(content_parts)
                self._remember(question, answer)
                self.last_answer, self.last_sources = answer, ctx["sources"]
                self.last_trace = ctx.get("trace")
                return

            # 本步是工具调用：回填 tool_calls、执行工具后继续循环
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["args"]}}
                    for c in calls.values()
                ],
            })
            for c in calls.values():
                args = self._parse_args(c["args"])
                result = execute_tool(c["name"], args, ctx)
                messages.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        answer = "抱歉，处理步骤过多，请换个问法再试。"
        self._remember(question, answer)
        self.last_answer, self.last_sources = answer, ctx["sources"]
        self.last_trace = ctx.get("trace")
        yield answer

    @staticmethod
    def _tool_calls_message(msg) -> dict:
        """把模型返回的 tool_calls 手动构造成 assistant 消息 dict（兼容不同 SDK 版本）。"""
        return {
            "role": "assistant",
            "content": msg.content or "",
            "tool_calls": [
                {"id": tc.id, "type": "function",
                 "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ],
        }

    def _remember(self, question: str, answer: str):
        """记下这一轮问答，超阈值时把旧轮次压缩成摘要。"""
        self.history.append({"role": "user", "content": question})
        self.history.append({"role": "assistant", "content": answer})
        if len(self.history) > self.SUMMARY_THRESHOLD:
            self._compress()

    def _compress(self):
        """把窗口之外的旧对话用 LLM 压成一段摘要，替换掉原文。"""
        overflow = self.history[:-self.WINDOW]
        self.history = self.history[-self.WINDOW:]
        convo = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}：{m['content']}" for m in overflow
        )
        try:
            resp = self.llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": "把下面这段对话压缩成 2-3 句中文摘要，保留关键事实与未完成的意图。"},
                    {"role": "user", "content": convo},
                ],
                temperature=0.2,
            )
            self.summary = resp.choices[0].message.content or self.summary
        except Exception:
            pass  # 摘要失败不影响主流程

    @staticmethod
    def _parse_args(raw: str) -> dict:
        """把模型给的 arguments（JSON 字符串）解析成 dict；解析失败返回空 dict。"""
        try:
            return json.loads(raw or "{}")
        except json.JSONDecodeError:
            return {}
