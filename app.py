"""Gradio 网页界面：复用 Agent 做多轮问答，支持上传文档后重新入库。

用法：python app.py
"""
import sys
import shutil
from pathlib import Path

# Windows 下强制 UTF-8 输出，避免中文乱码
for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

import gradio as gr

from rag.agent import Agent
from rag.config import DATA_DIR
from ingest import ingest

agent = Agent()


def respond(message: str, history=None) -> str:
    answer, sources = agent.answer(message)
    names = list(dict.fromkeys(Path(s).name for s in sources if s))
    if names:
        answer += "\n\n📚 来源：" + "、".join(names)
    return answer


def ingest_files(files) -> str:
    if not files:
        return "没有收到文件。"
    saved = []
    for f in files:
        dest = Path(DATA_DIR) / Path(f).name
        shutil.copy(f, dest)
        saved.append(dest.name)
    n = ingest()
    agent.refresh()
    return f"已入库 {len(saved)} 个文件，共 {n} 块，现在可以提问了。"


def reset_chat():
    agent.reset()
    return [], ""


with gr.Blocks(title="个人知识库问答 Agent") as demo:
    gr.Markdown("# 个人知识库问答 Agent\n知识问答 / 文档摘要 / 字数统计 / 多轮追问")

    with gr.Tab("问答"):
        chat = gr.Chatbot(label="对话", height=400)
        msg = gr.Textbox(label="问题", placeholder="输入问题后回车")
        clear = gr.Button("清空对话")

        def _submit(text, history):
            history = history or []
            history.append([text, None])
            return "", history

        def _reply(history):
            history[-1][1] = respond(history[-1][0], history)
            return history

        msg.submit(_submit, [msg, chat], [msg, chat], queue=False).then(_reply, chat, chat)
        clear.click(reset_chat, None, [chat, msg], queue=False)

    with gr.Tab("上传文档"):
        files = gr.File(file_count="multiple", label="选择 PDF / Word / Markdown 文件")
        ingest_btn = gr.Button("入库")
        status = gr.Textbox(label="状态", interactive=False)
        ingest_btn.click(ingest_files, files, status)


if __name__ == "__main__":
    demo.launch()
