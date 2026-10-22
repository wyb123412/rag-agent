"""模块 6：问答 —— 用 Agent（工具调用 + ReAct）自主判断「检索回答」还是「调用工具」，CLI 多轮对话。

用法：python chat.py
"""
import sys
from pathlib import Path

# Windows 下强制 UTF-8 输出/输入，避免中文乱码（Git Bash 等终端）
for _s in (sys.stdout, sys.stderr, sys.stdin):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

from rag.agent import Agent


def main():
    agent = Agent()
    print("=== 个人知识库问答 Agent（输入 exit 退出）===\n")
    print("我能：知识问答 / 文档摘要 / 统计字数，直接说你的需求即可。\n")

    while True:
        question = input("你：").strip()
        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            break

        answer, sources = agent.answer(question)

        print(f"\n助手：{answer}")

        seen = set()
        printed = False
        for src in sources:
            if src and src not in seen:
                seen.add(src)
                if not printed:
                    print("\n📚 参考来源：")
                    printed = True
                print(f"   - {Path(src).name}")
        print()


if __name__ == "__main__":
    main()
