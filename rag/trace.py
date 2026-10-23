"""轻量可观测性：记录检索链路各步骤的耗时，供命令行展示与调试，零第三方依赖。

在 tools 的检索工具里埋点（检索 / 图谱两段），Agent 结束后可通过 last_trace.summary()
拿到「检索 12.3ms · 图谱 8.1ms · 合计 20.4ms」这类耗时分解。
"""
import time


class TraceStep:
    def __init__(self, name: str):
        self.name = name
        self._t0 = time.perf_counter()
        self.elapsed_ms = 0.0

    def stop(self) -> "TraceStep":
        self.elapsed_ms = (time.perf_counter() - self._t0) * 1000
        return self


class Tracer:
    def __init__(self):
        self.steps: list[TraceStep] = []

    def step(self, name: str) -> TraceStep:
        s = TraceStep(name)
        self.steps.append(s)
        return s

    def summary(self) -> str:
        if not self.steps:
            return ""
        total = sum(s.elapsed_ms for s in self.steps)
        parts = [f"{s.name} {s.elapsed_ms:.1f}ms" for s in self.steps]
        return " · ".join(parts) + f" · 合计 {total:.1f}ms"
