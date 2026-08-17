"""Loop Agent：顺序预测主线的循环运行时（predict→judge→ingest→evolve ↺）。

循环契约（loop-engineering）：
  goal   在 chrono 刺激流上最大化 judge OA
  act    每次迭代 = 一个刺激单元：ensure situational → predict(仅历史) → judge → ingest GT → evolve
  verify judge_one 的 opinion_alignment_score
  stop   step == n_steps
  escape 单步异常 → 错误行（不计入 OA，--resume gap-fill 重跑）；
         进程级崩溃 → run_with_supervision 进程内看门狗按检查点重启（吸收 bat 职责）

本模块只承载「循环基础设施」：检查点扫描/追加、GraphAgent 工厂、监督重启。
预测决策本身全在 agent_graph（Graph Agent 内层）。
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable


class CheckpointStore:
    """sequential_predictions.jsonl 的 resume / gap-fill 语义（与 run_sequential 原内联逻辑一致）。

    只认「有预测正文且 judge 无错误」的行为完成；失败行（LLM 402/400 等）
    视为未完成，下次进入时从最早缺失步 gap-fill 重跑。
    """

    def __init__(self, pred_path: Path, n_steps: int, resume: bool) -> None:
        self.pred_path = pred_path
        self.n_steps = n_steps
        self.start_step = 0
        self.existing_rows: list[dict[str, Any]] = []
        self.done_steps: set[int] = set()
        if resume and pred_path.exists():
            self.existing_rows = self._load_jsonl(pred_path)
            for r in self.existing_rows:
                try:
                    s = int(r.get("step", -1))
                except (TypeError, ValueError):
                    continue
                if s < 0:
                    continue
                js = r.get("judge_scores") or {}
                if r.get("warmup"):
                    self.done_steps.add(s)
                    continue
                if (r.get("prediction") or "").strip() and not js.get("error") and not r.get("error"):
                    self.done_steps.add(s)
            if self.existing_rows:
                missing = [s for s in range(n_steps) if s not in self.done_steps]
                self.start_step = min(missing) if missing else n_steps
                # Gap-fill semantics restart from the earliest missing step, so
                # truncate any later rows to avoid duplicate step numbers when
                # they are appended again.
                if self.start_step < n_steps:
                    keep = [r for r in self.existing_rows if r.get("step", -1) < self.start_step]
                    if len(keep) != len(self.existing_rows):
                        self.pred_path.write_text(
                            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep),
                            encoding="utf-8",
                        )
        elif pred_path.exists():
            pred_path.unlink()

    @staticmethod
    def _load_jsonl(path: Path) -> list[dict[str, Any]]:
        rows = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
        return rows

    def append(self, row: dict[str, Any]) -> None:
        self.pred_path.parent.mkdir(parents=True, exist_ok=True)
        with self.pred_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_graph_agent(
    *,
    uid: str,
    memory: Any,
    theories: Any,
    deepseek: Any,
    run_state: Path,
    weak_match_threshold: float,
    source_profile: dict[str, Any] | None,
    path_tuning: dict[str, Any] | None,
    fm_budget: dict[str, Any] | None,
    agent_cfg: dict[str, Any] | None,
    agentx: bool = False,
) -> Any:
    """seq-CUV-Agent / seq-CUV-AgentX 工厂：GraphAgent（LangGraph 工具循环）。"""
    from .agent_graph import GraphAgent, GraphAgentX

    cls = GraphAgentX if agentx else GraphAgent
    return cls(
        uid,
        memory,
        theories,
        deepseek,
        state_dir=run_state,
        use_situational=True,
        weak_match_threshold=weak_match_threshold,
        source_profile=source_profile or {"available": False},
        tuning=dict(path_tuning or {}),
        failure_memory_budget=fm_budget,
        agent_cfg=agent_cfg,
    )


def run_with_supervision(
    run_fn: Callable[[], Any],
    *,
    max_restarts: int = 20,
    backoff_s: float = 10.0,
    log_prefix: str = "[loop-agent]",
) -> Any:
    """进程内看门狗：吸收 run_*_watchdog.bat 的「崩溃→等 10s→--resume 重启」职责。

    run_fn 每次进入都应从 CheckpointStore 重扫断点（run_method --resume 语义），
    因此重启不会重复计分。KeyboardInterrupt 不重试，直接上抛。
    """
    for attempt in range(max(1, max_restarts)):
        try:
            return run_fn()
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            if attempt >= max_restarts - 1:
                raise
            print(
                f"{log_prefix} crashed (attempt {attempt + 1}/{max_restarts}): "
                f"{type(e).__name__}: {e} — restart from checkpoint in {backoff_s:.0f}s",
                flush=True,
            )
            time.sleep(backoff_s)
    return None  # unreachable
