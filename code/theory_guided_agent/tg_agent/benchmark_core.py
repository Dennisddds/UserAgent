from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


def _http_json(url: str, payload: dict, headers: dict, timeout: int = 180) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace")[:800]
        raise RuntimeError(f"HTTP {e.code} {url}: {detail}") from e


@dataclass
class LLMConfig:
    api_key: str
    base_url: str
    model: str
    disable_thinking: bool = True


class OpenAICompatClient:
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self.base = cfg.base_url.rstrip("/")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.cfg.disable_thinking:
            # DeepSeek v4-pro / Qwen3.x thinking switches
            payload["thinking"] = {"type": "disabled"}
            payload["enable_thinking"] = False
        body = _http_json(
            f"{self.base}/chat/completions",
            payload,
            {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
        )
        msg = body["choices"][0]["message"]
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("reasoning_content") or "").strip()
        return content


JUDGE_SYSTEM = """你是公正的意见对齐评测员（opinion alignment judge）。
给定同一微博事件的【用户真实评论】与【模型预测评论】，从四个维度打分，每维分数为 0 到 1 的小数：
- stance: 立场/态度是否一致（支持/反对/质疑/中立等方向）
- core_judgment: 核心判断/主论点是否一致
- belief: 背后信念是否一致（对事实、制度、他者行为的基本看法）
- value: 价值取向是否一致（国家、安全、公平、秩序、实用等优先级）
opinion_alignment_score 必须等于上述四维的算术平均。
只输出一个 JSON 对象，不要 markdown，不要解释。
格式：{"stance":0.0,"core_judgment":0.0,"belief":0.0,"value":0.0,"opinion_alignment_score":0.0}
"""


def build_judge_user(context: str, ground_truth: str, prediction: str) -> str:
    return (
        f"【事件上下文】\n{context}\n\n"
        f"【真实评论 ground_truth】\n{ground_truth}\n\n"
        f"【预测评论 prediction】\n{prediction}\n\n"
        "请打分。"
    )


def parse_judge(text: str) -> dict[str, float]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return _zeros()
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            repaired = m.group(0)
            if repaired.count('"') % 2 == 1:
                repaired += '"'
            repaired += "}" * max(0, repaired.count("{") - repaired.count("}"))
            try:
                obj = json.loads(repaired)
            except json.JSONDecodeError:
                return _zeros()
    keys = ["stance", "core_judgment", "belief", "value"]
    scores = {}
    for k in keys:
        try:
            scores[k] = float(max(0.0, min(1.0, float(obj.get(k, 0.0)))))
        except (TypeError, ValueError):
            scores[k] = 0.0
    mean = sum(scores.values()) / 4.0
    # prefer model-provided mean if close; else recompute
    try:
        given = float(obj.get("opinion_alignment_score", mean))
        if abs(given - mean) > 0.15:
            given = mean
    except (TypeError, ValueError):
        given = mean
    scores["opinion_alignment_score"] = given
    return scores


def _zeros() -> dict[str, float]:
    return {
        "stance": 0.0,
        "core_judgment": 0.0,
        "belief": 0.0,
        "value": 0.0,
        "opinion_alignment_score": 0.0,
    }


def extract_context_and_gt(sample: dict) -> tuple[str, str]:
    gt = str(sample.get("completion") or "").strip()
    prompt = sample.get("prompt")
    context = ""
    if isinstance(prompt, list) and prompt:
        # [{role, content}, ...]
        parts = []
        for turn in prompt:
            if isinstance(turn, dict) and turn.get("content"):
                parts.append(str(turn["content"]))
        context = "\n".join(parts)
    elif isinstance(prompt, str):
        context = prompt
    else:
        context = str(sample.get("topic") or "")
    return context.strip(), gt


RESULT_DIMS = ["stance", "core_judgment", "belief", "value"]


def aggregate_metrics(
    rows: list[dict],
    *,
    weights: dict[str, float] | None = None,
    reason_weight: float = 0.0,
) -> dict[str, float]:
    """聚合 judge 分数。

    - weights: 结果四维可配权重（默认等权 → weighted_result == opinion_alignment_score）
    - reason_weight: >0 且行内 judge_scores 带 reason_correctness（rejudge_reason 回填）时，
      输出 composite = (1-w)*result + w*reason_correctness（民调 w=0 重结果；深访 w≈0.4 重原因）
    """
    keys = RESULT_DIMS + ["opinion_alignment_score"]
    acc = {k: 0.0 for k in keys}
    n = 0
    wr_acc = 0.0
    for r in rows:
        js = r.get("judge_scores") or {}
        if not js:
            continue
        n += 1
        for k in keys:
            acc[k] += float(js.get(k, 0.0))
        if weights:
            wsum = sum(float(weights.get(k, 0.0)) for k in RESULT_DIMS) or 1.0
            wr_acc += (
                sum(float(weights.get(k, 0.0)) * float(js.get(k, 0.0)) for k in RESULT_DIMS)
                / wsum
            )
    if n == 0:
        return {k: 0.0 for k in keys} | {"n": 0}
    out: dict[str, float] = {k: round(acc[k] / n, 4) for k in keys} | {"n": n}
    if weights:
        out["weighted_result"] = round(wr_acc / n, 4)
    if reason_weight > 0:
        rs = [
            float((r.get("judge_scores") or {}).get("reason_correctness"))
            for r in rows
            if (r.get("judge_scores") or {}).get("reason_correctness") is not None
        ]
        if rs:
            reason_mean = sum(rs) / len(rs)
            base = float(out.get("weighted_result") or out["opinion_alignment_score"])
            out["reason_correctness"] = round(reason_mean, 4)
            out["reason_n"] = len(rs)
            out["composite"] = round((1.0 - reason_weight) * base + reason_weight * reason_mean, 4)
    return out


def map_parallel(
    items: list[Any],
    fn: Callable[[Any], Any],
    *,
    workers: int = 6,
    desc: str = "work",
) -> list[Any]:
    out: list[Any] = [None] * len(items)
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, (i, x)): i for i, x in enumerate(items)}
        for fut in as_completed(futs):
            i = futs[fut]
            out[i] = fut.result()
            done += 1
            if done % 20 == 0 or done == len(items):
                print(f"[{desc}] {done}/{len(items)}", flush=True)
    return out
