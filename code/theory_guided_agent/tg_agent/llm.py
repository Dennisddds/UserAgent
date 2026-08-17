from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def load_env(env_file: str | Path) -> None:
    p = Path(env_file)
    if p.exists():
        load_dotenv(p, override=False)


class ToolCallingUnsupportedError(RuntimeError):
    """Endpoint rejected a request carrying `tools` (HTTP 400).

    The agent layer catches this and falls back to the XML tool protocol.
    """


_RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}


def _looks_like_deepseek(model: str, base_url: str) -> bool:
    blob = f"{model} {base_url}".lower()
    return "deepseek" in blob


def _looks_like_qwen(model: str, base_url: str) -> bool:
    blob = f"{model} {base_url}".lower()
    return "qwen" in blob


# RTWI-style markers: mining (factor/evidence) → reasoning (judgment/verbalize)
_RTWI_REASON_MARKERS = (
    "因此",
    "所以",
    "综上",
    "综合来看",
    "最终",
    "结论",
    "预测",
    "据此",
    "由此可见",
    "判断为",
    "我认为该用户",
    "该用户更可能",
    "therefore",
    "in conclusion",
    "final answer",
)


def _rtwi_white_box(reasoning: str, content: str) -> dict[str, Any]:
    """Split CoT into mining vs reasoning excerpts for white-box audits."""
    reasoning = reasoning or ""
    content = content or ""
    split_at = -1
    for mk in _RTWI_REASON_MARKERS:
        idx = reasoning.find(mk)
        if idx >= 40:  # keep some mining before the marker
            split_at = idx
            break
    if split_at < 0:
        # Fallback: first 45% mining, last 45% reasoning
        mid = max(1, int(len(reasoning) * 0.45))
        mining = reasoning[:mid]
        reason = reasoning[mid:]
    else:
        mining = reasoning[:split_at]
        reason = reasoning[split_at:]
    return {
        "has_reasoning": bool(reasoning),
        "reasoning_chars": len(reasoning),
        "answer_chars": len(content),
        "mining_excerpt": mining[:900],
        "reasoning_excerpt": reason[-900:] if reason else reasoning[-900:],
        "split_marker": next(
            (m for m in _RTWI_REASON_MARKERS if m in reasoning[max(0, split_at) : max(0, split_at) + 24]),
            "heuristic_mid",
        )
        if reasoning
        else "",
    }


class DeepSeekClient:
    """OpenAI-compatible chat client with optional white-box reasoning capture.

    When thinking is enabled, `reasoning_content` is kept separately from the
    final answer so UserAgent can audit RTWI-style mining vs reasoning stages.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "deepseek-v4-pro",
        *,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("LLM_API_KEY", "") or "EMPTY"
        self.base_url = (
            base_url
            or os.environ.get("LLM_BASE_URL")
            or "https://api.deepseek.com"
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL", "deepseek-v4-pro")
        if enable_thinking is None:
            env_flag = os.environ.get("LLM_ENABLE_THINKING", "").strip().lower()
            enable_thinking = env_flag in {"1", "true", "yes", "on"}
        self.enable_thinking = bool(enable_thinking)
        self.reasoning_effort = (
            reasoning_effort
            or os.environ.get("LLM_REASONING_EFFORT")
            or "high"
        )
        # Local Flash serve uses max_model_len≈4096; cap completions via env/config.
        self.local_max_tokens = int(
            os.environ.get("LLM_MAX_TOKENS")
            or os.environ.get("LLM_LOCAL_MAX_TOKENS")
            or "1024"
        )
        # Last call white-box fields
        self.last_message: dict[str, Any] = {}
        self.last_reasoning: str = ""
        self.last_content: str = ""
        if not self.api_key:
            raise RuntimeError("LLM_API_KEY missing")

    def _is_local_endpoint(self) -> bool:
        u = (self.base_url or "").lower()
        return "127.0.0.1" in u or "localhost" in u

    def capped_max_tokens(self, requested: int) -> int:
        """Keep prompt+completion under local context (DeepSeek-V4-Flash=4096)."""
        req = max(64, int(requested))
        if self._is_local_endpoint():
            return min(req, max(64, self.local_max_tokens))
        return req

    def _apply_thinking_controls(
        self,
        payload: dict[str, Any],
        *,
        disable_thinking: bool,
    ) -> None:
        """Endpoint-specific thinking / reasoning toggles."""
        if disable_thinking:
            # DeepSeek official API
            if _looks_like_deepseek(self.model, self.base_url):
                payload["thinking"] = {"type": "disabled"}
            # Qwen3 / local vLLM
            if _looks_like_qwen(self.model, self.base_url) or "127.0.0.1" in self.base_url or "localhost" in self.base_url:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            return

        # Thinking ON → white-box CoT (answer vs reasoning separated)
        if _looks_like_deepseek(self.model, self.base_url) and "api.deepseek.com" in self.base_url:
            # DeepSeek official API: reasoning_effort must be a top-level field,
            # NOT nested inside chat_template_kwargs (which is vLLM-only).
            payload["thinking"] = {"type": "enabled"}
            payload["reasoning_effort"] = self.reasoning_effort
        else:
            # Local vLLM: DeepSeek-V4-Flash / Qwen3-Thinking
            # DeepSeek-V4 returns CoT in message.reasoning (parser) or reasoning_content
            payload["chat_template_kwargs"] = {
                "enable_thinking": True,
                "thinking": True,
                "reasoning_effort": self.reasoning_effort,
            }
            if _looks_like_deepseek(self.model, self.base_url):
                # Some vLLM builds also honor top-level thinking flag
                payload["thinking"] = {"type": "enabled"}

    def chat_completion(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
        temperature: float = 0.4,
        max_tokens: int = 2500,
        disable_thinking: bool | None = None,
        retries: int = 3,
        backoff: float = 2.0,
    ) -> dict[str, Any]:
        """Raw chat-completions call returning the assistant `message` dict.

        Also populates `last_reasoning` / `last_content` for white-box audits.
        """
        if disable_thinking is None:
            disable_thinking = not self.enable_thinking

        max_tokens = self.capped_max_tokens(max_tokens)
        if self._is_local_endpoint():
            # Adaptive room: Flash serve max_model_len≈4096; shrink completion if prompt is long.
            ctx = int(os.environ.get("LLM_MAX_MODEL_LEN", "4096"))
            est_chars = 0
            for m in messages:
                est_chars += len(str(m.get("content") or ""))
                for tc in m.get("tool_calls") or []:
                    est_chars += len(json.dumps(tc, ensure_ascii=False))
            # Chinese-heavy prompts ≈ 1 token / 1.5–2 chars; stay conservative.
            est_in = max(1, int(est_chars / 1.6))
            room = max(128, ctx - est_in - 96)
            max_tokens = min(max_tokens, room)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        self._apply_thinking_controls(payload, disable_thinking=disable_thinking)
        if tools:
            payload["tools"] = tools
            if tool_choice is not None:
                payload["tool_choice"] = tool_choice

        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_exc: Exception | None = None
        for attempt in range(max(1, retries)):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions",
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=300) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                msg = body["choices"][0]["message"]
                self.last_message = msg
                content = (msg.get("content") or "").strip()
                # vLLM reasoning-parser may put CoT in reasoning / reasoning_content
                reasoning = (
                    (msg.get("reasoning_content") or msg.get("reasoning") or "")
                ).strip()
                # Some builds nest reasoning under message.reasoning.content
                if not reasoning and isinstance(msg.get("reasoning"), dict):
                    reasoning = str(
                        msg["reasoning"].get("content")
                        or msg["reasoning"].get("text")
                        or ""
                    ).strip()
                self.last_content = content
                self.last_reasoning = reasoning
                # Normalize so callers can always read reasoning_content
                if reasoning and not msg.get("reasoning_content"):
                    msg = dict(msg)
                    msg["reasoning_content"] = reasoning
                return msg
            except urllib.error.HTTPError as e:
                detail = e.read().decode("utf-8", errors="replace")[:500]
                if tools and e.code == 400:
                    raise ToolCallingUnsupportedError(
                        f"LLM HTTP 400 with tools payload (endpoint may not "
                        f"support tool calling): {detail}"
                    ) from e
                if e.code in _RETRYABLE_HTTP and attempt < retries - 1:
                    last_exc = e
                    time.sleep(backoff ** attempt)
                    continue
                raise RuntimeError(f"LLM HTTP {e.code}: {detail}") from e
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last_exc = e
                if attempt < retries - 1:
                    time.sleep(backoff ** attempt)
                    continue
                raise RuntimeError(f"LLM network error after {retries} attempts: {e}") from e
        raise RuntimeError(f"LLM call failed after {retries} attempts: {last_exc}")

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 2500,
        disable_thinking: bool | None = None,
    ) -> str:
        # When thinking is on, models may fill reasoning_content and leave content empty
        # until the final answer; prefer content, fall back to reasoning only if empty.
        msg = self.chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
        )
        content = (msg.get("content") or "").strip()
        if not content:
            content = (msg.get("reasoning_content") or msg.get("reasoning") or "").strip()
        return content

    def chat_with_trace(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.4,
        max_tokens: int = 2500,
        disable_thinking: bool | None = None,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: Any = None,
    ) -> dict[str, Any]:
        """Return answer + white-box reasoning for RTWI-style audits."""
        msg = self.chat_completion(
            messages,
            temperature=temperature,
            max_tokens=max_tokens,
            disable_thinking=disable_thinking,
            tools=tools,
            tool_choice=tool_choice,
        )
        content = (msg.get("content") or "").strip()
        reasoning = self.last_reasoning
        return {
            "content": content,
            "reasoning_content": reasoning,
            "tool_calls": msg.get("tool_calls"),
            "raw_message": msg,
            "white_box": _rtwi_white_box(reasoning, content),
        }
