"""Offline smoke for the Graph Agent (no API calls).

Stub the LLM with scripted tool-call responses and assert:
  1. happy path: decompose+retrieve_theory+retrieve_memory → finalize_prediction
     produces a valid PathOutput with c_trace.mode == "agent_graph" and graph absorption
  2. exhaustion: agent never finalizes → forced finalize (tool_choice pin) still produces output
  3. double exhaustion: forced finalize also fails → fast_fallback PathOutput

Run: python scripts/smoke_agent_graph.py
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tg_agent.agent_graph import GraphAgent  # noqa: E402
from tg_agent.models import MatchedTheory, RetrievedEvent, TheoryCard  # noqa: E402

STIMULUS = "微博热议话题：#某市新规#\n事件标题：某市出台新规\n事件摘要：某市发布城市治理新规，引发热议。"


class FakeMemory:
    def __init__(self) -> None:
        self.beliefs = ["城市治理需要精细化"]
        self.values = ["公平"]
        self.events = [1, 2, 3]
        self._event_tokens = [{"新规", "城市"}]

    def u_snapshot(self, max_motifs: int = 8) -> dict:
        return {"beliefs": self.beliefs, "interests": ["城市"], "communication": ["短句"],
                "motifs": [], "num_events": 3}

    def v_snapshot(self) -> dict:
        return {"persona_values": self.values}

    def identity_block(self) -> str:
        return "测试用户：关注城市治理的博主。"

    def retrieve(self, query: str, top_k: int = 4, recency_boost: float = 0.0) -> list:
        return [
            RetrievedEvent(
                map_id="m1", text="上次谈城市新规", score=0.08,
                event_title="旧规讨论", user_opinion="支持精细化管理，但别一刀切。",
            )
        ][:top_k]


class FakeTheories:
    def match(self, query: str, top_k: int = 6, **kw) -> list:
        card = TheoryCard(
            id="t1", name="程序正义", coordinate="procedural_justice",
            mechanism="程序公平感影响政策接受度", source="canonical",
            richness=0.8, grounded=True, conditions=["政策出台"],
            propositions=["程序公平→接受度↑"], summary="procedural justice theory",
        )
        return [MatchedTheory(card=card, score=0.09, why="route=coord:procedural_justice")]


class ScriptedLLM:
    """Queue-driven fake: each chat_completion pops one scripted assistant message."""

    def __init__(self, script: list[dict]) -> None:
        self.script = list(script)
        self.calls: list[dict] = []

    def chat_completion(self, messages, **kw) -> dict:
        self.calls.append(kw)
        if not self.script:
            raise AssertionError("ScriptedLLM ran out of scripted responses")
        return self.script.pop(0)

    def chat(self, messages, **kw) -> str:
        system = messages[0]["content"] if messages else ""
        if "传播学事件分析器" in system:
            return json.dumps({"factors": [
                {"id": "f1", "type": "policy", "text": "新规的公平性", "salience": 0.6},
                {"id": "f2", "type": "interest", "text": "对商户的影响", "salience": 0.4},
            ]}, ensure_ascii=False)
        # _fast_predict
        return json.dumps({
            "stance": "mixed",
            "emotion_probs": {"neutral": 1.0},
            "predicted_opinion": "方向对，但执行别一刀切。",
        }, ensure_ascii=False)


def _tool_call(name: str, args: dict, i: int = 0) -> dict:
    return {"id": f"call_{i}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}}


def make_agent(llm, tmp: str, **cfg_over) -> GraphAgent:
    cfg = {"max_tool_rounds": 3, "log_messages": False} | cfg_over
    return GraphAgent(
        "smoke_user", FakeMemory(), FakeTheories(), llm,
        state_dir=tmp, use_situational=False,
        source_profile={"available": False},
        tuning={"fast_path": {"enabled": False}},
        agent_cfg=cfg,
    )


def test_happy_path(tmp: str) -> None:
    llm = ScriptedLLM([
        # round 1: decompose + theory + memory
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("decompose_event", {}, 0),
            _tool_call("retrieve_theory", {"query": "城市新规 程序正义", "factor_id": "f1"}, 1),
            _tool_call("retrieve_memory", {"query": "新规", "factor_id": "f1"}, 2),
        ]},
        # round 2: finalize
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("finalize_prediction", {
                "stance": "mixed",
                "emotion_probs": {"neutral": 0.8, "joy": 0.2},
                "predicted_opinion": "支持精细化方向，但执行千万别一刀切。",
                "reason": "证据[旧规讨论]显示该用户一贯支持精细化但反对一刀切，程序正义坐标可解释。",
                "used": [{"factor_id": "f1", "coordinate": "procedural_justice", "evidence_idx": [0]}],
            }, 3),
        ]},
    ])
    agent = make_agent(llm, tmp)
    out = agent.predict(STIMULUS, post_id="p1", topic="某市新规")
    assert out.stance in {"support", "oppose", "mixed", "uncertain"}, out.stance
    assert out.predicted_opinion, "empty opinion"
    assert out.c_trace["mode"] == "agent_graph", out.c_trace["mode"]
    tools = out.c_trace["tools_called"]
    assert tools.get("decompose_event") == 1 and tools.get("retrieve_theory") == 1, tools
    assert tools.get("retrieve_memory") == 1 and tools.get("finalize_prediction") == 1, tools
    assert len(out.c_trace["tool_history"]) == 4, out.c_trace["tool_history"]
    assert out.c_trace["num_llm_calls"] == 2, out.c_trace["num_llm_calls"]
    assert out.factors and out.factors[0]["id"] == "f1", out.factors
    assert out.matched_theories and out.matched_theories[0]["id"] == "t1"
    assert agent.graph.stats()["edges"] > 0, "graph absorb did not run"
    assert "procedural_justice" in out.activated_coordinates
    print("PASS happy_path: stance=%s conf=%.3f tools=%s" % (out.stance, out.confidence, tools))


def test_forced_finalize(tmp: str) -> None:
    # agent keeps calling retrieve_memory, never finalizes; rounds cap = 3
    llm = ScriptedLLM([
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 0)]},
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 1)]},
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 2)]},
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 3)]},
        # forced finalize (tool_choice pinned)
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("finalize_prediction", {
                "stance": "support",
                "emotion_probs": {"neutral": 1.0},
                "predicted_opinion": "支持这个方向。",
                "reason": "个体证据支持。",
                "used": [{"factor_id": "f1", "coordinate": "", "evidence_idx": [0]}],
            }, 4),
        ]},
    ])
    agent = make_agent(llm, tmp, max_tool_rounds=3)
    out = agent.predict(STIMULUS, post_id="p2", topic="某市新规")
    assert out.stance == "support", out.stance
    assert out.c_trace["mode"] == "agent_graph"
    assert out.c_trace["num_tool_rounds"] == 3, out.c_trace["num_tool_rounds"]
    assert any("finalize_forced" in c for c in out.caveats), out.caveats
    # tool_choice pin was actually used on the last call
    assert llm.calls[-1].get("tool_choice", {}).get("function", {}).get("name") == "finalize_prediction"
    print("PASS forced_finalize: rounds=%s caveats=%s" % (out.c_trace["num_tool_rounds"], out.caveats))


def test_fast_fallback(tmp: str) -> None:
    llm = ScriptedLLM([
        # one round of useless tool calls, then no-parse prose → exhaust
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 0)]},
        {"role": "assistant", "content": "我觉得这个事情不好说", "tool_calls": []},
        # forced finalize attempt also fails (no finalize call)
        {"role": "assistant", "content": "", "tool_calls": [
            _tool_call("retrieve_memory", {"query": "新规"}, 1)]},
    ])
    agent = make_agent(llm, tmp)
    out = agent.predict(STIMULUS, post_id="p3", topic="某市新规")
    assert out.c_trace["mode"] == "fast_path", out.c_trace["mode"]
    assert out.c_trace.get("gate", {}).get("fallback") == "agent_exhausted"
    assert out.predicted_opinion, "fallback opinion empty"
    print("PASS fast_fallback: mode=%s opinion=%s" % (out.c_trace["mode"], out.predicted_opinion[:20]))


def main() -> None:
    for fn in (test_happy_path, test_forced_finalize, test_fast_fallback):
        with tempfile.TemporaryDirectory() as tmp:
            fn(tmp)
    print("ALL SMOKE TESTS PASSED")


if __name__ == "__main__":
    main()
