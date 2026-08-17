"""Lightweight LangGraph-style StateGraph (no hard dep on langgraph).

Mirrors the paper's core recipe parts:
  typed state · nodes · conditional edges · route history · retry budgets

If `langgraph` is installed later, the same node functions can be wrapped
into a real StateGraph; this module keeps the product contract local and
auditable without requiring the package.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Optional


NodeFn = Callable[[dict[str, Any]], dict[str, Any]]
RouterFn = Callable[[dict[str, Any]], str]


@dataclass
class GraphRunResult:
    state: dict[str, Any]
    route_history: list[str]
    final_node: str


@dataclass
class StateGraph:
    """Minimal conditional state graph with retry-aware routing."""

    name: str = "agent"
    nodes: dict[str, NodeFn] = field(default_factory=dict)
    edges: dict[str, str] = field(default_factory=dict)  # src -> dst (unconditional)
    conditional: dict[str, tuple[RouterFn, dict[str, str]]] = field(default_factory=dict)
    entry: str = ""
    finish: set[str] = field(default_factory=set)

    def add_node(self, name: str, fn: NodeFn) -> "StateGraph":
        self.nodes[name] = fn
        return self

    def add_edge(self, src: str, dst: str) -> "StateGraph":
        self.edges[src] = dst
        return self

    def add_conditional_edges(
        self, src: str, router: RouterFn, mapping: dict[str, str]
    ) -> "StateGraph":
        self.conditional[src] = (router, mapping)
        return self

    def set_entry(self, name: str) -> "StateGraph":
        self.entry = name
        return self

    def set_finish(self, *names: str) -> "StateGraph":
        self.finish.update(names)
        return self

    def invoke(
        self,
        state: dict[str, Any],
        *,
        max_steps: int = 32,
    ) -> GraphRunResult:
        if not self.entry:
            raise ValueError("entry node not set")
        cur = self.entry
        history: list[str] = list(state.get("route_history") or [])
        for _ in range(max_steps):
            if cur not in self.nodes:
                raise KeyError(f"unknown node: {cur}")
            history.append(cur)
            state["route_history"] = history
            state["current_node"] = cur
            patch = self.nodes[cur](state) or {}
            if patch is not state:
                state.update(patch)
            if cur in self.finish:
                return GraphRunResult(state=state, route_history=history, final_node=cur)
            if cur in self.conditional:
                router, mapping = self.conditional[cur]
                key = router(state)
                nxt = mapping.get(key) or mapping.get("__default__")
                if not nxt:
                    raise KeyError(f"router `{cur}` returned `{key}` with no mapping")
                cur = nxt
            elif cur in self.edges:
                cur = self.edges[cur]
            else:
                return GraphRunResult(state=state, route_history=history, final_node=cur)
        state["status"] = "max_steps_exceeded"
        return GraphRunResult(state=state, route_history=history, final_node=cur)
