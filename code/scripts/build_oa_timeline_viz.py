"""Extract OA-vs-step series from experiment outputs and emit the inline HTML
visualization fragment used in the conversation.
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(r"D:\UserSimuAgent\项目最新版\exp_outputs_v2")
OUT = Path(r"C:\Users\A\.codex\visualizations\2026\08\14\01a0005a-2821-7cc0-b9f0-9e8999ed7554\oa-timeline.html")


def series(path: Path) -> tuple[list[int], list[float]]:
    by_step: dict[int, float] = {}
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("warmup") or not (r.get("prediction") or "").strip():
                continue
            js = r.get("judge_scores") or {}
            if js.get("error") or r.get("error"):
                continue
            v = js.get("opinion_alignment_score")
            step = r.get("step")
            if isinstance(v, (int, float)) and step is not None:
                by_step[int(step)] = float(v)  # last row wins (gap-fill re-judged)
    steps = sorted(by_step)
    return steps, [by_step[s] for s in steps]


def main() -> None:
    cells = {
        "small_all_methods": {
            "seq-GenMinds_7463374646": ("s_genminds", "小样本 GenMinds"),
            "seq-CUV-TG_7463374646": ("s_tg", "小样本 TG"),
            "seq-CUV-Path_7463374646": ("s_path", "小样本 Path"),
            "seq-CUV-Fusion_7463374646": ("s_fusion", "小样本 Fusion"),
            "seq-CUV-Agent_7463374646": ("s_agent", "小样本 Agent"),
        },
        "big_all_methods": {
            "seq-GenMinds_1989660417": ("b_genminds", "大样本 GenMinds"),
        },
        "big_tg_fm": {
            "seq-CUV-TG_1989660417": ("b_tg_fm", "大样本 TG+错题本"),
        },
        "big_sample_0.3": {
            "seq-CUV-Agent_1989660417": ("b_s03", "大样本 抽样30%"),
        },
        "big_sample_0.5": {
            "seq-CUV-Agent_1989660417": ("b_s05", "大样本 抽样50%"),
        },
        "big_agent_fm_off": {
            "seq-CUV-Agent_1989660417": ("b_fm_off", "大样本 Agent无错题本"),
        },
        "big_agent_fast": {
            "seq-CUV-Agent_1989660417": ("b_fast", "大样本 Agent+Fast"),
        },
    }
    series_out = []
    panel_of = {}
    for cell, methods in cells.items():
        for sub, (sid, label) in methods.items():
            p = ROOT / cell / sub / "sequential_predictions.jsonl"
            if not p.exists():
                continue
            steps, oa = series(p)
            if not oa:
                continue
            panel = "a" if cell == "small_all_methods" else "b"
            panel_of[sid] = panel
            series_out.append({"id": sid, "label": label, "panel": panel, "steps": steps, "oa": oa})

    payload = json.dumps(series_out, ensure_ascii=False, separators=(",", ":"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(HTML_TEMPLATE.replace("__DATA__", payload), encoding="utf-8")
    print("wrote", OUT, "series:", len(series_out))


HTML_TEMPLATE = r"""<div id="oa-timeline-root">
  <div class="viz-controls" aria-label="OA 曲线模式">
    <button type="button" class="btn" id="mode-cum" aria-pressed="true">累计均值</button>
    <button type="button" class="btn" id="mode-roll" aria-pressed="false">滑动均值 50</button>
    <span class="text-small text-muted">x = 已见帖子数（时序步），y = OA</span>
  </div>
  <h3 id="panel-a-title">小样本用户 7463374646（34 个计分点）</h3>
  <div class="viz-row" id="legend-a" role="group" aria-label="小样本方法图例"></div>
  <div id="wrap-a" style="width:100%"><svg id="chart-a" role="img" aria-label="小样本用户 OA 随帖子数变化"></svg></div>
  <h3 id="panel-b-title">大样本用户 1989660417（最多 2707 帖，进行中）</h3>
  <div class="viz-row" id="legend-b" role="group" aria-label="大样本方法图例"></div>
  <div id="wrap-b" style="width:100%"><svg id="chart-b" role="img" aria-label="大样本用户 OA 随帖子数变化"></svg></div>
  <div class="tooltip" role="tooltip" id="oa-tip" style="position:absolute;pointer-events:none;display:none"></div>
</div>
<script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
<script>
(function () {
  const DATA = __DATA__;
  const root = document.getElementById("oa-timeline-root");
  const W = 50; // rolling window
  let mode = "cum";
  const visible = {};
  DATA.forEach((s) => { visible[s.id] = true; });

  const cssVar = (name) => getComputedStyle(root).getPropertyValue(name).trim() || "#888";
  const color = (i) => cssVar("--viz-series-" + ((i % 6) + 1));

  function valuesOf(s) {
    if (mode === "roll") {
      const out = [];
      let sum = 0;
      for (let i = 0; i < s.oa.length; i++) {
        sum += s.oa[i];
        const lo = Math.max(0, i - W + 1);
        if (lo > 0) sum -= s.oa[lo - 1];
        out.push(sum / (i - lo + 1));
      }
      return out;
    }
    const out = [];
    let sum = 0;
    for (let i = 0; i < s.oa.length; i++) { sum += s.oa[i]; out.push(sum / (i + 1)); }
    return out;
  }

  function stabilStep(s) {
    if (mode !== "cum" || s.steps.length < 100) return null;
    const v = valuesOf(s);
    const final = v[v.length - 1];
    for (let i = 99; i < v.length - 1; i++) {
      let ok = true;
      for (let j = i; j < v.length; j++) { if (Math.abs(v[j] - final) > 0.02) { ok = false; break; } }
      if (ok) return s.steps[i];
    }
    return null;
  }

  function drawPanel(pid) {
    const wrap = document.getElementById("wrap-" + pid);
    const svg = d3.select("#chart-" + pid);
    svg.selectAll("*").remove();
    const Wp = Math.max(wrap.clientWidth, 320);
    const H = 300;
    const m = { top: 18, right: 16, bottom: 34, left: 64 };
    svg.attr("viewBox", `0 0 ${Wp} ${H}`).attr("width", Wp).attr("height", H);
    const series = DATA.filter((s) => s.panel === pid);
    if (!series.length) return;

    const xAll = d3.extent(series.flatMap((s) => s.steps));
    const x = d3.scaleLinear().domain(xAll).range([m.left, Wp - m.right]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.bottom, m.top]);

    const g = svg.append("g");
    g.append("rect").attr("data-chart-frame", true)
      .attr("x", m.left).attr("y", m.top)
      .attr("width", Wp - m.left - m.right).attr("height", H - m.top - m.bottom)
      .attr("fill", "none").attr("stroke", "var(--border)").attr("stroke-width", 1);

    const yg = g.append("g").attr("transform", `translate(${m.left},0)`);
    yg.call(d3.axisLeft(y).ticks(5).tickFormat(d3.format(".1f")))
      .selectAll("text").attr("fill", "var(--foreground)").attr("font-size", 12);
    const xg = g.append("g").attr("transform", `translate(0,${H - m.bottom})`);
    xg.call(d3.axisBottom(x).ticks(Math.min(8, Math.floor((Wp - m.left - m.right) / 90))))
      .selectAll("text").attr("fill", "var(--foreground)").attr("font-size", 12);
    g.append("text").attr("class", "axis-title").attr("data-axis", "x")
      .attr("x", (m.left + Wp - m.right) / 2).attr("y", H - 4)
      .attr("text-anchor", "middle").attr("fill", "var(--muted-foreground)").attr("font-size", 12)
      .text("已见帖子数（步）");
    g.append("text").attr("class", "axis-title").attr("data-axis", "y")
      .attr("transform", "rotate(-90)").attr("x", -(H - m.bottom + m.top) / 2).attr("y", 14)
      .attr("text-anchor", "middle").attr("fill", "var(--muted-foreground)").attr("font-size", 12)
      .text("OA");

    series.forEach((s, i) => {
      const v = valuesOf(s);
      const pts = s.steps.map((st, j) => [x(st), y(v[j])]);
      const path = d3.line()(pts);
      g.append("path").attr("data-series-id", s.id)
        .attr("d", path).attr("fill", "none").attr("stroke", color(i)).attr("stroke-width", 1.6)
        .attr("opacity", visible[s.id] ? 1 : 0);
      if (mode === "cum") {
        const st = stabilStep(s);
        if (st != null) {
          g.append("line").attr("data-stab", s.id)
            .attr("x1", x(st)).attr("x2", x(st)).attr("y1", m.top).attr("y2", H - m.bottom)
            .attr("stroke", color(i)).attr("stroke-dasharray", "3 4").attr("stroke-width", 1)
            .attr("opacity", visible[s.id] ? 0.9 : 0);
        }
      }
    });

    const tip = document.getElementById("oa-tip");
    const overlay = g.append("rect").attr("data-chart-hit", true).attr("data-chart-hover-overlay", "cross-series")
      .attr("x", m.left).attr("y", m.top)
      .attr("width", Wp - m.left - m.right).attr("height", H - m.top - m.bottom)
      .attr("fill", "transparent");
    const guide = g.append("line").attr("data-chart-hover-guide", true)
      .attr("y1", m.top).attr("y2", H - m.bottom).attr("stroke", "var(--muted-foreground)")
      .attr("stroke-width", 1).attr("opacity", 0);
    const markers = g.append("g");

    overlay.on("mousemove", (ev) => {
      const px = d3.pointer(ev)[0];
      const st = Math.round(x.invert(px));
      guide.attr("x1", x(st)).attr("x2", x(st)).attr("opacity", 1);
      markers.selectAll("*").remove();
      const rows = [];
      series.forEach((s, i) => {
        if (!visible[s.id]) return;
        let j = d3.bisector((d) => d).center(s.steps, st);
        if (j < 0 || j >= s.steps.length) return;
        const v = valuesOf(s)[j];
        markers.append("circle").attr("data-chart-hover-marker", true)
          .attr("cx", x(s.steps[j])).attr("cy", y(v)).attr("r", 3.2)
          .attr("fill", color(i)).attr("stroke", "var(--background)").attr("stroke-width", 1);
        rows.push([color(i), s.label, v]);
      });
      if (rows.length) {
        tip.style.display = "block";
        tip.innerHTML = `<div class="text-small">步 ${st}</div>` + rows.map((r) =>
          `<div class="text-small" style="display:flex;align-items:center;gap:6px"><span style="width:9px;height:9px;border-radius:50%;background:${r[0]};display:inline-block"></span>${r[1]}: ${r[2].toFixed(3)}</div>`
        ).join("");
        const rect = wrap.getBoundingClientRect();
        tip.style.left = (ev.clientX - rect.left + 12) + "px";
        tip.style.top = (ev.clientY - rect.top - 10) + "px";
      } else {
        tip.style.display = "none";
      }
    }).on("mouseleave", () => { guide.attr("opacity", 0); markers.selectAll("*").remove(); tip.style.display = "none"; });
  }

  function buildLegend(pid) {
    const box = document.getElementById("legend-" + pid);
    box.innerHTML = "";
    DATA.filter((s) => s.panel === pid).forEach((s, i) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "btn btn-ghost";
      b.setAttribute("aria-pressed", "true");
      b.style.display = "inline-flex";
      b.style.alignItems = "center";
      b.style.gap = "6px";
      const sw = document.createElement("span");
      sw.style.width = "10px"; sw.style.height = "10px"; sw.style.borderRadius = "50%";
      sw.style.background = color(i); sw.style.display = "inline-block";
      b.appendChild(sw);
      b.appendChild(document.createTextNode(s.label));
      b.addEventListener("click", () => {
        visible[s.id] = !visible[s.id];
        b.setAttribute("aria-pressed", visible[s.id] ? "true" : "false");
        b.style.opacity = visible[s.id] ? 1 : 0.45;
        d3.select(`[data-series-id="${s.id}"]`).attr("opacity", visible[s.id] ? 1 : 0);
        const stab = d3.select(`[data-stab="${s.id}"]`);
        if (!stab.empty()) stab.attr("opacity", visible[s.id] ? 0.9 : 0);
      });
      box.appendChild(b);
    });
  }

  function draw() { drawPanel("a"); drawPanel("b"); }

  document.getElementById("mode-cum").addEventListener("click", function () {
    mode = "cum";
    this.setAttribute("aria-pressed", "true");
    document.getElementById("mode-roll").setAttribute("aria-pressed", "false");
    draw();
  });
  document.getElementById("mode-roll").addEventListener("click", function () {
    mode = "roll";
    this.setAttribute("aria-pressed", "true");
    document.getElementById("mode-cum").setAttribute("aria-pressed", "false");
    draw();
  });

  buildLegend("a");
  buildLegend("b");
  draw();
  new ResizeObserver(() => draw()).observe(document.getElementById("wrap-a"));
  new ResizeObserver(() => draw()).observe(document.getElementById("wrap-b"));
})();
</script>
"""


if __name__ == "__main__":
    main()
