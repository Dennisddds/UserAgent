from __future__ import annotations

"""Build standalone verbalizable-path webpage for small-user sequential run."""

import json
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "viz"
DATA_JSON = OUT_DIR / "small_user_sequential_data.json"
HTML_OUT = OUT_DIR / "verbalizable_path.html"


def export_data() -> dict:
    from pathlib import Path as P

    def load(p):
        return [json.loads(l) for l in P(p).open(encoding="utf-8") if l.strip()]

    cuv = load(
        "d:/UserAgent/outputs/benchmark_sequential_pathway/seq-CUV-TG_7463374646/sequential_predictions.jsonl"
    )
    gm = load(
        "d:/UserAgent/outputs/benchmark_sequential_pathway/seq-GenMinds_7463374646/sequential_predictions.jsonl"
    )
    sit = json.loads(
        P("d:/UserAgent/theory_guided_agent/data/users/7463374646_situational_env.json").read_text(
            encoding="utf-8"
        )
    )
    by_pid = {str(r["post_id"]).strip(): r for r in sit["records"]}

    def pack(rows):
        out = []
        for r in rows:
            pid = str(r.get("post_id") or "").strip()
            s = by_pid.get(pid) or {}
            env = s.get("environment") or {}
            path = s.get("observed_pathway_csv") or s.get("observed_pathway") or {}
            prop = s.get("propagation") or {}
            sources = s.get("information_sources") or []
            at = r.get("agent_trace") or {}
            js = r.get("judge_scores") or {}
            ts = r.get("timestamp")
            upstream = path.get("upstream_source") if isinstance(path, dict) else None
            out.append(
                {
                    "step": r.get("step"),
                    "post_id": pid,
                    "timestamp": ts,
                    "time": datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")
                    if ts
                    else "",
                    "topic": r.get("topic") or s.get("topic") or "",
                    "warmup": bool(r.get("warmup")),
                    "mem_before": r.get("num_events_before"),
                    "gt": r.get("ground_truth") or "",
                    "pred": r.get("prediction") or "",
                    "oa": js.get("opinion_alignment_score") if js else None,
                    "stance": js.get("stance") if js else None,
                    "core": js.get("core_judgment") if js else None,
                    "mode": (at.get("c_trace") or {}).get("mode")
                    or ("warmup" if r.get("warmup") else "genminds"),
                    "verbalization": at.get("verbalization") or "",
                    "coords": at.get("activated_coordinates")
                    or s.get("theory_coordinates")
                    or [],
                    "theories": [
                        {
                            "name": t.get("name"),
                            "coord": t.get("coordinate"),
                            "score": round(float(t.get("score") or 0), 3),
                            "mechanism": (t.get("mechanism") or "")[:180],
                        }
                        for t in (at.get("matched_theories") or [])[:3]
                    ],
                    "sit_summary": s.get("summary") or "",
                    "entry_channel": (path.get("entry_channel") if isinstance(path, dict) else "")
                    or "",
                    "hashtag": (path.get("hashtag") if isinstance(path, dict) else "") or "",
                    "upstream": (
                        f"{upstream.get('user_name','')}: {(upstream.get('text') or '')[:160]}"
                        if isinstance(upstream, dict)
                        and (upstream.get("text") or upstream.get("user_name"))
                        else ""
                    ),
                    "propagation_path": prop.get("path_to_user") or "",
                    "presentation_form": prop.get("presentation_form") or "",
                    "feed_cues": prop.get("salient_cues_in_feed") or [],
                    "sources": [
                        {
                            "title": x.get("title") or "",
                            "url": x.get("url") or x.get("link") or "",
                            "snippet": (x.get("snippet") or "")[:140],
                        }
                        for x in sources[:5]
                        if isinstance(x, dict)
                    ],
                    "evidence_gaps": s.get("evidence_gaps") or [],
                    "sit_comm": (env.get("communication") or {}).get("platform_climate")
                    or "",
                    "sit_psych": (env.get("psychological") or {}).get("public_mood") or "",
                    "sit_social": (env.get("social") or {}).get("event_backdrop") or "",
                    "sit_coords": s.get("theory_coordinates") or [],
                }
            )
        return out

    def slim_metrics(path: str) -> dict:
        m = json.loads(P(path).read_text(encoding="utf-8"))
        return {
            "overall": (m.get("benchmark") or {}).get("opinion_alignment_score"),
            "first5": (m.get("late_alignment") or {}).get("first_5"),
            "last5": (m.get("late_alignment") or {}).get("last_5"),
            "last10": (m.get("late_alignment") or {}).get("last_10"),
            "n_scored": m.get("num_scored"),
            "oa_series": m.get("oa_series") or [],
            "oa_rolling": m.get("oa_rolling") or [],
        }

    gm_by_step = {r["step"]: r for r in pack(gm)}
    cuv_rows = pack(cuv)
    for row in cuv_rows:
        g = gm_by_step.get(row["step"]) or {}
        row["gm_pred"] = g.get("pred") or ""
        row["gm_oa"] = g.get("oa")

    payload = {
        "user_id": "7463374646",
        "title": "Theory-Guided Sequential Agent",
        "subtitle": "可言化路径：真实获知路径 · 时序建构 · 理论校准 · 口头解释",
        "protocol": "CSV/检索证据路径 → predict(history<t) → judge → ingest → evolve",
        "env_kind": sit.get("kind") or "personal_reception_pathway",
        "steps": cuv_rows,
        "metrics": {
            "cuv": slim_metrics(
                "d:/UserAgent/outputs/benchmark_sequential_pathway/seq-CUV-TG_7463374646/metrics.json"
            ),
            "genminds": slim_metrics(
                "d:/UserAgent/outputs/benchmark_sequential_pathway/seq-GenMinds_7463374646/metrics.json"
            ),
        },
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_JSON.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return payload


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Theory-Guided Agent · 可言化路径</title>
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600;700&family=Newsreader:opsz,wght@6..72,500;6..72,600&display=swap" rel="stylesheet" />
<style>
  :root {
    --ink: #14201c;
    --muted: #5a6b64;
    --paper: #f3efe6;
    --panel: #fffdf8;
    --line: #d7d0c3;
    --accent: #0f6b5c;
    --accent-soft: #d8ebe6;
    --warn: #9a4d1c;
    --fall: #6b5b4b;
    --good: #1f7a45;
    --rail: #e8e2d6;
  }
  * { box-sizing: border-box; }
  html, body { margin: 0; padding: 0; }
  body {
    font-family: "Instrument Sans", system-ui, sans-serif;
    color: var(--ink);
    background:
      radial-gradient(1200px 600px at 10% -10%, #e7f2ee 0%, transparent 55%),
      radial-gradient(900px 500px at 100% 0%, #efe6d6 0%, transparent 50%),
      var(--paper);
    min-height: 100vh;
  }
  .wrap { max-width: 1120px; margin: 0 auto; padding: 36px 22px 80px; }
  .brand {
    font-family: "Newsreader", Georgia, serif;
    font-size: clamp(2.2rem, 5vw, 3.4rem);
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.05;
    margin: 0 0 10px;
  }
  .sub { color: var(--muted); font-size: 1.05rem; max-width: 42rem; line-height: 1.55; margin: 0 0 8px; }
  .proto {
    display: inline-block;
    margin-top: 14px;
    font-size: 0.85rem;
    color: var(--accent);
    border-bottom: 1px solid var(--accent);
    padding-bottom: 2px;
  }
  .stats {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin: 28px 0 18px;
  }
  .stat {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 14px 16px;
  }
  .stat .k { font-size: 0.75rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em; }
  .stat .v { font-family: "Newsreader", Georgia, serif; font-size: 1.7rem; margin-top: 4px; }
  .stat .d { font-size: 0.8rem; color: var(--muted); margin-top: 4px; }
  .chart-box {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 16px 18px 8px;
    margin-bottom: 22px;
  }
  .chart-box h2 { margin: 0 0 6px; font-size: 0.95rem; font-weight: 600; }
  .chart-cap { color: var(--muted); font-size: 0.78rem; margin-bottom: 8px; }
  svg.chart { width: 100%; height: 180px; display: block; }
  .scrub {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 16px 18px;
    margin-bottom: 18px;
  }
  .scrub-top { display: flex; justify-content: space-between; align-items: baseline; gap: 12px; margin-bottom: 10px; }
  .scrub-top h2 { margin: 0; font-size: 1rem; }
  .step-meta { color: var(--muted); font-size: 0.85rem; }
  input[type=range] {
    width: 100%;
    accent-color: var(--accent);
  }
  .ticks { display: flex; justify-content: space-between; color: var(--muted); font-size: 0.72rem; margin-top: 4px; }
  .path {
    display: grid;
    grid-template-columns: 1fr;
    gap: 10px;
    margin-bottom: 18px;
  }
  .node {
    background: var(--panel);
    border: 1px solid var(--line);
    padding: 14px 16px;
    position: relative;
  }
  .node::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--rail);
  }
  .node.active-theory::before { background: var(--accent); }
  .node.active-verb::before { background: var(--warn); }
  .node.active-pred::before { background: var(--good); }
  .node .label {
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 6px;
  }
  .node .body { font-size: 0.95rem; line-height: 1.55; white-space: pre-wrap; }
  .coords { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
  .coord {
    font-size: 0.75rem;
    padding: 3px 8px;
    background: var(--accent-soft);
    color: var(--accent);
    border: 1px solid #b7d5cd;
  }
  .coord.dim { background: #eee9df; color: var(--fall); border-color: var(--line); }
  .theories { margin-top: 8px; display: grid; gap: 8px; }
  .th {
    border-top: 1px dashed var(--line);
    padding-top: 8px;
    font-size: 0.88rem;
  }
  .th .nm { font-weight: 600; }
  .th .mk { color: var(--muted); font-size: 0.82rem; margin-top: 2px; }
  .compare {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
  }
  @media (max-width: 800px) {
    .stats { grid-template-columns: 1fr 1fr; }
    .compare { grid-template-columns: 1fr; }
  }
  .pill {
    display: inline-block;
    font-size: 0.72rem;
    padding: 2px 8px;
    border: 1px solid var(--line);
    color: var(--muted);
    margin-left: 8px;
    vertical-align: middle;
  }
  .pill.tg { border-color: #b7d5cd; color: var(--accent); background: var(--accent-soft); }
  .pill.fb { border-color: #dcc9b5; color: var(--warn); background: #f6ebe1; }
  .pill.wu { background: #eee9df; }
  .oa {
    font-family: "Newsreader", Georgia, serif;
    font-size: 1.4rem;
  }
  .controls { display: flex; gap: 8px; margin-top: 12px; }
  button {
    font-family: inherit;
    border: 1px solid var(--line);
    background: var(--panel);
    color: var(--ink);
    padding: 8px 12px;
    cursor: pointer;
    font-size: 0.85rem;
  }
  button:hover { border-color: var(--accent); color: var(--accent); }
  .foot { margin-top: 28px; color: var(--muted); font-size: 0.8rem; line-height: 1.5; }
</style>
</head>
<body>
  <div class="wrap">
    <h1 class="brand">Theory-Guided Agent</h1>
    <p class="sub" id="subtitle"></p>
    <div class="proto" id="protocol"></div>

    <div class="stats" id="stats"></div>

    <div class="chart-box">
      <h2>时序对齐曲线 · opinion alignment（滚动窗口=5）</h2>
      <div class="chart-cap">Source: small-user sequential run · user 7463374646 · seq-CUV-TG vs seq-GenMinds</div>
      <svg class="chart" id="chart" viewBox="0 0 1000 180" preserveAspectRatio="none"></svg>
    </div>

    <div class="scrub">
      <div class="scrub-top">
        <h2>逐步检视 · 可言化路径</h2>
        <div class="step-meta" id="stepMeta"></div>
      </div>
      <input type="range" id="slider" min="0" max="0" value="0" />
      <div class="ticks"><span>t=0 冷启动</span><span>记忆增长 →</span><span>对齐检验</span></div>
      <div class="controls">
        <button type="button" id="prev">上一步</button>
        <button type="button" id="next">下一步</button>
        <button type="button" id="play">自动播放</button>
      </div>
    </div>

    <div class="path" id="path"></div>

    <div class="compare" id="compare"></div>

    <p class="foot">
      协议：对每条帖子 t，先取 CSV/检索得到的<strong>真实获知路径</strong>（话题入口、上游源博、呈现信源 URL），
      再用历史 &lt; t 的记忆预测 → 与真实发言对齐 → 写入 Agent 记忆；
      理论只在路径坐标匹配足够强时进入，并用 verbalization 说出判断依据。弱匹配回退纯 GenMinds。
      无证据处标明 evidence_insufficient / evidence_gaps，不做舆论情绪脑补。
    </p>
  </div>

<script id="data" type="application/json">__DATA__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
const steps = DATA.steps;
const scoredIdx = steps.map((s,i)=>({s,i})).filter(x => x.s.oa != null).map(x => x.i);

document.getElementById('subtitle').textContent = DATA.subtitle + ' · user ' + DATA.user_id;
document.getElementById('protocol').textContent = DATA.protocol;

const mc = DATA.metrics.cuv, mg = DATA.metrics.genminds;
document.getElementById('stats').innerHTML = `
  <div class="stat"><div class="k">CUV-TG overall</div><div class="v">${fmt(mc.overall)}</div><div class="d">n=${mc.n_scored}</div></div>
  <div class="stat"><div class="k">GenMinds overall</div><div class="v">${fmt(mg.overall)}</div><div class="d">n=${mg.n_scored}</div></div>
  <div class="stat"><div class="k">CUV early → late</div><div class="v">${fmt(mc.first5)}→${fmt(mc.last5)}</div><div class="d">first5 → last5</div></div>
  <div class="stat"><div class="k">理论指导占比</div><div class="v">${theoryRate()}%</div><div class="d">theory_guided / scored</div></div>
`;

function fmt(x){ return (x==null||Number.isNaN(x)) ? '—' : Number(x).toFixed(2); }
function theoryRate(){
  const scored = steps.filter(s => !s.warmup);
  const tg = scored.filter(s => s.mode === 'theory_guided').length;
  return scored.length ? Math.round(100*tg/scored.length) : 0;
}

function drawChart(){
  const svg = document.getElementById('chart');
  const w=1000,h=180, pad=18;
  const seriesC = mc.oa_rolling || [];
  const seriesG = mg.oa_rolling || [];
  const n = Math.max(seriesC.length, seriesG.length, 1);
  const x = i => pad + (w-2*pad) * (n===1?0:i/(n-1));
  const y = v => h - pad - (h-2*pad) * Math.max(0, Math.min(1, v||0));
  function path(arr, color){
    if(!arr.length) return '';
    let d = arr.map((v,i)=>`${i?'L':'M'}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join(' ');
    return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" />`;
  }
  const grid = [0,0.25,0.5,0.75,1].map(v => {
    const yy=y(v);
    return `<line x1="${pad}" x2="${w-pad}" y1="${yy}" y2="${yy}" stroke="#e5dfd3" stroke-width="1" />
      <text x="4" y="${yy+3}" font-size="10" fill="#8a8073">${v}</text>`;
  }).join('');
  svg.innerHTML = grid + path(seriesG, '#8a8073') + path(seriesC, '#0f6b5c') +
    `<text x="${w-160}" y="16" font-size="11" fill="#0f6b5c">CUV-TG rolling</text>
     <text x="${w-160}" y="32" font-size="11" fill="#8a8073">GenMinds rolling</text>`;
}
drawChart();

const slider = document.getElementById('slider');
slider.max = String(steps.length - 1);
let playTimer = null;

function modePill(mode){
  if(mode==='theory_guided') return '<span class="pill tg">theory-guided</span>';
  if(mode==='genminds_fallback') return '<span class="pill fb">fallback GenMinds</span>';
  if(mode==='warmup') return '<span class="pill wu">warmup ingest</span>';
  return '<span class="pill">'+mode+'</span>';
}

function render(i){
  const s = steps[i];
  document.getElementById('stepMeta').innerHTML =
    `step ${s.step} / ${steps.length-1} · 记忆 ${s.mem_before} 条 · ${s.time || ''} ${modePill(s.mode)}`;

  const coords = (s.sit_coords||[]).map(c =>
    `<span class="coord">${c}</span>`
  ).join('') || '<span class="coord dim">（尚无情境坐标）</span>';

  const act = (s.coords||[]).map(c => `<span class="coord">${c}</span>`).join('');

  const theories = (s.theories||[]).map(t => `
    <div class="th">
      <div class="nm">${esc(t.name)} <span style="color:var(--muted);font-weight:400">· ${esc(t.coord)} · score ${t.score}</span></div>
      <div class="mk">${esc(t.mechanism||'')}</div>
    </div>`).join('') || '<div class="th"><div class="mk">本步未启用理论卡（warmup 或弱匹配回退）。</div></div>';

  const verb = s.verbalization
    ? esc(s.verbalization)
    : (s.warmup ? 'warmup：只入库，不预测。' : '（无 verbalization）');

  const srcHtml = (s.sources||[]).map(x =>
    `<div class="th"><div class="nm">${esc(x.title||'')}</div><div class="mk">${esc(x.url||'')}\n${esc(x.snippet||'')}</div></div>`
  ).join('') || '<div class="th"><div class="mk">无检索到的呈现信源</div></div>';

  const gaps = (s.evidence_gaps||[]).map(g => `<span class="coord dim">${esc(g)}</span>`).join('')
    || '<span class="coord dim">none listed</span>';

  document.getElementById('path').innerHTML = `
    <div class="node">
      <div class="label">1 · Stimulus / 话题</div>
      <div class="body"><strong>${esc(s.topic||'（无话题）')}</strong>\n真实用户发言（GT）：${esc(s.gt||'—')}</div>
    </div>
    <div class="node">
      <div class="label">2 · 真实获知路径（非推断）</div>
      <div class="body">入口：${esc(s.entry_channel||'—')}
标签：#${esc(s.hashtag||'')}#
上游源博：${esc(s.upstream||'none')}
传播到用户：${esc(s.propagation_path||'unknown')}
呈现方式：${esc(s.presentation_form||'—')}
摘要：${esc(s.sit_summary||'—')}
证据绑定三维：传播=${esc(s.sit_comm||'—')}；心理=${esc(s.sit_psych||'—')}；社会=${esc(s.sit_social||'—')}</div>
      <div class="coords">${coords}</div>
      <div class="theories"><div class="label" style="margin-top:8px">检索到的信息源 / 呈现</div>${srcHtml}</div>
      <div class="coords" style="margin-top:8px">${gaps}</div>
    </div>
    <div class="node active-theory">
      <div class="label">3 · 理论库匹配（由路径坐标校准）</div>
      <div class="body">激活坐标：</div>
      <div class="coords">${act || '<span class="coord dim">none</span>'}</div>
      <div class="theories">${theories}</div>
    </div>
    <div class="node active-verb">
      <div class="label">4 · 可言化 Verbalization</div>
      <div class="body">${verb}</div>
    </div>
    <div class="node active-pred">
      <div class="label">5 · Agent 预测 → 对齐 → 写入记忆</div>
      <div class="body">预测：${esc(s.pred||'（warmup 无预测）')}
对齐 OA：<span class="oa">${s.oa==null?'—':Number(s.oa).toFixed(2)}</span>
随后将真实帖子 ingest，记忆变为 ${Number(s.mem_before||0)+1} 条。</div>
    </div>
  `;

  document.getElementById('compare').innerHTML = `
    <div class="node">
      <div class="label">对照 · seq-CUV-TG</div>
      <div class="body">${esc(s.pred||'—')}\nOA ${s.oa==null?'—':Number(s.oa).toFixed(2)}</div>
    </div>
    <div class="node">
      <div class="label">对照 · seq-GenMinds</div>
      <div class="body">${esc(s.gm_pred||'—')}\nOA ${s.gm_oa==null?'—':Number(s.gm_oa).toFixed(2)}</div>
    </div>
  `;
}

function esc(t){
  return String(t||'').replace(/[&<>"']/g, m => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[m]));
}

slider.addEventListener('input', () => render(Number(slider.value)));
document.getElementById('prev').onclick = () => { slider.value = Math.max(0, Number(slider.value)-1); render(Number(slider.value)); };
document.getElementById('next').onclick = () => { slider.value = Math.min(steps.length-1, Number(slider.value)+1); render(Number(slider.value)); };
document.getElementById('play').onclick = () => {
  if(playTimer){ clearInterval(playTimer); playTimer=null; document.getElementById('play').textContent='自动播放'; return; }
  document.getElementById('play').textContent='暂停';
  playTimer = setInterval(() => {
    let v = Number(slider.value)+1;
    if(v >= steps.length){ v = 0; }
    slider.value = v; render(v);
  }, 1600);
};

// start at first scored theory-guided step
const start = steps.findIndex(s => s.mode === 'theory_guided');
slider.value = String(start >= 0 ? start : 0);
render(Number(slider.value));
</script>
</body>
</html>
"""


def build_html(payload: dict) -> Path:
    blob = json.dumps(payload, ensure_ascii=False)
    # prevent </script> breakout
    blob = blob.replace("<", "\\u003c").replace(">", "\\u003e")
    html = HTML_TEMPLATE.replace("__DATA__", blob)
    HTML_OUT.write_text(html, encoding="utf-8")
    return HTML_OUT


def main() -> None:
    payload = export_data()
    path = build_html(payload)
    print(f"wrote {DATA_JSON}")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
