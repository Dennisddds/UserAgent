#!/usr/bin/env bash
# Local Flash vs existing API Pro — small sample:
#   small user: ALL posts; big user: 30 scored steps only.
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/rtwi
cd /root/autodl-tmp/UserAgent/theory_guided_agent

export LLM_MODEL=DeepSeek-V4-Flash
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_API_KEY=EMPTY
export LLM_ENABLE_THINKING=true
export LLM_MAX_TOKENS=768
export LLM_MAX_MODEL_LEN=4096

OUT=/root/autodl-tmp/UserAgent/outputs/flash_full_vs_existing
LOG=/root/autodl-tmp/logs/flash_full_vs_existing.log
mkdir -p "$OUT" "$(dirname "$LOG")"
ts() { date '+%F %T'; }

run_user() {
  local uid="$1"
  local max_steps="$2"   # 0 = all
  local dest="${OUT}/seq-CUV-Agent_${uid}"
  if [[ -f "${dest}/metrics.json" ]]; then
    local n
    n=$(python3 -c "import json;print(json.load(open('${dest}/metrics.json')).get('num_scored') or 0)")
    if [[ "${n}" -gt 0 ]]; then
      echo "[$(ts)] SKIP ${uid} (finished num_scored=${n})"
      return 0
    fi
  fi
  echo "[$(ts)] START local user=${uid} warmup=5 max_steps=${max_steps}"
  python -m tg_agent.run_sequential \
    --methods seq-CUV-Agent \
    --user "$uid" \
    --warmup 5 \
    --max-steps "${max_steps}" \
    --no-ensure-situational \
    --fm-mode full \
    --out-root "$OUT" \
    --resume \
    --no-self-restart \
    2>&1 | tee -a "${OUT}/run_${uid}.log"
  echo "[$(ts)] END ${uid} exit=${PIPESTATUS[0]}"
  python scripts/export_whitebox_report.py \
    "${dest}" --out "${OUT}/WHITEBOX_${uid}.md" || true
}

compare() {
  python3 - <<'PY'
import json
from pathlib import Path

local_root = Path("/root/autodl-tmp/UserAgent/outputs/flash_full_vs_existing")
# Existing: small full agent; big use first comparable slice if available else full metrics note
existing = {
    "7463374646": Path("/root/autodl-tmp/UserAgent/outputs/benchmark_agent_small/seq-CUV-Agent_7463374646/metrics.json"),
    "1989660417": Path("/root/autodl-tmp/UserAgent/outputs/benchmark_agent_big/seq-CUV-Agent_1989660417/metrics.json"),
}
# Prefer recomputing existing OA on the SAME first N scored posts as local when jsonl exists
def first_n_oa(pred_path: Path, n: int):
    if not pred_path.exists() or n <= 0:
        return None, 0
    oas = []
    for line in pred_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("warmup"):
            continue
        oa = (r.get("judge_scores") or {}).get("opinion_alignment_score")
        if oa is None:
            continue
        oas.append(float(oa))
        if len(oas) >= n:
            break
    if not oas:
        return None, 0
    return round(sum(oas) / len(oas), 4), len(oas)

lines = [
    "# Local Flash (sample) vs existing API Pro",
    "",
    "Protocol: small=ALL, big=30 scored steps (warmup=5). Same judge.",
    "",
    "| user | side | model | n | OA | late5 | note |",
    "|---|---|---|---:|---:|---:|---|",
]
summary = []
for uid in ["7463374646", "1989660417"]:
    ep = existing[uid]
    em = json.loads(ep.read_text(encoding="utf-8")) if ep.exists() else {}
    eb = em.get("benchmark") or {}
    lp = local_root / f"seq-CUV-Agent_{uid}" / "metrics.json"
    local_n = 0
    local_oa = None
    late5 = None
    if lp.exists():
        lm = json.loads(lp.read_text(encoding="utf-8"))
        lb = lm.get("benchmark") or {}
        local_n = int(lm.get("num_scored") or 0)
        local_oa = lb.get("opinion_alignment_score")
        late5 = (lm.get("late_alignment") or {}).get("last_5")
        lines.append(
            f"| {uid} | local_Flash | {lm.get('predict_model')} | {local_n} | {local_oa} | {late5} | sample run |"
        )
    else:
        lines.append(f"| {uid} | local_Flash | — | — | pending | — | — |")

    # aligned existing slice
    ex_pred = ep.parent / "sequential_predictions.jsonl"
    aligned_oa, aligned_n = first_n_oa(ex_pred, local_n or (32 if uid == "7463374646" else 30))
    if aligned_oa is not None:
        lines.append(
            f"| {uid} | existing_API_aligned | {em.get('predict_model')} | {aligned_n} | {aligned_oa} | — | same first-n posts |"
        )
        if local_oa is not None:
            summary.append((uid, local_oa, aligned_oa, round(local_oa - aligned_oa, 4)))
    lines.append(
        f"| {uid} | existing_API_full | {em.get('predict_model')} | {em.get('num_scored')} | "
        f"{eb.get('opinion_alignment_score')} | {(em.get('late_alignment') or {}).get('last_5')} | published full run |"
    )

lines += ["", "## Delta (local − existing aligned)", ""]
for uid, lo, eo, d in summary:
    lines.append(f"- user {uid}: local {lo} vs aligned {eo} → **ΔOA={d:+.4f}**")

(local_root / "COMPARE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print((local_root / "COMPARE_SUMMARY.md").read_text(encoding="utf-8"))
PY
}

{
  echo "[$(ts)] === SAMPLE LOCAL VS EXISTING (small=ALL, big=30) ==="
  curl -sS -m 5 http://127.0.0.1:8001/v1/models | head -c 160; echo
  run_user 7463374646 0      # all small-user posts
  run_user 1989660417 30     # 30 scored steps after warmup
  compare
  echo "[$(ts)] === DONE ==="
} >>"$LOG" 2>&1
