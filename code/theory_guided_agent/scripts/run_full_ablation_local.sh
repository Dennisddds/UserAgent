#!/usr/bin/env bash
# Full-user local Flash ablations: 错题本 (fm) × 白盒思维链 (thinking)
# Logs: /root/autodl-tmp/logs/flash_full_ablation.log
set -u
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/rtwi
cd /root/autodl-tmp/UserAgent/theory_guided_agent

export LLM_MODEL=DeepSeek-V4-Flash
export LLM_BASE_URL=http://127.0.0.1:8001/v1
export LLM_API_KEY=EMPTY
export LLM_MAX_TOKENS=768
export LLM_MAX_MODEL_LEN=4096

OUT_ROOT=/root/autodl-tmp/UserAgent/outputs/flash_full_ablation
LOG=/root/autodl-tmp/logs/flash_full_ablation.log
mkdir -p "$OUT_ROOT" "$(dirname "$LOG")"

ts() { date '+%F %T'; }

run_one() {
  local uid="$1" tag="$2" fm="$3" think="$4"
  local out="${OUT_ROOT}/${tag}"
  mkdir -p "$out"
  if [[ -f "${out}/seq-CUV-Agent_${uid}/metrics.json" ]]; then
    echo "[$(ts)] SKIP ${tag}/${uid} (metrics exists)"
    return 0
  fi
  export LLM_ENABLE_THINKING="$think"
  echo "[$(ts)] START ${tag} user=${uid} fm=${fm} thinking=${think}"
  python -m tg_agent.run_sequential \
    --methods seq-CUV-Agent \
    --user "$uid" \
    --warmup 5 \
    --max-steps 0 \
    --no-ensure-situational \
    --fm-mode "$fm" \
    --out-root "$out" \
    --resume \
    --no-self-restart \
    2>&1 | tee -a "${out}/run.log"
  local ec=${PIPESTATUS[0]}
  echo "[$(ts)] END ${tag} user=${uid} exit=${ec}"
  if [[ -d "${out}/seq-CUV-Agent_${uid}" ]]; then
    python scripts/export_whitebox_report.py \
      "${out}/seq-CUV-Agent_${uid}" \
      --out "${out}/WHITEBOX_${uid}.md" || true
  fi
  return "$ec"
}

summarize() {
  python3 - <<'PY'
import json
from pathlib import Path
root = Path("/root/autodl-tmp/UserAgent/outputs/flash_full_ablation")
rows = []
for d in sorted(root.glob("*/seq-CUV-Agent_*/metrics.json")):
    m = json.loads(d.read_text(encoding="utf-8"))
    b = m.get("benchmark") or {}
    rows.append({
        "run": d.parents[1].name,
        "user": m.get("user_id"),
        "model": m.get("predict_model"),
        "n": m.get("num_scored"),
        "OA": b.get("opinion_alignment_score"),
        "late5": (m.get("late_alignment") or {}).get("last_5"),
        "stance": b.get("stance"),
        "core": b.get("core_judgment"),
        "belief": b.get("belief"),
        "value": b.get("value"),
    })
out = root / "ABLATION_SUMMARY.json"
out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
md = ["# Local Flash full ablation summary", ""]
md.append("| run | user | n | OA | late5 | stance | core | belief | value |")
md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
for r in rows:
    md.append(
        f"| {r['run']} | {r['user']} | {r['n']} | {r['OA']} | {r['late5']} | "
        f"{r['stance']} | {r['core']} | {r['belief']} | {r['value']} |"
    )
(root / "ABLATION_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print((root / "ABLATION_SUMMARY.md").read_text(encoding="utf-8"))
PY
}

{
  echo "[$(ts)] === FULL ABLATION QUEUE ==="
  curl -sS -m 5 http://127.0.0.1:8001/v1/models | head -c 200; echo

  # ---- Small user: 4 cells (full volume, ~10h) ----
  run_one 7463374646 small_full     full on
  run_one 7463374646 small_fm_only  full off
  run_one 7463374646 small_wb_only  off  on
  run_one 7463374646 small_base     off  off

  # ---- Big user: key score+whitebox cells (full volume; long) ----
  # full system vs no-错题本 (both keep thinking for whitebox fairness)
  run_one 1989660417 big_full    full on
  run_one 1989660417 big_fmoff   off  on
  # optional extras for 2×2 (uncomment if machine time allows)
  # run_one 1989660417 big_fm_only full off
  # run_one 1989660417 big_base    off  off

  summarize
  echo "[$(ts)] === ALL QUEUED RUNS FINISHED ==="
} >>"$LOG" 2>&1
