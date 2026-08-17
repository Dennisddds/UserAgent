"""In-place temporal-cutoff cleanup for situational-env JSON files.

Removes information_sources published after the post date (later-information
leakage), flags sources with unknown dates, and records everything in
`evidence_gaps` / `removed_sources`. Backs up each file to `*.precutoff.bak`
before the first modification.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


TARGETS = [
    Path(r"D:\UserSimuAgent\项目最新版\sit_env"),
    Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent\data\users"),
]


def _date_key(date_str: str) -> str:
    m = re.search(r"(20\d{2})[\/\-年]?(\d{1,2})[\/\-月]?(\d{1,2})?", str(date_str))
    if not m:
        return str(date_str)[:10]
    y, mo, d = m.group(1), m.group(2), m.group(3) or "1"
    return f"{y}-{int(mo):02d}-{int(d):02d}"


def _source_date(src: dict) -> str | None:
    for key in ("as_of", "date", "published", "pub_date", "time"):
        v = src.get(key)
        if v:
            dk = _date_key(str(v))
            if re.match(r"20\d{2}-\d{2}-\d{2}$", dk):
                return dk
    for key in ("role", "snippet", "title"):
        v = src.get(key)
        if v:
            dk = _date_key(str(v))
            if re.match(r"20\d{2}-\d{2}-\d{2}$", dk):
                return dk
    return None


def _enforce_temporal_cutoff(post: dict, sources: list) -> tuple[list, list]:
    post_dk = _date_key(str(post.get("date") or ""))
    kept, dropped = [], []
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        sd = _source_date(src)
        if sd is None:
            src = dict(src)
            src["date_unknown"] = True
            kept.append(src)
        elif post_dk and sd > post_dk:
            dropped.append(
                {
                    "title": src.get("title") or "",
                    "url": src.get("url") or src.get("link") or "",
                    "date": sd,
                    "role": "removed_later_than_cutoff",
                }
            )
        else:
            kept.append(src)
    return kept, dropped


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("roots", nargs="*", default=[str(t) for t in TARGETS])
    args = ap.parse_args()
    roots = [Path(r) for r in args.roots]
    seen: set[Path] = set()
    for root in roots:
        if not root.exists():
            continue
        if root.is_file():
            paths = [root]
        else:
            paths = sorted(root.rglob("*situational_env*.json"))
        for path in paths:
            if "precutoff" in path.name or ".bak" in path.name:
                continue
            if path in seen:
                continue
            seen.add(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                print(f"SKIP {path.name}: {type(e).__name__}")
                continue
            records = data.get("records") or []
            if isinstance(records, dict):
                records = list(records.values())
            if not isinstance(records, list):
                continue
            changed = False
            removed_total = 0
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                sources = rec.get("information_sources") or []
                if not isinstance(sources, list):
                    continue
                kept, dropped = _enforce_temporal_cutoff(rec, sources)
                if dropped or any(s.get("date_unknown") for s in kept):
                    changed = True
                removed_total += len(dropped)
                if dropped:
                    rec["information_sources"] = kept
                    rec["removed_sources"] = dropped
                    gaps = list(rec.get("evidence_gaps") or [])
                    tag = f"removed_{len(dropped)}_sources_later_than_cutoff"
                    if tag not in gaps:
                        gaps.append(tag)
                    rec["evidence_gaps"] = gaps
                if any(s.get("date_unknown") for s in kept):
                    gaps = list(rec.get("evidence_gaps") or [])
                    tag = f"kept_{sum(1 for s in kept if s.get('date_unknown'))}_sources_with_unknown_date"
                    if tag not in gaps:
                        gaps.append(tag)
                    rec["evidence_gaps"] = gaps
            if changed:
                bak = path.with_suffix(path.suffix + ".precutoff.bak")
                if not bak.exists():
                    shutil.copyfile(path, bak)
                path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"CLEANED {path.name}: removed={removed_total}")
            else:
                print(f"OK      {path.name}")


if __name__ == "__main__":
    main()
