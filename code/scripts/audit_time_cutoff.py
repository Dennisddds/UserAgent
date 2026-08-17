"""Temporal-cutoff compliance audit (read-only).

For every situational-env record, verify that each information source `as_of`
date is on or before the post date, and that `temporal_cutoff` is not after the
post date. This quantifies later-information leakage for the "does time really
matter" question.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path


ROOTS = [
    Path(r"D:\UserSimuAgent\项目最新版\sit_env"),
    Path(r"D:\UserSimuAgent\项目最新版\UserAgent\theory_guided_agent\data\users"),
]


def parse_date(s: str | None) -> datetime | None:
    if not s:
        return None
    s = str(s).strip()
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s[:19], fmt)
        except ValueError:
            continue
    m = re.match(r"(\d{4}-\d{2}-\d{2})", s)
    if m:
        return datetime.strptime(m.group(1), "%Y-%m-%d")
    return None


def parse_date_flexible(s: str | None) -> datetime | None:
    """Handle ISO, slash, and Chinese date strings found in role narratives."""
    if not s:
        return None
    m = re.search(r"(20\d{2})\s*[\/\-年]\s*(\d{1,2})\s*[\/\-月]\s*(\d{1,2})?日?", str(s))
    if m:
        y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3) or 1)
        try:
            return datetime(y, mo, d)
        except ValueError:
            return None
    return parse_date(s)


def source_date(src: dict) -> datetime | None:
    for key in ("as_of", "date", "published", "pub_date", "time"):
        v = src.get(key)
        if v:
            dt = parse_date_flexible(str(v))
            if dt:
                return dt
    for key in ("role", "snippet", "title"):
        v = src.get(key)
        if v:
            dt = parse_date_flexible(str(v))
            if dt:
                return dt
    return None


def audit_file(path: Path) -> dict | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    records = data.get("records") or []
    if isinstance(records, dict):
        records = list(records.values())
    n = ok = late = unknown = 0
    cutoff_late = 0
    examples: list[dict] = []
    for rec in records:
        if not isinstance(rec, dict):
            continue
        post_dt = parse_date_flexible(rec.get("date"))
        sources = rec.get("information_sources") or []
        cutoff_dt = parse_date_flexible((rec.get("observed_pathway_csv") or {}).get("temporal_cutoff"))
        if post_dt and cutoff_dt and cutoff_dt > post_dt:
            cutoff_late += 1
        for src in sources:
            if not isinstance(src, dict):
                continue
            n += 1
            as_of = source_date(src)
            if as_of is None:
                unknown += 1
                continue
            if post_dt is None:
                unknown += 1
            elif as_of > post_dt:
                late += 1
                if len(examples) < 5:
                    examples.append(
                        {
                            "post_date": str(rec.get("date")),
                            "as_of": src.get("as_of") or src.get("date"),
                            "title": (src.get("title") or "")[:100],
                            "post_id": str(rec.get("post_id") or ""),
                        }
                    )
            else:
                ok += 1
    if n == 0 and cutoff_late == 0:
        return None
    total_with_date = ok + late
    return {
        "sources": n,
        "compliant": ok,
        "late": late,
        "unknown_date": unknown,
        "compliance_rate": round(ok / total_with_date, 4) if total_with_date else None,
        "cutoff_late_records": cutoff_late,
        "late_examples": examples,
    }


def main() -> None:
    out_rows = []
    seen = set()
    for root in ROOTS:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*situational_env*.json")):
            key = path.name
            if key in seen:
                continue
            seen.add(key)
            stats = audit_file(path)
            if stats:
                out_rows.append({"file": path.name, **stats})
    out = Path(r"D:\UserSimuAgent\项目最新版\time_cutoff_audit.json")
    out.write_text(json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out} with {len(out_rows)} files")
    for row in out_rows:
        print(
            f"{row['file']:48s} sources={row['sources']:5d} ok={row['compliant']:5d} "
            f"late={row['late']:4d} unknown={row['unknown_date']:5d} "
            f"compliance={row['compliance_rate']}"
        )
    late_files = [r for r in out_rows if r["late"]]
    print(f"files with violations: {len(late_files)}")


if __name__ == "__main__":
    main()
