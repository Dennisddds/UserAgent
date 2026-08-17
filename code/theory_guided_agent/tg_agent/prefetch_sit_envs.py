"""并行批量补建 situational env（个人接收路径）。

情境环境内容只取决于「帖子 + 发帖时间 cutoff」，与方法无关——
v1/v2/任何 run 共享 data/users/{uid}_situational_env{suffix}.json。
主 run 前跑一遍本脚本，主 run 的每步就只剩 LLM 预测+评测开销。

用法：
    python -m tg_agent.prefetch_sit_envs --user 1989660417 --workers 6

注意：请先停掉正在写同一 env 文件的主 run（文件锁是进程内的，
跨进程 concurrent write 会丢记录——丢的只是缓存，不影响正确性，但浪费检索）。
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tg_agent.llm import DeepSeekClient, load_env  # noqa: E402
from tg_agent.run_sequential import load_chrono_events  # noqa: E402
from tg_agent.situational_env import (  # noqa: E402
    ensure_situational_for_post,
    event_to_post,
    load_situational_store,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", default="1989660417")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--sit-suffix", default="_weibo_ai")
    ap.add_argument("--retrieval", default="weibo_ai")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    load_env(cfg["paths"]["env_file"])
    if not os.environ.get("WEIBO_COOKIE", "").strip():
        raise SystemExit("WEIBO_COOKIE missing")

    sit_path = ROOT / "data" / "users" / f"{args.user}_situational_env{args.sit_suffix}.json"
    events = load_chrono_events(args.user)
    store = load_situational_store(sit_path)
    records = store.get("records") or []
    have = {str(r.get("post_id") or "") for r in records} | {str(r.get("bid") or "") for r in records}
    todo = []
    for e in events:
        p = event_to_post(e)
        pid = str(p.get("post_id") or "").strip()
        bid = str(p.get("bid") or "").strip()
        if (pid and pid in have) or (bid and bid in have):
            continue
        todo.append(p)
    print(f"[prefetch] user={args.user} events={len(events)} have={len(events)-len(todo)} todo={len(todo)}", flush=True)
    if not todo:
        return

    llm = DeepSeekClient(model=cfg["llm"]["model"])
    env_file = cfg["paths"]["env_file"]
    t0 = time.time()
    done = fail = 0

    def work(post: dict) -> str:
        ensure_situational_for_post(
            user_id=args.user,
            post=post,
            out_path=sit_path,
            llm=llm,
            env_file=env_file,
            retrieval=args.retrieval,
        )
        return str(post.get("post_id") or post.get("bid") or "")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, p): p for p in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
                done += 1
            except Exception as ex:  # noqa: BLE001
                fail += 1
                pid = str(futs[fut].get("post_id") or "?")
                print(f"[prefetch] FAIL {pid}: {str(ex)[:120]}", flush=True)
            if (done + fail) % 25 == 0:
                el = time.time() - t0
                rate = (done + fail) / max(el, 1)
                eta = (len(todo) - done - fail) / max(rate, 1e-9) / 60
                print(f"[prefetch] {done+fail}/{len(todo)} ok={done} fail={fail} "
                      f"rate={rate*60:.1f}/min eta={eta:.0f}min", flush=True)
    print(f"[prefetch] DONE ok={done} fail={fail} elapsed={(time.time()-t0)/60:.0f}min", flush=True)


if __name__ == "__main__":
    main()
