#!/usr/bin/env python3
"""段ごとの時間のまとめ（grasp/timing.py と run_pipeline.sh が書いた *.timing.jsonl を読むだけ・標準ライブラリだけ）。

2026-11-25・作業者: Shreyan。選別のどの段に時間がかかっているかを見る。

    python3 grasp/tools/timing_report.py grasp/runs/pipeline_<日時>.timing.jsonl          # 回ごとの内訳と、段ごとの合計
    python3 grasp/tools/timing_report.py <jsonl> --timeline 2                               # 2 回目を時刻の順に並べる
    python3 grasp/tools/timing_report.py <jsonl> --top 60                                  # 段ごとの表を 60 行まで

入れ子の段（例 snap/yoloe (load+predict) の中の yoloe/predict）は、親の段の時間にも入っている。
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import sys

DETAIL = {"arm/move": "phase", "arm/position correction": "phase", "arm/follow dense path": "phase",
          "run/check segment": "name", "run/route (check or detour)": "name", "run/check dense path": "name"}
BASE = ("run", "round", "proc", "pid", "thread", "stage", "t_start", "t_end", "dur_s")


def load(path):
    rows = []
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                try:
                    rows.append(json.loads(ln))
                except ValueError:
                    print(f"[report] 読めない行を飛ばす: {ln[:80]}", file=sys.stderr)
    return rows


def label(r):
    st = r["stage"]
    if st in DETAIL and r.get(DETAIL[st]):
        st = f"{st} [{r[DETAIL[st]]}]"
    return f"{r['proc']}: {st}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("jsonl")
    ap.add_argument("--top", type=int, default=40, help="段ごとの表の行数")
    ap.add_argument("--timeline", type=int, default=None, help="この回の段を時刻の順に全部出す")
    a = ap.parse_args(argv)
    rows = load(a.jsonl)
    if not rows:
        print(f"[report] 記録が無い: {a.jsonl}")
        return 1

    pipe = [r for r in rows if r["proc"] == "run_pipeline.sh"]
    rounds = sorted({r["round"] for r in pipe if r["round"] > 0})
    stages = sorted({r["stage"] for r in pipe if r["round"] > 0 and r["stage"] != "round total"})
    if rounds:
        print("\n=== 回ごとの内訳 [s]（run_pipeline.sh の段）")
        w = max(len(s) for s in stages + ["round total"])
        print(f"{'段':{w}s} " + " ".join(f"{'#' + str(k):>8s}" for k in rounds))
        for st in stages + ["round total"]:
            cells = []
            for k in rounds:
                v = [r["dur_s"] for r in pipe if r["round"] == k and r["stage"] == st]
                cells.append(f"{sum(v):8.1f}" if v else f"{'—':>8s}")
            print(f"{st:{w}s} " + " ".join(cells))
        res = {r["round"]: r.get("result", "") for r in pipe if r["stage"] == "round total"}
        print(f"{'result':{w}s} " + " ".join(f"{res.get(k, ''):>8s}" for k in rounds))
    for r in pipe:
        if r["round"] == 0 or r["stage"] == "whole run":
            print(f"{r['stage']}: {r['dur_s']:.1f} s")

    agg = collections.defaultdict(list)
    for r in rows:
        if r["proc"] != "run_pipeline.sh":
            agg[label(r)].append(r["dur_s"])
    print(f"\n=== 段ごとの合計（全部の回・Python のプロセスの中・合計の大きい順・上から {a.top} 行）")
    print(f"{'合計 s':>8s} {'回数':>5s} {'平均 s':>8s} {'最大 s':>8s}  段")
    for name, v in sorted(agg.items(), key=lambda kv: -sum(kv[1]))[:a.top]:
        print(f"{sum(v):8.2f} {len(v):5d} {sum(v) / len(v):8.3f} {max(v):8.3f}  {name}")

    if a.timeline is not None:
        tl = sorted((r for r in rows if r["round"] == a.timeline), key=lambda r: (r["t_start"], -r["dur_s"]))
        if tl:
            t0 = tl[0]["t_start"]
            print(f"\n=== {a.timeline} 回目の時刻の順（始まりからの秒・長さ）")
            for r in tl:
                extra = {k: v for k, v in r.items() if k not in BASE}
                print(f"+{r['t_start'] - t0:7.2f} {r['dur_s']:7.2f}  {r['proc']:<16s} {r['stage']}" + (f"  {extra}" if extra else ""))
            print(f"（始まり {dt.datetime.fromtimestamp(t0):%H:%M:%S}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
