#!/usr/bin/env python3
"""両腕を同時に動かしてよいか（左右の腕の通り道が y で離れているか）を確かめる。**腕は動かさない。**

2026-11-25・作業者: Shreyan。選別を速くするため、左腕と右腕で別々の物を同時に掴んで置く（run_pipeline.sh）。
cuRobo のサーバーは、もう片方の腕を休みの姿勢で止まっている物としてしか見ない（grasp/curobo/server.py）ので、両腕が同時に動くときの
腕どうしの当たりは見ていない。そこで、回ごとの計画（plan.json）から、その回で腕が通る所を点でなぞり、左腕の点の y の最小が、
右腕の点の y の最大より gap 以上大きいときだけ「同時に動かしてよい」とする（付け根の座標・+y が左）。

なぞる点: 関節 3〜7・その間の中点・指先の中心・爪の先（plan.arm_points）。姿勢は 始め → pregrasp → 掴む → 持ち上げ →
（運ぶ → 置く → 運ぶ）→ 始め の順に、関節の直線で間を steps 等分する。cuRobo の回り込みは関節の直線から外れるので、gap に余裕を持たせる
（既定 15 cm・GRASP_DUAL_GAP）。持っている物は指先のまわりにあるので、gap に物の半分の大きさも入れておく。

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/dual.py <左の回> <右の回>     # 0 = 同時に動かしてよい・1 = だめ
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

GAP = float(os.environ.get("GRASP_DUAL_GAP", "0.15"))


def keyposes(plan: dict) -> list:
    """その回で腕が順に通る姿勢 [deg] の列。"""
    seq = [plan["q_start_deg"], plan["q_pre_deg"], plan["q_grasp_deg"], plan["q_lift_deg"]]
    if plan.get("q_carry_deg") is not None and plan.get("q_place_deg") is not None:
        seq += [plan["q_carry_deg"], plan["q_place_deg"], plan["q_carry_deg"]]
    seq.append(plan["q_start_deg"])
    return [np.asarray(q, float) for q in seq]


def swept_points(plan: dict, steps: int = 8) -> np.ndarray:
    """計画の姿勢の間を関節の直線でなぞったときの腕の点（付け根の座標・(N,3)）。"""
    from grasp.plan import arm_points
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(side=plan["arm"])
    seq = keyposes(plan)
    pts = []
    for qa, qb in zip(seq[:-1], seq[1:]):
        for s in np.linspace(0.0, 1.0, steps + 1):
            pts.append(arm_points(ik, np.deg2rad(qa + (qb - qa) * s)))
    return np.vstack(pts)


def lanes_ok(plan_left: dict, plan_right: dict, gap: float = GAP) -> tuple[bool, float]:
    """(同時に動かしてよいか, 左の点の y の最小 − 右の点の y の最大 [m])。腕の名前がそろっていなければだめ。"""
    if plan_left.get("arm") != "left" or plan_right.get("arm") != "right":
        return False, float("-inf")
    g = float(swept_points(plan_left)[:, 1].min() - swept_points(plan_right)[:, 1].max())
    return g >= gap, g


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_left", help="左腕で掴む回（plan.json の arm が left）")
    ap.add_argument("cap_right", help="右腕で掴む回（plan.json の arm が right）")
    ap.add_argument("--gap", type=float, default=GAP, help="左右の腕の点の y の最小のすき間 [m]")
    a = ap.parse_args(argv)
    pl = json.load(open(os.path.join(a.cap_left, "plan.json")))
    pr = json.load(open(os.path.join(a.cap_right, "plan.json")))
    ok, g = lanes_ok(pl, pr, a.gap)
    rec = {"ok": ok, "gap_m": g, "need_m": a.gap, "left": a.cap_left, "right": a.cap_right}
    for d in (a.cap_left, a.cap_right):
        with open(os.path.join(d, "dual.json"), "w") as f:
            json.dump(rec, f, indent=1)
    print(f"[dual] {'同時に動かす' if ok else '同時には動かさない'}: 左右の腕の通り道のすき間 {g * 100:.1f} cm（要る {a.gap * 100:.0f} cm）")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
