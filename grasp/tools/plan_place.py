#!/usr/bin/env python3
"""Plan where to put the object down after the lift, so the next trial starts from a new placement (self-reset).
**Does not move the arm.** Writes q_carry_deg / q_place_deg / place_* into <cap_dir>/plan.json.

The place pose is the grasp pose moved horizontally to a new spot and turned about the vertical axis through
the TCP (so the block lands at a new position AND a new yaw), 1 cm higher than the grasp (it drops 1 cm).
Spots: a grid inside the reach band that the chest camera sees (re-calibrated 11/20), at least --min-move from
the current block. Each candidate must solve IK on the same branch as the lift (<=60 deg per joint), and the
lift -> carry -> place -> carry path must keep link7 >= floor + 5 cm and the jaw corners >= 8 mm above the floor.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/plan_place.py grasp/runs/cap_xxx [--seed 3]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import pinocchio as pin  # noqa: E402

from grasp.plan import CFG, T_L7_G, floor_plane_base  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402

# reach band x visible area (base frame, metres), from the 11/20 reach / visibility maps
SPOTS = [(x, y) for x in (0.37, 0.40, 0.43) for y in (0.00, 0.04, 0.08, 0.12)]  # y=+0.16 dropped: 3 misses in a row there (iter_034)
YAWS = (-30, -15, 15, 30, 0)
LIMIT_MARGIN_DEG = 5.0  # --to: 運ぶ途中と置く姿勢は、どの関節も可動域の端からこれ以上（端ぎりぎりは補正で可動域にぶつかる・2026-11-21）


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-move", type=float, default=0.06)
    ap.add_argument("--to", default=None,
                    help="置く先を今の物の場所からの移動 dx,dy [m]（腕の付け根の座標・右は y が負）で決める。"
                         "例 --to 0,-0.20 で右に 20 cm（2026-11-21 運ぶ途中の手をよける試し）。向きは 0・±15° を試す")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    ob = np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    cur = np.array([(ob[:, 0].min() + ob[:, 0].max()) / 2, (ob[:, 1].min() + ob[:, 1].max()) / 2])
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    ik = OpenArmLeftIK()
    Tg = np.array(plan["T_grasp_base"])
    q_grasp, q_lift = np.deg2rad(plan["q_grasp_deg"]), np.deg2rad(plan["q_lift_deg"])
    tip = CFG["flange_to_tip_m"]

    def l7(T):
        return T @ np.linalg.inv(T_L7_G)

    def solve(T, seed):
        best = None
        extra = [seed + rng.normal(0, 0.5, 7) for _ in range(30)] if a.to else [seed + rng.normal(0, 0.3, 7) for _ in range(4)]
        for sd in [seed, q_grasp, q_lift] + extra:  # --to は置き先が 1 か所なので初期値を多めに試す
            q, ok, _, _ = ik.solve(pin.SE3(l7(T)[:3, :3], l7(T)[:3, 3]), np.clip(sd, ik.lo, ik.hi), position_only=False, max_iter=200)
            if ok and ik.within_limits(q):
                dmax = np.max(np.abs(np.rad2deg(q - seed)))
                if best is None or dmax < best[0]:
                    best = (dmax, q)
        return best

    def corners_low(qr):
        M = ik.fk(qr); T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation
        Tgg = T @ T_L7_G; tp = Tgg[:3, 3] + tip * Tgg[:3, 2]
        cs = [tp + sx * 0.058 * Tgg[:3, 0] + sy * 0.024 * Tgg[:3, 1] for sx in (-1, 1) for sy in (-1, 1)]
        return min(float(n_b @ c + d_b) for c in cs), float(n_b @ M.translation + d_b)

    def path_ok(qa, qb, low_ok=False):
        for s in np.linspace(0, 1, 20):
            c, l = corners_low(qa + (qb - qa) * s)
            if l < 0.05 or (c < 0.008 and not low_ok):
                return False
        return True

    if a.to:
        spots, yaws = [tuple(cur + np.array([float(v) for v in a.to.split(",")]))], (0, 15, -15, 30, -30)
    else:
        spots, yaws = SPOTS, YAWS
    order = [i for i in (range(len(spots)) if a.to else rng.permutation(len(spots)))
             if a.to or np.hypot(*(np.array(spots[i]) - cur)) >= a.min_move]
    tcp_g = Tg[:3, 3] + (tip - 0.015) * Tg[:3, 2]
    best_to = None
    for i in order:
        tgt = np.array(spots[i])
        for yaw in (yaws if a.to else rng.permutation(yaws)):
            c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
            Rz = np.eye(4); Rz[:2, :2] = [[c, -s_], [s_, c]]
            Tc = np.eye(4); Tc[:3, 3] = -tcp_g
            Tb = np.eye(4); Tb[:3, 3] = tcp_g + np.r_[tgt - cur, 0.01]
            Tp = Tb @ Rz @ Tc @ Tg                      # turn about the vertical through the TCP, move, +1 cm
            Tcarry = Tp.copy(); Tcarry[2, 3] += 0.10
            sp = solve(Tp, q_grasp)
            if sp is None or sp[0] > 60:
                continue
            sc = solve(Tcarry, sp[1])
            if sc is None or sc[0] > (45 if a.to else 30) or np.max(np.abs(np.rad2deg(sc[1] - q_lift))) > 60:  # --to は置き先が決まっているので 45° まで（右へ 15 cm で 30° ちょうど・2026-11-21）
                continue
            qp, qc = sp[1], sc[1]
            if corners_low(qp)[0] < 0.008:
                continue
            if not (path_ok(q_lift, qc) and path_ok(qc, qp, low_ok=True) and path_ok(qp, qc, low_ok=True)):
                continue
            if a.to:  # 置き先が決まっているときは、向きを全部試して可動域の余裕（端までの最小の角度）が一番大きいものを選ぶ
                m = min(float(np.min(np.rad2deg(np.minimum(q - ik.lo, ik.hi - q)))) for q in (qc, qp))
                if m >= LIMIT_MARGIN_DEG and (best_to is None or m > best_to[0]):
                    best_to = (m, qc, qp, tgt, yaw)
                continue
            plan.update(q_carry_deg=np.rad2deg(qc).tolist(), q_place_deg=np.rad2deg(qp).tolist(),
                        place_target_xy=tgt.tolist(), place_yaw_deg=int(yaw), place_from_xy=cur.round(3).tolist())
            json.dump(plan, open(os.path.join(a.cap_dir, "plan.json"), "w"), indent=2)
            print(f"[place] from {np.round(cur, 3).tolist()} to {tgt.tolist()} yaw {int(yaw):+d} deg; "
                  f"q_place {np.round(np.rad2deg(qp), 1).tolist()}")
            return 0
    if best_to is not None:
        m, qc, qp, tgt, yaw = best_to
        plan.update(q_carry_deg=np.rad2deg(qc).tolist(), q_place_deg=np.rad2deg(qp).tolist(),
                    place_target_xy=tgt.tolist(), place_yaw_deg=int(yaw), place_from_xy=cur.round(3).tolist())
        json.dump(plan, open(os.path.join(a.cap_dir, "plan.json"), "w"), indent=2)
        print(f"[place] from {np.round(cur, 3).tolist()} to {np.round(tgt, 3).tolist()} yaw {int(yaw):+d} deg; "
              f"関節の可動域の端まで最小 {m:.1f}°; q_place {np.round(np.rad2deg(qp), 1).tolist()}")
        return 0
    print(f"[place] no reachable placement found (block at {np.round(cur, 3).tolist()})"
          + (f"（--to は関節の可動域の端から {LIMIT_MARGIN_DEG:.0f}° 以上の置き方だけ）" if a.to else ""))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
