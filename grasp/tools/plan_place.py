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

from grasp.plan import CFG, floor_plane_base, t_l7_g  # noqa: E402
from ik.openarm_left_ik import OpenArmIK  # noqa: E402

# reach band x visible area (base frame, metres), from the 11/20 reach / visibility maps
SPOTS = [(x, y) for x in (0.37, 0.40, 0.43) for y in (0.00, 0.04, 0.08, 0.12)]  # y=+0.16 dropped: 3 misses in a row there (iter_034)
YAWS = (-30, -15, 15, 30, 0)
LIMIT_MARGIN_DEG = float(os.environ.get("GRASP_PLACE_LIMIT_MARGIN", "5.0"))  # --to: 運ぶ途中と置く姿勢は、どの関節も可動域の端からこれ以上（端ぎりぎりは補正で可動域にぶつかる・2026-11-21）


def main(argv=None) -> int:
    """cuRobo の IK は先に LM の段だけで解き（速い）、置き方が 1 つも残らなければ L-BFGS まで全部まわして解き直す（2026-11-23）。"""
    rc = _main(argv, lm_only=True)
    if rc != 0 and os.environ.get("GRASP_PLAN_IK") == "curobo" and "--no-full-retry" not in (argv or sys.argv[1:]):
        print("[place] LM の段だけでは置き方が残らなかった。IK を全部まわして解き直す")
        rc = _main(argv, lm_only=False)
    return rc


def _main(argv=None, lm_only: bool = False) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-move", type=float, default=0.06)
    ap.add_argument("--to", default=None,
                    help="置く先を今の物の場所からの移動 dx,dy [m]（腕の付け根の座標・右は y が負）で決める。"
                         "例 --to 0,-0.20 で右に 20 cm（2026-11-21 運ぶ途中の手をよける試し）。向きは 0・±15° を試す")
    ap.add_argument("--at", default=None,
                    help="置く先を腕の付け根の座標の x,y [m] で決める（2026-11-22: 木材をテープの輪の中に置く）。向きは 0・±15・±30° を試す")
    ap.add_argument("--carry-h", type=float, default=0.10,
                    help="運ぶ姿勢を置く姿勢の何 m 上にするか（既定 0.10）。近くへずらすだけなら低くしてよい（2026-11-23）")
    ap.add_argument("--no-full-retry", action="store_true",
                    help="LM の段だけで解けなくても、IK を全部まわして解き直さない（置き先の候補を多く試すとき・1 か所 0.5 秒かかる）")
    ap.add_argument("--surface-z", type=float, default=None,
                    help="置く面（重量計の天板など）の上面の高さ [m・付け根の座標]。物の底をその 1 cm 上に下ろし、爪の先の角が面から "
                         "8 mm 以上上にあることも確かめる（無ければ机の上・2026-11-23）")
    ap.add_argument("--yaw-auto", action="store_true",
                    help="物の机に沿った外形の縦横の比が 1.6 未満（丸い・四角い）なら --any-yaw、長い物は ±30° だけ試す（2026-11-23・どんな物でも）")
    ap.add_argument("--any-yaw", action="store_true",
                    help="--to でも置く向きを一回り（15° ごと）試す（立てた丸太など向きを問わない物・2026-11-23）")
    a = ap.parse_args(argv)
    rng = np.random.default_rng(a.seed)
    plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
    side = plan.get("arm", "left")  # 2026-11-22: 右腕の計画でも置き先を作る
    T_L7_G = t_l7_g(side)
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    ob = np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    cur = np.array([(ob[:, 0].min() + ob[:, 0].max()) / 2, (ob[:, 1].min() + ob[:, 1].max()) / 2])
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    ik = OpenArmIK(side=side)
    Tg = np.array(plan["T_grasp_base"])
    q_grasp, q_lift = np.deg2rad(plan["q_grasp_deg"]), np.deg2rad(plan["q_lift_deg"])
    tip = CFG["flange_to_tip_m"]

    def l7(T):
        return T @ np.linalg.inv(T_L7_G)

    from grasp.curobo import client as curobo
    curobo.set_arm(side)
    use_curobo = os.environ.get("GRASP_PLAN_IK") == "curobo" and curobo.enabled()

    def pick(Qrow, OKrow, seed):
        """cuRobo が 1 つの目標に返した解の中から、可動域の端から余裕のある解を先に、その中で初期値に近い解を選ぶ。"""
        best = None
        for qd, ok in zip(Qrow, OKrow):
            r = np.deg2rad(qd)
            if ok and ik.within_limits(r):
                dmax = np.max(np.abs(qd - np.rad2deg(seed)))
                margin = float(np.min(np.rad2deg(np.minimum(r - ik.lo, ik.hi - r))))
                key = (margin < LIMIT_MARGIN_DEG, dmax)
                if best is None or key < best[2]:
                    best = (dmax, r, key)
        return None if best is None else best[:2]

    def solve_many(Ts, seeds):
        """cuRobo の IK で目標をまとめて 1 回で解く（2026-11-23 Munenori）。サーバーは 1 回ごとに 32 個に埋めて解くので、
        1 個ずつ送ると向きの数だけ 1〜3 秒かかっていた（置き先 1 か所・25 向きで 35〜53 秒）。"""
        if not Ts:
            return []
        Q, OK = curobo.ik(np.stack([l7(T) for T in Ts]), seed_deg=np.rad2deg(np.stack(seeds)), return_seeds=8, lm_only=lm_only)
        return [pick(Q[n], OK[n], seeds[n]) for n in range(len(Ts))]

    cache = {}  # (置き先の番号, 向き, "place" か "carry") → 解。use_curobo のときだけ先にまとめて解く

    def solve(T, seed, key=None):
        if use_curobo:  # 2026-11-22: Pinocchio の IK は置く姿勢を 24 向き中 16 で解けなかった（初期値の取りこぼし）
            if key in cache:
                return cache[key]
            return solve_many([T], [np.asarray(seed)])[0]
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

    if a.at:
        a.to = ",".join(str(v) for v in (np.array([float(v) for v in a.at.split(",")]) - cur))  # 以下は --to と同じ（1 か所に決めて置く）
    if a.yaw_auto:
        from grasp.objects import footprint
        long_, short = footprint(ob, up=[0.0, 0.0, 1.0])
        a.any_yaw = long_ / max(short, 1e-3) < 1.6
        print(f"[place] 物の外形 {long_ * 1000:.0f}×{short * 1000:.0f} mm → 置く向きは "
              + ("一回り試す（向きを問わない）" if a.any_yaw else "±30° だけ（長い物の向きを変えない）"))
    if a.to:
        spots, yaws = [tuple(cur + np.array([float(v) for v in a.to.split(",")]))], (0, 15, -15, 30, -30)
        if a.at or a.any_yaw:  # 2026-11-22: 真上から掴むと J7 が可動域の端（-86〜-90°）に近く、±30° では置く姿勢が解けなかった。丸い物は向きを問わないので一回り試す
            yaws = tuple(range(0, 180, 15)) + tuple(range(-15, -180, -15)) + (180,)
    else:  # 2026-11-23: 前は --at が無いと --to でもここに来て、決まった格子（SPOTS）に置いていた（--to が効かなかった）
        spots, yaws = SPOTS, YAWS
    order = [i for i in (range(len(spots)) if a.to else rng.permutation(len(spots)))
             if a.to or np.hypot(*(np.array(spots[i]) - cur)) >= a.min_move]
    tcp_g = Tg[:3, 3] + (tip - 0.015) * Tg[:3, 2]
    best_to = None
    why = {}  # 置き方を外した理由ごとの数（見つからなかったときに出す）
    dz_place = 0.01  # 物の底を、置く面の 1 cm 上へ（机なら掴んだ高さの 1 cm 上）
    if a.surface_z is not None:  # 物の底は今の机の上にあるとみなし、置く面の高さの分だけ上げる
        floor_here = -(d_b + n_b[0] * cur[0] + n_b[1] * cur[1]) / n_b[2]
        dz_place = a.surface_z - floor_here + 0.01
        print(f"[place] 置く面の上面 z {a.surface_z:.3f}（物の下の机 {floor_here:.3f}）→ {dz_place * 100:.1f} cm 上げて置く")

    def place_T(tgt, yaw):
        """置く姿勢（TCP を通る鉛直の軸で yaw 回し、置き先へ動かし、1 cm 上げる）と、その 10 cm 上の運ぶ姿勢。"""
        c, s_ = np.cos(np.radians(yaw)), np.sin(np.radians(yaw))
        Rz = np.eye(4); Rz[:2, :2] = [[c, -s_], [s_, c]]
        Tc = np.eye(4); Tc[:3, 3] = -tcp_g
        Tb = np.eye(4); Tb[:3, 3] = tcp_g + np.r_[tgt - cur, dz_place]
        Tp = Tb @ Rz @ Tc @ Tg
        Tcarry = Tp.copy(); Tcarry[2, 3] += a.carry_h
        return Tp, Tcarry

    if use_curobo:  # 置く姿勢を全部 1 回で、解けた物の運ぶ姿勢をもう 1 回で解く（置く姿勢の解を初期値にする）
        combos = [(i, int(yaw)) for i in order for yaw in yaws]
        TT = [place_T(np.array(spots[i]), yaw) for i, yaw in combos]
        for (i, yaw), sp in zip(combos, solve_many([t[0] for t in TT], [q_grasp] * len(TT))):
            cache[(i, yaw, "place")] = sp
        todo = [n for n, (i, yaw) in enumerate(combos) if cache[(i, yaw, "place")] is not None
                and cache[(i, yaw, "place")][0] <= 60]
        for n, sc in zip(todo, solve_many([TT[n][1] for n in todo], [cache[combos[n] + ("place",)][1] for n in todo])):
            cache[combos[n] + ("carry",)] = sc
    for i in order:
        tgt = np.array(spots[i])
        for yaw in (yaws if a.to else rng.permutation(yaws)):
            yaw = int(yaw)
            Tp, Tcarry = place_T(tgt, yaw)
            sp = solve(Tp, q_grasp, key=(i, yaw, "place"))
            if sp is None or sp[0] > 60:
                why["place_ik" if sp is None else "place_far"] = why.get("place_ik" if sp is None else "place_far", 0) + 1
                continue
            sc = solve(Tcarry, sp[1], key=(i, yaw, "carry"))
            if sc is None or sc[0] > (45 if a.to else 30) or np.max(np.abs(np.rad2deg(sc[1] - q_lift))) > 60:  # --to は置き先が決まっているので 45° まで（右へ 15 cm で 30° ちょうど・2026-11-21）
                why["carry"] = why.get("carry", 0) + 1
                continue
            qp, qc = sp[1], sc[1]
            if corners_low(qp)[0] < 0.008:
                why["corners_low"] = why.get("corners_low", 0) + 1
                continue
            if a.surface_z is not None and corners_low(qp)[0] - (a.surface_z - floor_here) < 0.008:
                why["surface_low"] = why.get("surface_low", 0) + 1  # 爪の先の角が置く面に当たる
                continue
            if not (path_ok(q_lift, qc) and path_ok(qc, qp, low_ok=True) and path_ok(qp, qc, low_ok=True)):
                why["path"] = why.get("path", 0) + 1
                continue
            if a.to:  # 置き先が決まっているときは、向きを全部試して可動域の余裕（端までの最小の角度）が一番大きいものを選ぶ
                m = min(float(np.min(np.rad2deg(np.minimum(q - ik.lo, ik.hi - q)))) for q in (qc, qp))
                if m < LIMIT_MARGIN_DEG:
                    why["limit_margin"] = why.get("limit_margin", 0) + 1
                    why.setdefault("margins", []).append(round(m, 1))
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
    print(f"[place] 外した理由ごとの数: {why}")
    print(f"[place] no reachable placement found (block at {np.round(cur, 3).tolist()})"
          + (f"（--to は関節の可動域の端から {LIMIT_MARGIN_DEG:.0f}° 以上の置き方だけ）" if a.to else ""))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
