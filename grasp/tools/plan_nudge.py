#!/usr/bin/env python3
"""掴めない所にある物を、閉じた爪で数 cm 押して、掴める所へずらす押し方を計画する。**腕は動かさない。**

2026-11-25・作業者: Shreyan。ロボットの前 20 cm・真ん中（y ≈ 0）に置いた平たい発泡スチロール（高さ 2.3 cm）は、左右どちらの腕でも
真上からの掴み方が届かなかった（GraspGenX 2000 個・傾き 60°・爪のすき間 6 mm でも 0）。同じ物が 5 cm 左なら左腕で掴めた。
そこで、物を押して届く所へずらす。grasp/tools/plan_push.py（アルミのバー・左腕だけ・rigid.png が要る）とは別に、どんな物にも使う。

押し方: 物の点（object.npy）の机に沿った外形から、押す向きの反対側の縁に、閉じた爪の先を物の高さの半分（机から 8 mm 以上）に下ろし、
押す向きに --dist だけ机に沿って押し、真上へ戻す。押す向き・腕・爪の傾き（0〜45°）・爪の向きを順に試し、全部の姿勢が IK で解けて、
keep-out の箱に入らず、爪の先の角が机から 3 mm 以上上にあり、押す区間で関節が大きく跳ばない最初のものを選ぶ。
押す向きは、物を腕が届く所（ロボットから遠い向き・腕の側）へ動かす向きから試す。

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/plan_nudge.py grasp/runs/<回> --q <左の今の角度> [--q-right ...]
    → <回>/nudge.json（腕・q_above・q_down・q_path（押す 4 段）・q_rise・押す向き）。見つからなければ 1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from grasp import arm as armcfg  # noqa: E402
from grasp.plan import CFG, floor_plane_base, in_keepout, se3, t_l7_g  # noqa: E402
from ik.openarm_left_ik import OpenArmIK  # noqa: E402

FX, FY = 0.022, 0.019  # 指 1 本の閉じる向きの厚み・指の幅の半分（plan.py と同じ）


def directions(cy: float, side: str) -> list:
    """押す向きの候補（付け根の座標の xy・単位ベクトル）。ロボットから遠ざける・その腕の側へ寄せる向きから。"""
    s = 1.0 if side == "left" else -1.0
    out = []
    for ang in (35, 0, 60, 90):  # 前（+x）から腕の側へ回した角度 [deg]
        a = np.deg2rad(ang)
        out.append(np.array([np.cos(a), s * np.sin(a)]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True, help="左腕の今の角度（lerobot の度）")
    ap.add_argument("--q-right", default=None, help="右腕の今の角度（無ければ GRASP_REST_RIGHT）")
    ap.add_argument("--dist", type=float, default=0.07, help="押す長さ [m]")
    ap.add_argument("--above", type=float, default=0.05, help="接する所のこれだけ上から真っすぐ下ろす [m]")
    ap.add_argument("--gap", type=float, default=0.008, help="爪の先を物の縁からこれだけ離して下ろす [m]")
    ap.add_argument("--max-tilt", type=float, default=45.0)
    ap.add_argument("--arms", default=None, help="試す腕（left,right）。既定は物の側の腕から両方")
    a = ap.parse_args(argv)
    d = a.cap_dir
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    n_b, d_b = floor_plane_base(d, T_bc)
    obj = np.load(os.path.join(d, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    h_obj = float(np.percentile(obj @ n_b + d_b, 98))
    mu = obj[:, :2].mean(0)
    tip_h = max(0.008, 0.5 * h_obj)
    tipz = CFG["flange_to_tip_m"]
    q_cur = {"left": [float(v) for v in a.q.split(",")],
             "right": [float(v) for v in a.q_right.split(",")] if a.q_right else (armcfg.default_rest("right") or armcfg.mirror_q(
                 [float(v) for v in a.q.split(",")]).tolist())}
    sides = a.arms.split(",") if a.arms else (["left", "right"] if mu[1] >= 0 else ["right", "left"])
    print(f"[nudge] 物の中心 {np.round(mu, 3).tolist()}・高さ {h_obj * 1000:.0f} mm → 爪の先を机から {tip_h * 1000:.0f} mm に下ろして {a.dist * 100:.0f} cm 押す")
    tried = {}
    for side in sides:
        ik = OpenArmIK(side=side)
        Tl7g = t_l7_g(side)
        q0 = np.deg2rad(q_cur[side])
        rng = np.random.default_rng(0)
        rand_seeds = [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(8)]
        for dirv in directions(mu[1], side):
            # 押す向きの反対側の縁（物の点の、押す向きへの射影の最小）に、爪の先を gap と指の厚みの分だけ離して下ろす
            proj = (obj[:, :2] - mu) @ dirv
            back = float(np.percentile(proj, 2))
            cxy = mu + dirv * (back - a.gap - FX)
            for tilt in (0.0, 15.0, 30.0, 45.0):
                if tilt > a.max_tilt:
                    continue
                for yaw in (0, 90):
                    for tdir in (180, 0, 90, 270):  # 傾ける向き（押す向きから回した角度。180 = 押す向きの反対へ手首を倒す）
                        c_, s_ = np.cos(np.deg2rad(tdir)), np.sin(np.deg2rad(tdir))
                        hdir = np.r_[c_ * dirv - s_ * np.array([-dirv[1], dirv[0]]), 0.0]
                        t = np.deg2rad(tilt)
                        z = np.cos(t) * np.array([0, 0, -1.0]) + np.sin(t) * hdir
                        p_ = np.r_[dirv, 0.0]; p_ = p_ - np.dot(p_, z) * z; p_ /= np.linalg.norm(p_)
                        c2, s2 = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
                        y = c2 * p_ + s2 * np.cross(z, p_)
                        R = np.c_[np.cross(y, z), y, z]
                        tip = np.r_[cxy, (tip_h - d_b - n_b[:2] @ cxy) / n_b[2]]
                        steps = [("down", np.zeros(3)), ("above", -z * a.above)] + \
                                [(f"push{k}", np.r_[dirv * a.dist * k / 4, 0.0]) for k in (1, 2, 3, 4)] + \
                                [("rise", np.r_[dirv * a.dist, 0.0] - z * a.above)]
                        poses, why = {}, None
                        for name, off in steps:
                            Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = tip + off - tipz * z
                            # 爪の先の角（閉じた爪・幅 ±FY・厚み ±FX）が机から 3 mm 以上上か
                            cs = [Tg[:3, 3] + tipz * z + sx * FX * R[:, 0] + sy * FY * R[:, 1] for sx in (-1, 1) for sy in (-1, 1)]
                            if min(float(n_b @ c + d_b) for c in cs) < 0.003:
                                why = "floor"; break
                            prev = poses.get("down") if name in ("above", "push1") else (list(poses.values())[-1] if poses else None)
                            seeds = ([prev] if prev is not None else []) + [q0] + (rand_seeds if not poses else [])
                            qs = None
                            for sd in seeds:
                                q, ok, _, _ = ik.solve(se3(Tg @ np.linalg.inv(Tl7g)), sd, position_only=False, max_iter=300)
                                if ok and ik.within_limits(q) and not in_keepout(ik, q):
                                    qs = q; break
                            if qs is None:
                                why = f"ik:{name}"; break
                            poses[name] = qs
                        if why is None:
                            seq = [poses["down"]] + [poses[f"push{k}"] for k in (1, 2, 3, 4)]
                            jump = max(np.max(np.abs(q2 - q1)) for q1, q2 in zip(seq[:-1], seq[1:]))
                            if jump >= np.deg2rad(12):
                                why = "jump"
                        if why is not None:
                            tried[why] = tried.get(why, 0) + 1
                            continue
                        out = {"mode": "nudge", "arm": side, "q_start_deg": list(np.rad2deg(q0)),
                               "q_above_deg": list(np.rad2deg(poses["above"])), "q_down_deg": list(np.rad2deg(poses["down"])),
                               "q_path_deg": [list(np.rad2deg(poses[f"push{k}"])) for k in (1, 2, 3, 4)],
                               "q_rise_deg": list(np.rad2deg(poses["rise"])), "contact_base_m": tip.tolist(),
                               "push_dir_base": dirv.tolist(), "push_dist_m": a.dist, "tilt_deg": tilt, "object_xy": mu.tolist()}
                        json.dump(out, open(os.path.join(d, "nudge.json"), "w"), indent=1)
                        print(f"[nudge] {side} 腕: 接する所 {np.round(tip, 3).tolist()}・押す向き {np.round(dirv, 2).tolist()}・"
                              f"傾き {tilt:.0f}° → 物の中心は {np.round(mu + dirv * a.dist, 3).tolist()} へ")
                        return 0
    print(f"[nudge] 押し方が見つからない（理由ごとの数 {tried}）")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
