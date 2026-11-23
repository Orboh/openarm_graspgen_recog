#!/usr/bin/env python3
"""IK の誤差を測る（実機なし・腕を動かさない）。2026-11-23 Munenori（Kota の PC の Claude Code から）。

置くときのずれを「IK の誤差」「PD の誤差」「TCP・モデルの誤差」に分ける 1 段目。plan.json に残った関節角
（q_grasp / q_carry / q_place）を URDF の FK（Pinocchio・urdf/openarm_v10_bimanual.urdf）に戻し、
計画した目標の姿勢との差を出す。cuRobo の IK は位置 2 mm・向き 0.02 rad 以内を成功にする
（grasp/curobo/ik.py）ので、その分が TCP にどれだけ出るかを見る。右腕は cuRobo が左のモデルの鏡写しで解くので、
右の URDF の FK で測れば鏡写しの誤差も入る。

測るもの（どれも URDF の上だけの話。実機の関節がどこに行ったかは含まない）:
  - link7 の位置 [mm]・向き [deg] の差（目標 vs FK(q)）
  - TCP（指先から 15 mm 手前）の位置の差 [mm]
  - 物の動き: 掴んだ姿勢から置く姿勢への手の動き（FK の q_grasp→q_place）で物の中心がどこへ行くかと、
    計画の動き（T_grasp→T_place）で行くはずの所の差 [mm]。掴むときの IK の誤差は物と一緒に動くので打ち消し合い、
    置き先のずれに効くのはこの値。

置く目標の姿勢は plan_place.py と同じ式で作り直す。place_from_xy は 1 mm 単位に丸めて残っているので、object.npy と
GRASP_T_BASE_CAM から丸める前の値を作り直す（丸めた値と 0.6 mm 以内で合うときだけ。合わなければ丸めた値・±0.5 mm の幅）。
走りながらの置き先の解き直し（run_grasp.py の replan_place_from_grasp・Pinocchio の IK・eps 1e-4）は plan.json に残らないので測らない。

    set -a; . grasp/config/sites/kitakyushu-collect.env; set +a   # 爪の設定（GRASP_GRIPPER_JSON）をその置き場にそろえる
    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/ik_check.py grasp/runs/cap_resident_*
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

from grasp.plan import CFG, t_l7_g  # noqa: E402
from ik.openarm_left_ik import OpenArmIK  # noqa: E402

TCP_Z = CFG["flange_to_tip_m"] - 0.015  # グリッパーの座標での TCP（plan.py・plan_place.py と同じ）


def fk_l7(ik: OpenArmIK, q_deg) -> np.ndarray:
    """lerobot の度 → link7 の 4x4（腕の付け根の座標）。"""
    M = ik.fk(np.deg2rad(np.asarray(q_deg, float)))
    T = np.eye(4)
    T[:3, :3] = M.rotation
    T[:3, 3] = M.translation
    return T


def rot_err_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    """2 つの回転の差の角度 [deg]。"""
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def place_from_unrounded(cap_dir: str, plan: Dict) -> Optional[np.ndarray]:
    """plan_place.py と同じ式で、丸める前の物の場所（腕の付け根の座標の x, y）を作り直す。作れなければ None。"""
    ob_path = os.path.join(cap_dir, "object.npy")
    tbc_path = os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))
    if not (os.path.exists(ob_path) and os.path.exists(tbc_path)):
        return None
    T_bc = np.array(json.load(open(tbc_path))["T_base_cam"])
    ob = np.load(ob_path) @ T_bc[:3, :3].T + T_bc[:3, 3]
    cur = np.array([(ob[:, 0].min() + ob[:, 0].max()) / 2, (ob[:, 1].min() + ob[:, 1].max()) / 2])
    if np.max(np.abs(cur - np.array(plan["place_from_xy"]))) > 6e-4:  # 計画の後にカメラの較正が変わった
        return None
    return cur


def place_target(plan: Dict, cur: Optional[np.ndarray] = None) -> Optional[np.ndarray]:
    """plan_place.py と同じ式で、置く姿勢（グリッパーの座標・腕の付け根の座標）を作り直す。置き先が無ければ None。"""
    if "place_target_xy" not in plan or "place_from_xy" not in plan:
        return None
    Tg = np.array(plan["T_grasp_base"])
    tcp_g = Tg[:3, 3] + TCP_Z * Tg[:3, 2]
    tgt = np.array(plan["place_target_xy"])
    cur = np.array(plan["place_from_xy"]) if cur is None else cur
    c, s = np.cos(np.radians(plan["place_yaw_deg"])), np.sin(np.radians(plan["place_yaw_deg"]))
    Rz = np.eye(4); Rz[:2, :2] = [[c, -s], [s, c]]
    Tc = np.eye(4); Tc[:3, 3] = -tcp_g
    Tb = np.eye(4); Tb[:3, 3] = tcp_g + np.r_[tgt - cur, 0.01]
    return Tb @ Rz @ Tc @ Tg


def check_plan(cap_dir: str) -> Optional[Dict]:
    """1 つの記録の plan.json を測る。測れる姿勢が無ければ None。"""
    path = os.path.join(cap_dir, "plan.json")
    if not os.path.exists(path):
        return None
    plan = json.load(open(path))
    if "q_grasp_deg" not in plan or "T_grasp_base" not in plan:
        return None
    side = plan.get("arm", "left")
    ik = OpenArmIK(side=side)
    T_L7_G = t_l7_g(side)
    G_inv = np.linalg.inv(T_L7_G)
    tcp_local = np.r_[0.0, 0.0, TCP_Z, 1.0]

    Tg = np.array(plan["T_grasp_base"])
    targets = {"grasp": (Tg, plan.get("q_grasp_deg"))}
    cur = place_from_unrounded(cap_dir, plan) if "place_from_xy" in plan else None
    out_round = cur is None and "place_from_xy" in plan
    Tp = place_target(plan, cur)
    if Tp is not None:
        Tcarry = Tp.copy(); Tcarry[2, 3] += 0.10
        targets["carry"] = (Tcarry, plan.get("q_carry_deg"))
        targets["place"] = (Tp, plan.get("q_place_deg"))

    out = {"cap": os.path.basename(cap_dir.rstrip("/")), "arm": side, "place_from_rounded": out_round}
    fk = {}
    for name, (T_grip, q) in targets.items():
        if q is None:
            continue
        T_goal = T_grip @ G_inv          # link7 の目標
        T_got = fk_l7(ik, q)             # その関節角での link7
        fk[name] = T_got
        tcp_goal = (T_goal @ T_L7_G @ tcp_local)[:3]
        tcp_got = (T_got @ T_L7_G @ tcp_local)[:3]
        out[name] = {
            "l7_pos_mm": float(np.linalg.norm(T_got[:3, 3] - T_goal[:3, 3]) * 1000),
            "rot_deg": rot_err_deg(T_goal[:3, :3], T_got[:3, :3]),
            "tcp_mm": float(np.linalg.norm(tcp_got - tcp_goal) * 1000),
            "tcp_xyz_mm": ((tcp_got - tcp_goal) * 1000).tolist(),
        }
    if "grasp" in fk and "place" in fk:
        # 物の中心は計画の TCP にあるとみる。物は手と一緒に動く: 置いた物 = L7_置く · L7_掴む^-1 · 物
        obj = np.r_[(Tg @ tcp_local)[:3], 1.0]
        D_plan = (Tp @ G_inv) @ np.linalg.inv(Tg @ G_inv)
        D_fk = fk["place"] @ np.linalg.inv(fk["grasp"])
        d = (D_fk @ obj - D_plan @ obj)[:3]
        out["object_mm"] = float(np.linalg.norm(d) * 1000)
        out["object_xyz_mm"] = (d * 1000).tolist()
    return out


def fmt(r: Dict, key: str) -> str:
    """1 つの姿勢の差を短く。"""
    if key not in r:
        return "-"
    v = r[key]
    return f"{v['tcp_mm']:6.2f}mm ({v['l7_pos_mm']:5.2f}mm {v['rot_deg']:4.2f}°)"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cap_dirs", nargs="+", help="plan.json のある記録のフォルダ（grasp/runs/cap_*）")
    ap.add_argument("--json", default=None, help="結果を JSON にも書く")
    a = ap.parse_args(argv)

    print(f"[ik-check] 爪の設定: {os.environ.get('GRASP_GRIPPER_JSON', '既定（grasp/config/openarm_left_gripper.json）')}"
          f"  TCP = グリッパーの z {TCP_Z:.4f} m")
    print("[ik-check] 各列: TCP の差（link7 の位置の差・向きの差）。物 = 掴む→置くの手の動きで物の中心が行く所の差")
    print(f"{'記録':32s} {'腕':5s} {'掴む':24s} {'運ぶ':24s} {'置く':24s} {'物':>7s}  物の差 xyz [mm]")
    rows = []
    for d in sorted(set(a.cap_dirs)):
        r = check_plan(d)
        if r is None:
            continue
        rows.append(r)
        ob = f"{r['object_mm']:6.2f}" if "object_mm" in r else "     -"
        obx = " ".join(f"{v:+5.1f}" for v in r["object_xyz_mm"]) if "object_xyz_mm" in r else ""
        mark = "  ※置き元は丸めた値（±0.5 mm）" if r["place_from_rounded"] else ""
        print(f"{r['cap']:32s} {r['arm']:5s} {fmt(r, 'grasp'):24s} {fmt(r, 'carry'):24s} {fmt(r, 'place'):24s} {ob}  {obx}{mark}")
    if not rows:
        print("[ik-check] 測れる plan.json が無い")
        return 1
    for key in ("grasp", "place"):
        v = np.array([r[key]["tcp_mm"] for r in rows if key in r])
        if len(v):
            print(f"[ik-check] {key} の TCP の差: 中央値 {np.median(v):.1f} mm・最大 {v.max():.1f} mm（{len(v)} 件）")
    v = np.array([r["object_mm"] for r in rows if "object_mm" in r])
    if len(v):
        print(f"[ik-check] 物の置き先の差（IK だけ）: 中央値 {np.median(v):.1f} mm・最大 {v.max():.1f} mm（{len(v)} 件）")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False)
        print(f"[ik-check] {a.json} に書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
