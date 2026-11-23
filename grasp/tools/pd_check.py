#!/usr/bin/env python3
"""PD の誤差を測る（実機なし・腕を動かさない・記録を読むだけ）。2026-11-23 Munenori（Kota の PC の Claude Code から）。

置くときのずれを分ける 2 段目（1 段目の IK は grasp/tools/ik_check.py）。run_grasp.py が各段の終わりに traj.csv に残した
**実際の**関節角（モーターのエンコーダの読み）と、その段の**指令の**関節角を比べ、URDF の FK で手先の mm に直す。

段と指令の対応（run_grasp.py）:
  approach・close gripper → q_grasp / lift・hold → q_lift / carry to the new spot → q_carry / put down at the new spot・release → q_place
置く・運ぶの指令は、閉じた直後に解き直していれば（run.log の「置き先の解き直し: … 動かした」）plan.json の値と違うので、
run_grasp.py の replan_place_from_grasp と同じ式で作り直す（閉じた直後の読みは traj.csv の close gripper の行で代える。
run.log の「手を [..] mm 動かした」と作り直した値の差を「解き直しの再現」として出す）。

物の置き先（エンコーダの読みと URDF の上での話）:
  物は閉じた時の手（close gripper の行）に対して動かないとみて、置いた時（put down の行）の物の中心 = L7(置く・読み) · L7(閉じる・読み)^-1 · 物。
  計画の置き先（plan_place.py の式）との差を出す。減速機のガタ・たわみ・URDF と実機の差・TCP の取り方の誤差は**入らない**
  （エンコーダが読めない・モデルが同じだから）。それらはカメラで見た実際の置き先との差で見る（3 段目）。

    set -a; . grasp/config/sites/kitakyushu-collect.env; set +a
    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/pd_check.py grasp/runs/cap_resident_*
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from typing import Dict, List, Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

import pinocchio as pin  # noqa: E402

from grasp.plan import t_l7_g  # noqa: E402
from grasp.tools.ik_check import TCP_Z, fk_l7, place_from_unrounded, place_target  # noqa: E402
from ik.openarm_left_ik import OpenArmIK  # noqa: E402

J = [f"joint_{i}" for i in range(1, 8)]
PHASES = [("approach", "grasp"), ("close gripper", "grasp"), ("lift", "lift"), ("hold", "lift"),
          ("carry to the new spot", "carry"), ("put down at the new spot", "place"), ("release", "place")]
SHORT = {"approach": "近づく", "close gripper": "閉じる", "lift": "上げる", "hold": "保持",
         "carry to the new spot": "運ぶ", "put down at the new spot": "置く", "release": "離す"}
RE_REPLAN = re.compile(r"置き先の解き直し: 閉じた後の差 最大 ([\d.]+)° → 置く姿勢の手を \[([^\]]+)\] mm 動かした")


def read_traj(path: str) -> Dict[str, np.ndarray]:
    """traj.csv → {段の名前: 関節角 (7,) [deg]}。同じ名前は最初の行（置いた後の release を取るため put down の後のものを優先）。"""
    rows = list(csv.DictReader(open(path)))
    out: Dict[str, np.ndarray] = {}
    after_put = False
    for r in rows:
        ph = r["phase"]
        q = np.array([float(r[f"{j}.pos"]) for j in J])
        if ph == "put down at the new spot":
            after_put = True
        if ph == "release" and after_put and "release" not in out:
            out[ph] = q
        elif ph != "release" and ph not in out:
            out[ph] = q
    return out


def replan_goals(ik: OpenArmIK, plan: Dict, q_close: np.ndarray):
    """run_grasp.py の replan_place_from_grasp と同じ式で、解き直した運ぶ・置く指令を作る → (carry, place, 置く手の動き mm)。"""
    def l7(qd):
        return fk_l7(ik, qd)
    D = np.linalg.inv(l7(plan["q_grasp_deg"])) @ l7(q_close)
    new = []
    for q_old in (plan["q_carry_deg"], plan["q_place_deg"]):
        Tn = l7(q_old) @ D
        q, ok, _, _ = ik.solve(pin.SE3(Tn[:3, :3], Tn[:3, 3]), np.deg2rad(q_old), position_only=False, max_iter=300)
        if not ok:
            return None
        new.append(np.rad2deg(q))
    moved = (l7(new[1]) @ np.linalg.inv(l7(plan["q_place_deg"])))[:3, 3] * 1000
    return new[0], new[1], moved


def check_run(cap_dir: str) -> Optional[Dict]:
    """1 回の試行の PD の誤差を測る。traj.csv と plan.json が無ければ None。"""
    tp, pp = os.path.join(cap_dir, "traj.csv"), os.path.join(cap_dir, "plan.json")
    if not (os.path.exists(tp) and os.path.exists(pp)) or os.path.getsize(tp) == 0:
        return None
    plan = json.load(open(pp))
    traj = read_traj(tp)
    if "approach" not in traj:
        return None
    side = plan.get("arm", "left")
    ik = OpenArmIK(side=side)
    T_L7_G = t_l7_g(side)
    tcp_local = np.r_[0.0, 0.0, TCP_Z, 1.0]
    out: Dict = {"cap": os.path.basename(cap_dir.rstrip("/")), "arm": side, "notes": []}
    if os.path.getmtime(pp) > os.path.getmtime(tp) + 1:
        out["notes"].append("plan.json が試行の後に書き換わっている")

    goals = {"grasp": plan["q_grasp_deg"], "lift": plan["q_lift_deg"],
             "carry": plan.get("q_carry_deg"), "place": plan.get("q_place_deg")}
    log = os.path.join(cap_dir, "run.log")
    m = RE_REPLAN.search(open(log, encoding="utf-8", errors="replace").read()) if os.path.exists(log) else None
    out["replanned"] = bool(m)
    if m and "close gripper" in traj and goals["place"] is not None:
        r = replan_goals(ik, plan, traj["close gripper"])
        if r is None:
            out["notes"].append("解き直しを再現できない（IK が解けない）")
        else:
            goals["carry"], goals["place"], moved = r
            logged = np.array([float(v) for v in m.group(2).split(",")])
            out["replan_reproduce_mm"] = float(np.linalg.norm(moved - logged))

    def tcp(qd):
        return (fk_l7(ik, qd) @ T_L7_G @ tcp_local)[:3]

    out["phases"] = {}
    for ph, key in PHASES:
        if ph not in traj or goals.get(key) is None:
            continue
        g, q = np.asarray(goals[key], float), traj[ph]
        e = q - g
        out["phases"][ph] = {"joint_err_deg": e.tolist(), "max_joint_deg": float(np.max(np.abs(e))),
                             "worst_joint": int(np.argmax(np.abs(e))) + 1,
                             "tcp_mm": float(np.linalg.norm(tcp(q) - tcp(g)) * 1000),
                             "tcp_xyz_mm": ((tcp(q) - tcp(g)) * 1000).tolist()}

    # 物の置き先: 閉じた時の手に対して物が動かないとみる
    if "close gripper" in traj and "put down at the new spot" in traj and plan.get("q_place_deg") is not None:
        cur = place_from_unrounded(cap_dir, plan)
        Tp = place_target(plan, cur)
        if Tp is not None:
            G_inv = np.linalg.inv(T_L7_G)
            Tg = np.array(plan["T_grasp_base"])
            obj = np.r_[(Tg @ tcp_local)[:3], 1.0]
            want = (Tp @ G_inv) @ np.linalg.inv(Tg @ G_inv) @ obj
            for name, row in (("put", "put down at the new spot"), ("release", "release")):
                if row not in traj:
                    continue
                got = fk_l7(ik, traj[row]) @ np.linalg.inv(fk_l7(ik, traj["close gripper"])) @ obj
                d = (got - want)[:3] * 1000
                out[f"object_{name}_xyz_mm"] = d.tolist()
                out[f"object_{name}_xy_mm"] = float(np.hypot(d[0], d[1]))
            # 閉じた時の手のずれ（計画の掴む姿勢から）: 解き直しが打ち消すはずの分
            d = (fk_l7(ik, traj["close gripper"]) @ T_L7_G @ tcp_local - Tg @ tcp_local)[:3] * 1000
            out["close_shift_xyz_mm"] = d.tolist()
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cap_dirs", nargs="+", help="traj.csv と plan.json のある記録のフォルダ（grasp/runs/cap_*）")
    ap.add_argument("--json", default=None, help="結果を JSON にも書く")
    a = ap.parse_args(argv)

    rows = [r for r in (check_run(d) for d in sorted(set(a.cap_dirs))) if r is not None]
    if not rows:
        print("[pd-check] 測れる記録が無い（traj.csv と plan.json が要る）")
        return 1
    print(f"[pd-check] 爪の設定: {os.environ.get('GRASP_GRIPPER_JSON', '既定')}  TCP = グリッパーの z {TCP_Z:.4f} m")
    print("[pd-check] 段ごと: TCP の差 mm / 一番ずれた関節 J<番号> <度>（実際 − 指令・エンコーダの読み）")
    cols = [ph for ph, _ in PHASES]
    print(f"{'記録':30s} {'腕':5s} " + " ".join(f"{SHORT[c]:>13s}" for c in cols) + "   物 xy（置く/離す）mm  解き直し")
    for r in rows:
        cells = []
        for c in cols:
            v = r["phases"].get(c)
            cells.append(f"{v['tcp_mm']:5.1f}/J{v['worst_joint']}{v['max_joint_deg']:+5.1f}"[:13].rjust(13) if v else " " * 13)
        # 符号付きの最大の関節の誤差を出す
        for i, c in enumerate(cols):
            v = r["phases"].get(c)
            if v:
                e = v["joint_err_deg"][v["worst_joint"] - 1]
                cells[i] = f"{v['tcp_mm']:5.1f}/J{v['worst_joint']}{e:+5.1f}".rjust(13)
        ob = (f"{r['object_put_xy_mm']:5.1f}/{r.get('object_release_xy_mm', float('nan')):5.1f}"
              if "object_put_xy_mm" in r else "     -     ")
        rp = ("あり" + (f"（再現 {r['replan_reproduce_mm']:.1f}mm）" if "replan_reproduce_mm" in r else "")) if r["replanned"] else "なし"
        note = ("  ※" + "・".join(r["notes"])) if r["notes"] else ""
        print(f"{r['cap']:30s} {r['arm']:5s} " + " ".join(cells) + f"   {ob:>13s}  {rp}{note}")

    print()
    for c in cols:
        v = np.array([r["phases"][c]["tcp_mm"] for r in rows if c in r["phases"]])
        mj = np.array([r["phases"][c]["max_joint_deg"] for r in rows if c in r["phases"]])
        if len(v):
            print(f"[pd-check] {SHORT[c]:4s}: TCP の差 中央値 {np.median(v):5.1f} mm・最大 {v.max():5.1f} mm / "
                  f"一番ずれた関節 中央値 {np.median(mj):.1f}°・最大 {mj.max():.1f}°（{len(v)} 回）")
    for c in ("put down at the new spot", "close gripper"):
        E = np.array([r["phases"][c]["joint_err_deg"] for r in rows if c in r["phases"]])
        if len(E):
            print(f"[pd-check] {SHORT[c]}の関節ごとの誤差 平均 ± 標準偏差 [deg]: "
                  + " ".join(f"J{i + 1} {E[:, i].mean():+.1f}±{E[:, i].std():.1f}" for i in range(7)))
    for k in ("object_put", "object_release"):
        P = np.array([r[f"{k}_xyz_mm"] for r in rows if f"{k}_xyz_mm" in r])
        if len(P):
            print(f"[pd-check] 物の置き先の差（{'置いた時' if k == 'object_put' else '離した後'}・エンコーダと URDF の上）: "
                  f"xy 中央値 {np.median(np.hypot(P[:, 0], P[:, 1])):.1f} mm・最大 {np.hypot(P[:, 0], P[:, 1]).max():.1f} mm・"
                  f"平均 xyz {np.round(P.mean(0), 1).tolist()} mm（{len(P)} 回）")
    if a.json:
        json.dump(rows, open(a.json, "w"), indent=1, ensure_ascii=False)
        print(f"[pd-check] {a.json} に書いた")
    return 0


if __name__ == "__main__":
    sys.exit(main())
