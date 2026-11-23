#!/usr/bin/env python3
"""Fit the jaw-centre (TCP) position in link7 AND wrist-joint zero offsets from several hand-taught poses.

Why: one teach (11/20, J6 +36 / J7 +25) fixed the TCP to (22, -10, 210) mm in link7, but grasps at a
different wrist (J6 -40 / J7 +86) still missed by cm. A rigid TCP error would be fixed by one teach in
any wrist pose, so a residual that changes with the wrist points at joint zero/sign errors.
Each teach: arm moved by hand (torque off) until the block sits centred between the OPEN jaws; joint
angles read over CAN (no motion). The block centre comes from a chest capture (bbox centre xy,
top - 3 cm in z = same definition as the first teach).

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/teach_fit.py grasp/runs/teach/teach.jsonl
    # teach.jsonl rows: {"q_deg": [7], "cap": "cap_YYYYmmdd_HHMMSS"}
Writes grasp/runs/teach/fit.json; with --write also grasp/config/openarm_<arm>_wrist_fit.json（--arm right で右腕）.
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
from ik.openarm_left_ik import OpenArmIK  # noqa: E402

T_BC = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
Z_WEIGHT = 0.3   # vertical placement of the jaws on the block is loosely defined -> down-weight z


def block_centre(cap):
    o = np.load(os.path.join(REPO, "grasp", "runs", cap, "object.npy"))
    b = o @ T_BC[:3, :3].T + T_BC[:3, 3]
    return np.array([(b[:, 0].min() + b[:, 0].max()) / 2, (b[:, 1].min() + b[:, 1].max()) / 2, b[:, 2].max() - 0.03])


def main():
    from scipy.optimize import least_squares
    ap = argparse.ArgumentParser()
    ap.add_argument("teach")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--arm", choices=["left", "right"], default="left",
                    help="教えた腕（2026-11-22: 右腕で真上から掴むと左へ 16 mm・爪の先が 3 cm 低くずれたので右でも当てはめる）")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.teach) if l.strip()]
    ik = OpenArmIK(os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf"), side=a.arm)
    Q = np.array([r["q_deg"] for r in rows], float)
    C = np.array([block_centre(r["cap"]) for r in rows])
    w = np.array([1, 1, Z_WEIGHT])

    def resid(x, joints, idx):
        p, dl = x[:3], np.zeros(7)
        dl[list(joints)] = x[3:]
        out = []
        for i in idx:
            M = ik.fk(np.deg2rad(Q[i] + dl))
            out.append((M.rotation @ p + M.translation - C[i]) * w)
        return np.concatenate(out)

    # J7 is not fitted: a J7 zero offset only rotates the TCP about link7's z axis, which the free TCP xy already
    # absorbs (verified on synthetic teaches: fit residual 0 with J7 folded into the TCP).
    models = {"tcp only": (), "tcp + J6": (5,), "tcp + J5,J6": (4, 5), "tcp + J4,J5,J6": (3, 4, 5),
              "tcp + J3,J5,J6": (2, 4, 5), "tcp + J2,J3,J4": (1, 2, 3)}
    report = {}
    p0 = np.array([0.0, 0.0, 0.2145])  # 北九州の爪の型の値（kitakyushu-collect.gripper.json）から始める
    for name, joints in models.items():
        nu = 3 + len(joints)
        if 3 * len(rows) < nu + 2:
            continue
        x0 = np.r_[p0, np.zeros(len(joints))]
        lb = np.r_[[-0.08, -0.08, 0.12], [-15] * len(joints)]; ub = np.r_[[0.08, 0.08, 0.30], [15] * len(joints)]
        f = least_squares(resid, x0, bounds=(lb, ub), args=(joints, range(len(rows))))
        per = np.linalg.norm(f.fun.reshape(-1, 3)[:, :2], axis=1) * 1000
        zres = f.fun.reshape(-1, 3)[:, 2] / Z_WEIGHT * 1000  # 高さは重みを軽くしているので、元の mm に戻して見る
        loo = []
        for k in range(len(rows)):  # leave-one-out: predict the held-out teach
            idx = [i for i in range(len(rows)) if i != k]
            if 3 * len(idx) < nu + 1:
                break
            g = least_squares(resid, f.x, bounds=(lb, ub), args=(joints, idx))
            loo.append(float(np.linalg.norm(resid(g.x, joints, [k])[:2]) * 1000))
        report[name] = {"tcp_in_link7_m": f.x[:3].round(4).tolist(),
                        "joint_offsets_deg": {f"joint_{j+1}": round(float(v), 2) for j, v in zip(joints, f.x[3:])},
                        "xy_resid_mm": per.round(1).tolist(), "z_resid_mm": zres.round(1).tolist(), "xy_rms_mm": float(np.sqrt(np.mean(per ** 2))),
                        "loo_xy_mm": [round(v, 1) for v in loo],
                        "loo_median_mm": float(np.median(loo)) if loo else None}
        print(f"{name:16s} rms {report[name]['xy_rms_mm']:5.1f} mm  LOO med "
              f"{report[name]['loo_median_mm'] if loo else float('nan'):5.1f} mm  tcp {report[name]['tcp_in_link7_m']}  "
              f"offsets {report[name]['joint_offsets_deg']}")
    # pick: the simplest model whose LOO median is within 1 mm of the best LOO median
    ok = {k: v for k, v in report.items() if v["loo_median_mm"] is not None}
    best = min(ok, key=lambda k: ok[k]["loo_median_mm"]) if ok else None
    pick = next((k for k in report if k in ok and ok[k]["loo_median_mm"] <= ok[best]["loo_median_mm"] + 1.0), best)
    out = {"n_teach": len(rows), "models": report, "picked": pick}
    os.makedirs(os.path.dirname(os.path.abspath(a.teach)), exist_ok=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(a.teach)), "fit.json"), "w"), indent=2)
    print("picked:", pick)
    if a.write and pick:
        cfg = {"_note": f"fit from {len(rows)} hand-taught poses ({a.teach}); model '{pick}'. joint offsets are added to "
                        "the lerobot joint reading to get the URDF angle (urdf = reading + offset).",
               "tcp_in_link7_m": report[pick]["tcp_in_link7_m"],
               "joint_offsets_deg": report[pick]["joint_offsets_deg"],
               "loo_median_mm": report[pick]["loo_median_mm"]}
        fn = os.path.join(REPO, "grasp", "config", f"openarm_{a.arm}_wrist_fit.json")
        json.dump(cfg, open(fn, "w"), indent=2)
        print(f"wrote {os.path.relpath(fn, REPO)}")


if __name__ == "__main__":
    main()
