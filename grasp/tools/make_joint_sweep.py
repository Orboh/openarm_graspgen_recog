#!/usr/bin/env python3
"""Make a one-joint-at-a-time sweep pose list for kinematic calibration of the left arm. **Does not move the arm.**

Why (2026-11-20): with the ArUco hand-eye data, each pose cluster fits to 1-2 mm on its own but the two
clusters together miss by 14-25 mm, i.e. the arm model (joint zeros / URDF) is off between configurations.
Eleven frames in two clusters could not tell which joint. Sweeping one joint at a time puts the marker on a
circle about that joint's axis, which pins down that joint's zero.

Picks, per pose cluster, the detected frame whose marker is nearest the image centre as the base, then for
each of J2..J6 visits base +15, +30, base, -15, -30, base. Keeps a pose only if
  - the marker is predicted inside the image (margin) by that cluster's own camera+marker fit (1-2 mm there),
  - TCP >= floor + 6 cm and link7 >= floor + 10 cm (as handeye_near.py), along the path from the previous pose too.
Writes a JSON list for `handeye_near.py --poses <json> [--go]`.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/make_joint_sweep.py <handeye run dir> -o sweep.json
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
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402

FLOOR_Z = float(os.environ.get("GRASP_FLOOR_Z", "0.27")) + 0.02         # same as handeye_near.py
if os.environ.get("GRASP_SITE") and "GRASP_FLOOR_Z" not in os.environ:  # 東京の台車以外で床の高さを決め忘れたら動かさない
    sys.exit("[grasp] GRASP_FLOOR_Z が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の値のまま床の確認をしない。grasp/config/sites/ で決める")
LO = np.degrees([-3.45, -3.30, -1.50, -0.01, -1.50, -0.75, -1.50]) + 3
HI = np.degrees([1.35, 0.15, 1.50, 2.40, 1.50, 0.75, 1.50]) - 3
STEPS = (10, 20, 30, 0, -10, -20, -30, 0)
MARGIN_PX = 60


def main() -> int:
    from scipy.optimize import least_squares
    from scipy.spatial.transform import Rotation as R
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="hand-eye run with samples.npz, color_/depth_ frames")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--joints", default="2,3,4,5,6")
    ap.add_argument("--bases", default=None, help="JSON {cluster: q_deg}: use these base poses (e.g. wrist turned so the marker faces the camera)")
    ap.add_argument("--marker-link7", default=None, help="x,y,z [m] marker point in link7 (e.g. moved to the fingertip); default = fitted from the run")
    ap.add_argument("--j5-bend-j6", type=float, default=0.0, help="for the J5 sweep, bend J6 by this much [deg] so the marker has a lever arm about J5")
    ap.add_argument("--raise-base", type=float, default=0.04, help="lift the base pose this much (same orientation) for floor room [m]")
    a = ap.parse_args()
    import cv2
    s = np.load(os.path.join(a.run_dir, "samples.npz")); Q, K = s["q_deg"], s["K"]
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    P, QQ, UV = [], [], []
    for k in range(len(Q)):
        col = cv2.imread(os.path.join(a.run_dir, f"color_{k:03d}.jpg"))
        dep = np.load(os.path.join(a.run_dir, f"depth_{k:03d}.npy")).astype(np.float32)
        cs, ids, _ = det.detectMarkers(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
        if ids is None or 0 not in ids.ravel():
            continue
        c = cs[list(ids.ravel()).index(0)].reshape(4, 2); u, v = c.mean(0)
        zs = dep[int(c[:, 1].min()):int(c[:, 1].max()) + 1, int(c[:, 0].min()):int(c[:, 0].max()) + 1]; zs = zs[zs > 0]
        if len(zs) < 10:
            continue
        z = float(np.median(zs))
        P.append([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]); QQ.append(Q[k]); UV.append((u, v))
    P, QQ, UV = np.array(P), np.array(QQ), np.array(UV)
    ik = OpenArmLeftIK()

    def fk(q):
        M = ik.fk(np.deg2rad(q)); return M.rotation, M.translation

    def res(x, idx):
        Rc = R.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; off = x[6:9]
        return np.concatenate([Rc @ P[i] + t - (fk(QQ[i])[0] @ off + fk(QQ[i])[1]) for i in idx])

    Tbc0 = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    x0 = np.r_[R.from_matrix(Tbc0[:3, :3]).as_rotvec(), Tbc0[:3, 3], [0, 0.05, 0.11]]
    clusters = {"grasp-like": [i for i in range(len(P)) if QQ[i][0] < -40], "wrist-forward": [i for i in range(len(P)) if QQ[i][0] >= -40]}
    out_poses, info = {}, []
    joints = [int(j) for j in a.joints.split(",")]

    marker_pt = [np.zeros(3)]

    def path_ok(qa, qb):
        for s_ in np.linspace(0, 1, 20):
            q = np.asarray(qa) + (np.asarray(qb) - np.asarray(qa)) * s_
            Rl, tl = fk(q); tcp = tl + Rl @ [0, 0, 0.2145]; mk = tl + Rl @ marker_pt[0]
            if tcp[2] < FLOOR_Z + 0.06 or tl[2] < FLOOR_Z + 0.10 or mk[2] < FLOOR_Z + 0.06:
                return False
        return True

    for name, idx in clusters.items():
        if len(idx) < 3:
            continue
        f = least_squares(res, x0, args=(idx,), loss="soft_l1", f_scale=0.01)
        rms = np.sqrt(np.mean(np.linalg.norm(f.fun.reshape(-1, 3), axis=1) ** 2)) * 1000
        Rc = R.from_rotvec(f.x[:3]).as_matrix(); t = f.x[3:6]; off = f.x[6:9]
        if a.marker_link7:
            off = np.array([float(v) for v in a.marker_link7.split(",")])
        Rcb, tcb = Rc.T, -Rc.T @ t  # base -> camera
        b = min(idx, key=lambda i: np.hypot(UV[i][0] - K[0, 2], UV[i][1] - K[1, 2]))
        base = QQ[b]
        if a.bases:
            base = np.array(json.load(open(a.bases))["bases"][name], float)
        if a.raise_base > 0 and not a.bases:  # same hand orientation, a little higher: more room above the floor for the sweeps
            import pinocchio as pin
            M = ik.fk(np.deg2rad(base))
            qb, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, a.raise_base])), np.deg2rad(base), position_only=False)
            if ok and ik.within_limits(qb) and np.max(np.abs(np.rad2deg(qb) - base)) < 25:
                base = np.rad2deg(qb)

        def project(q):
            Rl, tl = fk(q); pc = Rcb @ (Rl @ off + tl) + tcb
            return K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]
        marker_pt[0] = off
        prev = base; out_poses[name] = [[round(float(v_), 2) for v_ in base]]
        print(f"[sweep] {name}: base from frame with marker at px {np.round(UV[b]).tolist()}, local fit {rms:.1f} mm, base q {np.round(base, 1).tolist()}")
        for j in joints:
            jb = base.copy()
            if j == 5 and a.j5_bend_j6:  # lever arm about J5: bend the wrist (pick the side that keeps the marker in view)
                cands = [base.copy(), base.copy()]
                cands[0][5] = np.clip(base[5] + a.j5_bend_j6, LO[5], HI[5]); cands[1][5] = np.clip(base[5] - a.j5_bend_j6, LO[5], HI[5])
                def centred(q):
                    u, v = project(q); return np.hypot(u - K[0, 2], v - K[1, 2])
                jb = min(cands, key=centred)
                if path_ok(prev, jb) and path_ok(jb, jb):
                    out_poses[name].append([round(float(v_), 2) for v_ in jb]); prev = jb
                    info.append({"cluster": name, "joint": 6, "step": 0, "kept": True, "px": [round(x) for x in project(jb)], "note": "J5 lever base"})
                else:
                    jb = base.copy()
            for st in STEPS:
                q = jb.copy(); q[j - 1] = np.clip(jb[j - 1] + st, LO[j - 1], HI[j - 1])
                u, v = project(q)
                vis = MARGIN_PX <= u <= 640 - MARGIN_PX and MARGIN_PX <= v <= 480 - MARGIN_PX
                why = None if vis else "marker out of view"
                if why is None and not path_ok(q, q):
                    why = "pose too close to the floor"
                if why is None and not path_ok(prev, q):
                    why = "path from the previous pose too close to the floor"
                if why:
                    info.append({"cluster": name, "joint": j, "step": st, "kept": False, "px": [round(u), round(v)], "why": why})
                    continue
                out_poses[name].append([round(float(v_), 2) for v_ in q]); prev = q
                info.append({"cluster": name, "joint": j, "step": st, "kept": True, "px": [round(u), round(v)]})
    kept = [i for i in info if i["kept"]]
    for name, poses in out_poses.items():  # one file per cluster: each run goes start -> raise -> these -> raise -> start
        fn = a.out.replace(".json", f"_{name}.json")
        json.dump({"poses": poses, "info": [i for i in info if i["cluster"] == name]}, open(fn, "w"), indent=1)
        print(f"[sweep] {name}: {len(poses)} poses -> {fn}")
        for j in joints:
            ks = [i["step"] for i in kept if i["cluster"] == name and i["joint"] == j and i["step"] != 0]
            print(f"   J{j}: steps kept {ks}")
    for i in info:
        if not i["kept"] and i["step"] != 0:
            print(f"   dropped {i['cluster']} J{i['joint']} {i['step']:+d}: {i['why']} (px {i['px']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
