#!/usr/bin/env python3
"""Where are the fingertips really? Compare the FK fingertip with the chest-camera point cloud. **Does not move the arm.**

2026-11-22 (Sota: 左手を使えるようにしよ). The method used for the right arm at 00:48 (handeye_near_20261122_004818),
made into a tool for both arms (GRASP_ARM=left|right):
  - points within 20 cm of the hand, above floor + 12 cm, from the chest RGB-D of each sample (handeye_near.py output)
  - the connected cluster (8 mm) that contains the flange = the hand + fingers
  - fingertip = mean of the cluster points within 2 cm of the farthest point along the jaw axis
  - difference to the FK fingertip, in the base frame and in the gripper frame (x: across the jaws, y: closing, z: jaw axis)
Writes tcp_check.json + clu_<k>.jpg (green: cluster, blue: measured tip, red: FK tip) into the sample folder.

    GRASP_ARM=left ~/work/GraspGenX/.venv/bin/python grasp/tools/tcp_check.py <handeye_near_dir> [--skip 0,10]
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from grasp.plan import CFG, T_L7_G  # noqa: E402
from ik.openarm_right_ik import arm_ik_class  # noqa: E402

FLOOR = float(os.environ.get("GRASP_FLOOR_Z", "0.28"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--skip", default="", help="sample indices to ignore (e.g. the standby poses)")
    ap.add_argument("--t-base-cam", default=os.path.join(REPO, "grasp/config/T_base_cam.json"))
    a = ap.parse_args()
    skip = {int(v) for v in a.skip.split(",") if v}
    ik = arm_ik_class()(); tipz = CFG["flange_to_tip_m"]
    s = np.load(os.path.join(a.dir, "samples.npz")); Q, K = s["q_deg"], s["K"]
    T = np.array(json.load(open(a.t_base_cam))["T_base_cam"]); Ti = np.linalg.inv(T)
    rows = []
    for k in range(len(Q)):
        if k in skip:
            continue
        M = ik.fk(np.deg2rad(Q[k])); G = np.eye(4); G[:3, :3] = M.rotation; G[:3, 3] = M.translation; G = G @ T_L7_G
        tip = G[:3, 3] + G[:3, 2] * tipz; fl = G[:3, 3]
        dep = np.load(os.path.join(a.dir, f"depth_{k:03d}.npy")).astype(np.float32)
        vv, uu = np.nonzero(dep > 0); z = dep[vv, uu]
        P = np.c_[(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z] @ T[:3, :3].T + T[:3, 3]
        sel = (np.linalg.norm(P - (fl + tip) / 2, axis=1) < 0.20) & (P[:, 2] > FLOOR + 0.12 - 0.06)
        Ps = P[sel][::2]; px = np.c_[uu, vv][sel][::2]
        if len(Ps) < 100:
            print(f"#{k}: too few points near the hand ({len(Ps)})"); continue
        tr = cKDTree(Ps); pairs = tr.query_pairs(0.008, output_type="ndarray")
        A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(Ps), len(Ps)))
        _, lab = connected_components(A, directed=False)
        dfl, ifl = tr.query(fl)
        C = Ps[lab == lab[ifl]]; Cp = px[lab == lab[ifl]]
        ax = G[:3, 2]; proj = (C - fl) @ ax
        tz = C[proj > proj.max() - 0.02].mean(0)
        d_base = tz - tip; d_grip = G[:3, :3].T @ d_base
        # 00:48 lesson: along the camera ray the cloud shows the near face of the fingers (2-3 cm bias); only the part
        # across the ray (what the image shows) is trustworthy
        r = tip - T[:3, 3]; r /= np.linalg.norm(r); d_perp = d_base - r * (d_base @ r)
        rows.append({"k": k, "q_deg": [round(float(v), 1) for v in Q[k]],
                     "tilt_deg": round(float(np.degrees(np.arccos(np.clip(-ax[2], -1, 1)))), 1),
                     "cluster_pts": int(len(C)), "flange_gap_m": round(float(dfl), 3),
                     "reach_obs_m": round(float(proj.max()), 3), "reach_fk_m": round(float(tipz), 3),
                     "tip_fk": tip.round(4).tolist(), "tip_obs": tz.round(4).tolist(),
                     "d_base_mm": (d_base * 1000).round(1).tolist(), "d_grip_mm": (d_grip * 1000).round(1).tolist(),
                     "d_across_ray_mm": (d_perp * 1000).round(1).tolist(), "across_ray_norm_mm": round(float(np.linalg.norm(d_perp) * 1000), 1),
                     "d_across_ray_grip_mm": (G[:3, :3].T @ d_perp * 1000).round(1).tolist()})
        col = cv2.imread(os.path.join(a.dir, f"color_{k:03d}.jpg"))
        for (u, v) in Cp[::3]:
            col[v, u] = (0, 255, 0)
        for p, c in ((tz, (255, 0, 0)), (tip, (0, 0, 255))):
            q = Ti[:3, :3] @ p + Ti[:3, 3]
            cv2.circle(col, (int(K[0, 0] * q[0] / q[2] + K[0, 2]), int(K[1, 1] * q[1] / q[2] + K[1, 2])), 7, c, 2)
        cv2.putText(col, f"#{k} tilt {rows[-1]['tilt_deg']:.0f}  d {np.linalg.norm(d_base)*1000:.0f} mm", (8, 22), 0, 0.6, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(a.dir, f"clu_{k}.jpg"), col)
        print(f"#{k:2d} tilt {rows[-1]['tilt_deg']:4.0f}  cluster {len(C):5d}  reach obs {proj.max():.3f} / fk {tipz:.3f}  "
              f"across the ray {np.linalg.norm(d_perp)*1000:4.0f} mm (gripper {np.round(G[:3, :3].T @ d_perp*1000)})  [3D d grip {np.round(d_grip*1000)}]")
    if rows:
        D = np.array([r["d_grip_mm"] for r in rows]); B = np.array([r["d_base_mm"] for r in rows])
        summ = {"n": len(rows), "median_d_grip_mm": np.median(D, 0).round(1).tolist(), "median_d_base_mm": np.median(B, 0).round(1).tolist(),
                "spread_d_grip_mm": D.std(0).round(1).tolist(), "max_norm_mm": round(float(np.linalg.norm(B, axis=1).max()), 1),
                "across_ray_median_mm": round(float(np.median([r["across_ray_norm_mm"] for r in rows])), 1),
                "across_ray_max_mm": round(float(max(r["across_ray_norm_mm"] for r in rows)), 1)}
        print("[tcp] median d (gripper frame) mm", summ["median_d_grip_mm"], " spread", summ["spread_d_grip_mm"],
              " | across the ray: median", summ["across_ray_median_mm"], "mm, worst", summ["across_ray_max_mm"], "mm")
        json.dump({"arm": os.environ.get("GRASP_ARM", "left"), "rows": rows, "summary": summ},
                  open(os.path.join(a.dir, "tcp_check.json"), "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
