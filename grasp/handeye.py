#!/usr/bin/env python3
"""Step 4: estimate the chest camera's pose (T_base_cam) from the samples collected by handeye_collect.py.

What it does:
  1. Detect black blobs (the wrist D405 box) in each frame → 3D point in the camera frame using depth (multiple candidates allowed)
  2. FK of the left arm's link7 from the joint angles (ik/openarm_left_ik.py)
  3. Unknowns: T_base_cam (6) + box centre in link7 (3).
     Minimise || T_base_cam·p_cam − T_base_link7(q)·p_off || with RANSAC → least_squares
  4. Write grasp/config/T_base_cam.json and handeye_check.jpg (green = predicted, red = detected)

    python grasp/handeye.py ~/grasp_runs/handeye_<ts> --repo ~/projects/openarm-orboh
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as Rot


def detect_dark_blobs(color, depth, K, vmax=60, size_m=(0.02, 0.10), z_m=(0.25, 1.0)):
    hsv = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    mask = ((hsv[..., 2] < vmax) & (depth > 0)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask)
    out = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 150:
            continue
        zs = depth[lab == i]
        z = float(np.median(zs[zs > 0]))
        if not (z_m[0] <= z <= z_m[1]):
            continue
        size = np.sqrt(area) * z / K[0, 0]
        if not (size_m[0] <= size <= size_m[1]):
            continue
        u, v = cent[i]
        p = np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])
        out.append((p, (int(u), int(v)), size))
    return out


def detect_aruco(color, depth, K, marker_id=0):
    """ArUco (DICT_4X4_50) centre → 3D point in the camera frame from depth. Returns [] if not found."""
    det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    corners, ids, _ = det.detectMarkers(cv2.cvtColor(color, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return []
    out = []
    for c, i in zip(corners, ids.ravel()):
        if int(i) != marker_id:
            continue
        c = c.reshape(4, 2); u, v = c.mean(0)
        x0, x1 = int(c[:, 0].min()), int(c[:, 0].max()) + 1; y0, y1 = int(c[:, 1].min()), int(c[:, 1].max()) + 1
        zs = depth[y0:y1, x0:x1]; zs = zs[zs > 0]
        if len(zs) < 10:
            continue
        z = float(np.median(zs))
        out.append((np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z]), (int(u), int(v)), 0.05))
    return out


def residuals(x, P_cam, T_l7):
    R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; off = x[6:9]
    a = P_cam @ R.T + t
    b = np.einsum("nij,j->ni", T_l7[:, :3, :3], off) + T_l7[:, :3, 3]
    return (a - b).ravel()


def fit(P_cam, T_l7, x0):
    r = least_squares(residuals, x0, args=(P_cam, T_l7), loss="soft_l1", f_scale=0.01, max_nfev=400)
    return r.x


def per_frame_err(x, cands, T_l7):
    R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]; off = x[6:9]
    errs, pick = [], []
    for c, T in zip(cands, T_l7):
        b = T[:3, :3] @ off + T[:3, 3]
        e = [np.linalg.norm(R @ p + t - b) for p, _, _ in c] or [np.inf]
        k = int(np.argmin(e)); errs.append(e[k]); pick.append(k)
    return np.array(errs), pick


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(__file__), ".."))
    ap.add_argument("--thresh", type=float, default=0.02)
    ap.add_argument("--iters", type=int, default=600)
    ap.add_argument("--detector", choices=["aruco", "dark"], default="aruco")
    ap.add_argument("--marker-id", type=int, default=0,
                    help="ArUco（DICT_4X4_50）の番号。東京は 0、北九州の 2x2 の板は 24〜27 のどれか（2026-11-21）")
    ap.add_argument("--arm", choices=["left", "right"], default="left",
                    help="マーカーを付けた腕（2026-11-22: 右腕の運動学の確かめに、右で解いた結果を左の結果と比べる）")
    a = ap.parse_args()
    sys.path.insert(0, a.repo)
    from ik.openarm_left_ik import OpenArmIK

    s = np.load(os.path.join(a.run_dir, "samples.npz"))
    Q, K = s["q_deg"], s["K"]
    ik = OpenArmIK(os.path.join(a.repo, "urdf", "openarm_v10_bimanual.urdf"), side=a.arm)
    T_l7, cands, imgs = [], [], []
    for k in range(len(Q)):
        M = ik.fk(np.deg2rad(Q[k])); T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation
        col = cv2.imread(os.path.join(a.run_dir, f"color_{k:03d}.jpg"))
        dep = np.load(os.path.join(a.run_dir, f"depth_{k:03d}.npy")).astype(np.float32)
        c = detect_aruco(col, dep, K, marker_id=a.marker_id) if a.detector == "aruco" else detect_dark_blobs(col, dep, K)
        if c:
            T_l7.append(T); cands.append(c); imgs.append(col)
    T_l7 = np.array(T_l7)
    print(f"[handeye] samples {len(Q)}, with detections {len(cands)}, candidates per frame " +
          str([len(c) for c in cands]))
    if len(cands) < 5:
        print("[handeye] too few detections (need ≥5). Collect again with poses where the black box is visible"); return 1

    rng = np.random.default_rng(0)
    best = (None, -1, np.inf)
    for it in range(a.iters):
        idx = rng.choice(len(cands), size=min(6, len(cands)), replace=False)
        P = np.array([cands[i][rng.integers(len(cands[i]))][0] for i in idx])
        x0 = np.r_[Rot.random(random_state=int(rng.integers(1 << 30))).as_rotvec(), rng.normal(0, 0.3, 3), [0, 0, 0.1]]
        x = fit(P, T_l7[idx], x0)
        if np.linalg.norm(x[6:9]) > 0.3:
            continue
        e, _ = per_frame_err(x, cands, T_l7)
        nin = int(np.sum(e < a.thresh)); med = float(np.median(e[e < a.thresh])) if nin else np.inf
        if nin > best[1] or (nin == best[1] and med < best[2]):
            best = (x, nin, med)
    x, nin, _ = best
    e, pick = per_frame_err(x, cands, T_l7)
    inl = np.where(e < a.thresh)[0]
    P = np.array([cands[i][pick[i]][0] for i in inl])
    x = least_squares(residuals, x, args=(P, T_l7[inl])).x
    e, pick = per_frame_err(x, cands, T_l7)
    inl = np.where(e < a.thresh)[0]
    rms = float(np.sqrt(np.mean(e[inl] ** 2))) if len(inl) else np.inf
    R = Rot.from_rotvec(x[:3]).as_matrix(); T_bc = np.eye(4); T_bc[:3, :3] = R; T_bc[:3, 3] = x[3:6]
    print(f"[handeye] inliers {len(inl)}/{len(cands)}  RMS {rms*1000:.1f} mm  per-frame error mm = {np.round(e*1000,1).tolist()}")
    print(f"[handeye] camera position (base frame) = {np.round(x[3:6],3).tolist()}  box offset (link7) = {np.round(x[6:9],3).tolist()}")

    res = {"T_base_cam": T_bc.tolist(), "box_offset_link7_m": x[6:9].tolist(), "rms_m": rms,
           "inliers": int(len(inl)), "frames": int(len(cands)), "arm": a.arm, "run_dir": os.path.abspath(a.run_dir),
           "created": time.strftime("%Y-%m-%d %H:%M:%S"),
           "note": "base = root of urdf/openarm_v10_bimanual.urdf (ik/openarm_left_ik.py). cam = RealSense colour optical frame."}
    json.dump(res, open(os.path.join(a.run_dir, "T_base_cam.json"), "w"), indent=2)

    # check image: predicted (green) vs detected (red)
    Tcb = np.linalg.inv(T_bc); tiles = []
    for i in range(len(cands)):
        img = imgs[i].copy()
        b = T_l7[i][:3, :3] @ x[6:9] + T_l7[i][:3, 3]
        pc = Tcb[:3, :3] @ b + Tcb[:3, 3]
        if pc[2] > 0:
            u, v = (K @ pc)[:2] / pc[2]
            cv2.circle(img, (int(u), int(v)), 9, (0, 255, 0), 2)
        cv2.circle(img, cands[i][pick[i]][1], 4, (0, 0, 255), -1)
        cv2.putText(img, f"#{i} err {e[i]*1000:.0f}mm", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if e[i] < a.thresh else (0, 0, 255), 2)
        tiles.append(cv2.resize(img, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)])
    cv2.imwrite(os.path.join(a.run_dir, "handeye_check.jpg"), grid)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
