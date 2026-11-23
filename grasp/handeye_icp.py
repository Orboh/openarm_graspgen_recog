#!/usr/bin/env python3
"""Step 4 (markerless): fit the arm's 3D shape to the depth point cloud to find the chest camera's pose (T_base_cam).

For each frame, place the left arm links' (link3–7) URDF mesh surface points with FK from that frame's joint angles,
and solve for the one camera pose that best fits all frames' depth clouds (truncated ICP).
No marker needed. The custom jaws differ from the URDF, so they're left out (link7 and above only).

    python grasp/handeye_icp.py grasp/runs/handeye_<ts>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from urdf_fk import LeftArmFK  # noqa: E402

LINKS = [f"openarm_left_link{i}" for i in range(3, 8)]


def depth_cloud(depth, K, zmax=1.1, step=3):
    d = depth[::step, ::step].astype(np.float32)
    v, u = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    ok = (d > 0.2) & (d < zmax)
    z = d[ok]; x = (u[ok] - K[0, 2]) * z / K[0, 0]; y = (v[ok] - K[1, 2]) * z / K[1, 1]
    return np.stack([x, y, z], 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--trunc", type=float, default=0.03)
    ap.add_argument("--starts", type=int, default=60)
    a = ap.parse_args()
    s = np.load(os.path.join(a.run_dir, "samples.npz")); Q, K = s["q_deg"], s["K"]
    fk = LeftArmFK()
    S = fk.surface_points(LINKS, n_per_link=400)
    frames = []
    for k in range(len(Q)):
        P = fk.link_poses(np.deg2rad(Q[k]))
        mb = np.vstack([S[l] @ P[l][:3, :3].T + P[l][:3, 3] for l in S])  # model in base frame
        dep = np.load(os.path.join(a.run_dir, f"depth_{k:03d}.npy")).astype(np.float32)
        sc = depth_cloud(dep, K)
        frames.append((mb, cKDTree(sc), sc))
    print(f"[icp] {len(frames)} frames, model {len(frames[0][0])} pts/frame")

    def res(x):
        R = Rot.from_rotvec(x[:3]).as_matrix(); t = x[3:6]  # base->cam
        out = []
        for mb, tree, _ in frames:
            mc = mb @ R.T + t
            vis = mc[:, 2] > 0.15
            d, _ = tree.query(mc[vis], distance_upper_bound=a.trunc)
            d = np.where(np.isfinite(d), d, a.trunc)
            r = np.full(len(mc), a.trunc); r[vis] = d
            out.append(r)
        return np.concatenate(out)

    def make_x(cam_pos, pitch_deg, yaw_deg=0.0):
        # camera optical frame in base: z forward-down, x = robot's right (-y), y = image down
        p, yw = np.radians(pitch_deg), np.radians(yaw_deg)
        z = np.array([np.cos(p) * np.cos(yw), np.cos(p) * np.sin(yw), -np.sin(p)])
        xr = np.array([np.sin(yw), -np.cos(yw), 0.0])
        y = np.cross(z, xr)
        R_bc = np.stack([xr, y, z], 1)          # cam axes in base
        R_cb = R_bc.T
        return np.r_[Rot.from_matrix(R_cb).as_rotvec(), -R_cb @ np.asarray(cam_pos)]

    base_trunc = a.trunc
    def cost(x, tr):
        a.trunc = tr; r = res(x); a.trunc = base_trunc; return float(np.mean(r ** 2))
    cands = []
    for px in (-0.1, 0.0, 0.1, 0.2):
        for py in (-0.1, 0.0, 0.1, 0.2):
            for pz in (0.5, 0.65, 0.8, 0.95):
                for pitch in (30, 45, 60, 75, 90):
                    for yaw in (-20, 0, 20):
                        x0 = make_x([px, py, pz], pitch, yaw); cands.append((cost(x0, 0.10), x0))
    cands.sort(key=lambda c: c[0])
    print(f"[icp] grid {len(cands)} starts; best coarse rms {np.sqrt(cands[0][0])*1000:.0f} mm", flush=True)
    best = (None, np.inf)
    for c0, x0 in cands[:a.starts]:
        x = x0
        for tr in (0.10, 0.05, 0.03, 0.015):
            a.trunc = tr
            x = least_squares(res, x, max_nfev=80, x_scale=[0.3] * 3 + [0.1] * 3).x
        a.trunc = 0.015; r = res(x); cst = float(np.mean(r ** 2))
        if cst < best[1]:
            best = (x, cst); print(f"  rms(1.5cm) {np.sqrt(cst)*1000:.1f} mm  within1cm {np.mean(r<0.01):.0%}", flush=True)
    x = best[0]; a.trunc = 0.015
    rf = res(x); rms = float(np.sqrt(np.mean(rf ** 2))); inl = float(np.mean(rf < 0.01))
    R = Rot.from_rotvec(x[:3]).as_matrix(); T_cb = np.eye(4); T_cb[:3, :3] = R; T_cb[:3, 3] = x[3:6]
    T_bc = np.linalg.inv(T_cb)  # cam pose in base
    print(f"[icp] rms(trunc {a.trunc*100:.0f}cm) {rms*1000:.1f} mm, model points within 1 cm {inl:.0%}")
    print(f"[icp] camera position in base frame {np.round(T_bc[:3,3],3).tolist()}")
    json.dump({"T_base_cam": T_bc.tolist(), "rms_trunc_m": rms, "frac_within_1cm": inl, "method": "urdf-icp links3-7",
               "run_dir": os.path.abspath(a.run_dir), "created": time.strftime("%F %T")},
              open(os.path.join(a.run_dir, "T_base_cam_icp.json"), "w"), indent=2)
    # check image: project the model onto each frame
    tiles = []
    for k, (mb, _, _) in enumerate(frames):
        img = cv2.imread(os.path.join(a.run_dir, f"color_{k:03d}.jpg"))
        mc = mb @ R.T + x[3:6]; mc = mc[mc[:, 2] > 0.1]
        uv = (mc @ K.T); uv = (uv[:, :2] / uv[:, 2:]).astype(int)
        for u, v in uv[::2]:
            if 0 <= u < 640 and 0 <= v < 480:
                cv2.circle(img, (u, v), 1, (0, 255, 0), -1)
        cv2.putText(img, f"#{k}", (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        tiles.append(cv2.resize(img, (320, 240)))
    while len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(os.path.join(a.run_dir, "handeye_icp_check.jpg"), np.vstack([np.hstack(tiles[r:r + 4]) for r in range(0, len(tiles), 4)]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
