#!/usr/bin/env python3
"""Offline smoke test for Steps 1–3 (no robot/camera needed).

Using a depth image (depth.npy [m] + K) from a real RGB-D camera:
  depth → point cloud (capture.depth_to_points) → plane removal + DBSCAN (segment) → GraspGenX (infer)
and saves the top grasps as a PLY (object + gripper axes).

Default input is a GraspGenX real-world sample (D4xx-class depth). Pass the output of grasp/capture.py with --cap.

    PYTHONPATH=~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/tools/smoke_offline.py \
        --sample ~/work/GraspGenX/assets/sample_data/real_world/00 --gripper-name robotiq_2f_85
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grasp.capture import depth_to_points  # noqa: E402
from grasp.segment import SegParams, segment_largest_object  # noqa: E402
from grasp.infer import infer_grasps, load_sweep  # noqa: E402


def axes_points(T: np.ndarray, L: float = 0.06, n: int = 30):
    """Draw the grasp pose as points (x=red closing, y=green, z=blue approach)."""
    t = np.linspace(0, L, n)[:, None]
    out, col = [], []
    for k, c in enumerate([[1, 0, 0], [0, 1, 0], [0, 0, 1]]):
        out.append(T[:3, 3] + t * T[:3, k])
        col.append(np.tile(c, (n, 1)))
    return np.vstack(out), np.vstack(col)


def main():
    import open3d as o3d

    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample", help="GraspGenX sample dir (depth.npy + meta_data.json)")
    g.add_argument("--cap", help="output dir of grasp/capture.py")
    ap.add_argument("--gripper-name", default=None)
    ap.add_argument("--sweep-json", default=None)
    ap.add_argument("--zmax", type=float, default=1.5)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    src = a.sample or a.cap
    depth = np.load(os.path.join(src, "depth.npy")).astype(np.float32)
    if a.sample:
        K = np.array(json.load(open(os.path.join(src, "meta_data.json")))["intrinsics"])
    else:
        K = np.array(json.load(open(os.path.join(src, "intrinsics.json")))["K"])
    if np.nanmax(depth) > 20:  # mm → m
        depth = depth / 1000.0
    pts, _ = depth_to_points(depth, K, zmax=a.zmax)
    print(f"[step1] scene points={len(pts)}")

    obj, dbg = segment_largest_object(pts, SegParams(zmax=a.zmax), return_debug=True)
    print(f"[step2] object pts={len(obj)} extent={np.round(obj.max(0) - obj.min(0), 3)} {dbg}")

    sweep = load_sweep(a.sweep_json) if a.sweep_json else None
    grasps, scores = infer_grasps(obj, gripper_name=a.gripper_name, sweep=sweep)
    if len(scores) == 0:
        print("[step3] NO GRASPS"); sys.exit(1)
    print(f"[step3] top5 scores={np.round(scores[:5], 3)}")
    # Sanity: the grasp origin + approach direction should end near the object
    d = [np.min(np.linalg.norm(obj - (G[:3, 3] + 0.10 * G[:3, 2]), axis=1)) for G in grasps[:10]]
    print(f"[step3] top10 dist(base+10cm*approach → object) [m] = {np.round(d, 3)}")

    out = a.out or os.path.join(os.environ.get("CLAUDE_JOB_DIR", "/tmp"), "smoke_grasps.ply")
    P, C = [obj], [np.tile([0.6, 0.6, 0.6], (len(obj), 1))]
    for G in grasps[:5]:
        p, c = axes_points(G); P.append(p); C.append(c)
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(P).astype(np.float64)))
    pc.colors = o3d.utility.Vector3dVector(np.vstack(C).astype(np.float64))
    o3d.io.write_point_cloud(out, pc)
    np.savez(out.replace(".ply", ".npz"), grasps=grasps, scores=scores, obj=obj)
    print(f"[ok] wrote {out}")


if __name__ == "__main__":
    main()
