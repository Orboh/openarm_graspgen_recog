#!/usr/bin/env python3
"""Step 3: object point cloud → grasp candidates (GraspGenX server client).

The GraspGenX server runs in its own env (~/work/GraspGenX/.venv). The pipeline side only needs
pyzmq / msgpack / msgpack-numpy / numpy (no torch).

Gripper specification:
  - `--gripper-name robotiq_2f_85` etc.: name-based (only grippers that exist in the server's assets)
  - `--sweep-json grasp/config/<hand>.json`: 12-number sweep volume params (works for an unregistered hand.
     This is the path for the custom hand: just measure the inner finger volume at open/half-open).

Output frame: same frame as the input point cloud (camera optical frame). Grasp convention = GraspGenX:
  +Z = approach direction, +X = closing direction, origin = gripper base (fingertip_depth behind the fingertips).

Server startup (for reference):
  cd ~/work/GraspGenX && .venv/bin/python client-server/graspgenx_server.py \
      --config ext/graspgenx_checkpoints/release \
      --assets_dir ext/gripper_descriptions/gripper_descriptions/assets --port 5556
"""
from __future__ import annotations

import json
import time

import numpy as np

# Mirror of the GraspGenX client wire format (graspgenx.serving.GraspGenXClient).
# Importing it would pull in the whole graspgenx package, so if it's not installed use only the minimal ZMQ part.


def _client(host: str, port: int):
    try:
        from graspgenx.serving import GraspGenXClient  # in the GraspGenX env
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "graspgenx.serving not found. Add ~/work/GraspGenX to PYTHONPATH "
            "or install pyzmq msgpack msgpack-numpy and use the GraspGenX client") from e
    return GraspGenXClient(host=host, port=port)


def infer_grasps(obj_pc: np.ndarray, *, gripper_name: str | None = None, sweep: dict | None = None,
                 host: str = "localhost", port: int = 5556, num_grasps: int = 200, topk: int = 50,
                 max_points: int = 3500, seed: int = 0, sweep_num: int | None = None):
    """Returns (grasps[M,4,4], scores[M]) sorted by score descending. On failure returns empty arrays (don't move the arm).

    sweep_num: 爪の箱（sweep）で呼ぶとき、1 回で出す候補の数。None なら GraspGenX の既定（100）。
    2026-11-21 北九州で測ると、100 個×15 回は 1.95 秒、1500 個×1 回は 0.79 秒だった。
    """
    pts = np.asarray(obj_pc, dtype=np.float32)
    if len(pts) > max_points:  # match the model's input size
        idx = np.random.default_rng(seed).choice(len(pts), max_points, replace=False)
        pts = pts[idx]
    t0 = time.time()
    with _client(host, port) as c:
        if sweep is not None and sweep_num:
            g, s = c.infer_object(pts, sweep, num_grasps=sweep_num, topk_num_grasps=sweep_num)
            topk = max(topk, sweep_num)
        elif sweep is not None:
            g, s = c.infer_object(pts, sweep)
        else:
            g, s = c.infer(pts, gripper_name=gripper_name, num_grasps=num_grasps,
                           grasp_threshold=-1.0, topk_num_grasps=topk)
    g, s = np.asarray(g), np.asarray(s)
    if len(s) == 0:
        return np.zeros((0, 4, 4), np.float32), np.zeros((0,), np.float32)
    order = np.argsort(-s)[:topk]
    print(f"[infer] {len(s)} grasps in {time.time() - t0:.2f}s, top score={s[order[0]]:.3f}")
    return g[order], s[order]


def load_sweep(path: str) -> dict:
    with open(path) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


def scene_cloud(cap_dir: str, obj_pc: np.ndarray, margin: float = 0.15) -> np.ndarray:
    """Whole-scene cloud (camera frame) for GraspNet-baseline: every depth pixel within `margin` of the object's
    bounding box in x/y and up to `margin` behind it, so the floor and neighbours are in it (its collision check needs them)."""
    import os
    K = json.load(open(os.path.join(cap_dir, "intrinsics.json")))
    d = np.load(os.path.join(cap_dir, "depth.npy")).astype(np.float32)
    if d.max() > 50:  # raw units
        d = d * float(K.get("depth_scale", 0.001))
    v, u = np.nonzero(d > 0)
    z = d[v, u]
    P = np.stack([(u - K["ppx"]) * z / K["fx"], (v - K["ppy"]) * z / K["fy"], z], 1)
    lo, hi = obj_pc.min(0) - margin, obj_pc.max(0) + margin
    keep = np.all((P[:, :2] >= lo[:2]) & (P[:, :2] <= hi[:2]), 1) & (P[:, 2] <= hi[2]) & (P[:, 2] >= 0.2)
    return P[keep].astype(np.float32)


def infer_graspnet(scene_pc: np.ndarray, *, host: str = "localhost", port: int = 5557, seed: int = 0,
                   collision_thresh: float = 0.01, max_width: float = 0.10):
    """GraspNet-baseline server (grasp/tools/graspnet_server.py) -> (grasps[M,4,4] GraspGenX convention, scores[M]).
    Drops grasps wider than `max_width` (GraspNet was trained for another gripper; OpenArm opens 110 mm, keep 1 cm spare).
    On failure returns empty arrays (don't move the arm)."""
    import msgpack
    import msgpack_numpy as mn
    import zmq
    t0 = time.time()
    sock = zmq.Context.instance().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 60000); sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    try:
        sock.send(msgpack.packb({"points": np.asarray(scene_pc, np.float32), "seed": seed,
                                 "collision_thresh": collision_thresh}, default=mn.encode, use_bin_type=True))
        rep = msgpack.unpackb(sock.recv(), object_hook=mn.decode, raw=False)
    finally:
        sock.close()
    if "error" in rep:
        print(f"[infer] graspnet error: {rep['error']}")
        return np.zeros((0, 4, 4), np.float32), np.zeros((0,), np.float32)
    G, S, W = np.asarray(rep["G"]), np.asarray(rep["S"]), np.asarray(rep["W"])
    ok = W <= max_width
    print(f"[infer] graspnet {len(S)} grasps in {time.time() - t0:.2f}s, {int(ok.sum())} within width {max_width * 1000:.0f} mm, "
          f"top score={S[ok][0] if ok.any() else float('nan'):.3f}")
    return G[ok], S[ok]
