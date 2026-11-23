#!/usr/bin/env python3
"""GraspNet-baseline grasp server (step 7 of Kota's task: the comparison model). **Does not move the arm.**

Runs in its own conda env (torch 2.4 cu121 + the repo's pointnet2 / knn CUDA ops), like the GraspGenX server:

    source ~/miniconda3/etc/profile.d/conda.sh && conda activate graspnet
    python grasp/tools/graspnet_server.py --checkpoint ~/work/graspnet-baseline/logs/checkpoint-rs.tar --port 5557

Wire: ZMQ REQ/REP, msgpack + msgpack-numpy.
  request  {"points": (N,3) float32 scene cloud in the camera optical frame [m], "num_point": 20000,
            "collision_thresh": 0.01, "voxel_size": 0.01, "seed": 0}
  reply    {"G": (M,4,4), "S": (M,), "W": (M,) width [m], "D": (M,) depth [m]} sorted by score, or {"error": str}

G is converted to the GraspGenX convention used by the rest of grasp/ (grasp/infer.py): +Z approach, +X closing,
origin = gripper base `--fingertip-depth` behind the fingertips. GraspNet (graspnetAPI) uses x = approach,
y = closing (width), and its fingers reach `depth` past the grasp point, so tips = t + depth * x_gn and
    R_G = [y_gn, z_gn, x_gn]  (columns; a cyclic permutation, det +1),   origin = tips - fingertip_depth * x_gn.
GraspNet-baseline is licensed for non-commercial research only (see ~/work/graspnet-baseline/LICENSE).
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

GN = os.path.expanduser(os.environ.get("GRASPNET_BASELINE", "~/work/graspnet-baseline"))
for sub in ("models", "dataset", "utils"):
    sys.path.append(os.path.join(GN, sub))


def to_graspgenx(arr: np.ndarray, fingertip_depth: float) -> np.ndarray:
    """graspnetAPI GraspGroup array rows [score, width, height, depth, R(9), t(3), obj_id] -> (M,4,4) GraspGenX poses."""
    R = arr[:, 4:13].reshape(-1, 3, 3)
    t, d = arr[:, 13:16], arr[:, 3]
    x, y, z = R[:, :, 0], R[:, :, 1], R[:, :, 2]
    G = np.tile(np.eye(4), (len(arr), 1, 1))
    G[:, :3, 0], G[:, :3, 1], G[:, :3, 2] = y, z, x
    G[:, :3, 3] = t + (d - fingertip_depth)[:, None] * x
    return G


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(GN, "logs", "checkpoint-rs.tar"))
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--fingertip-depth", type=float, default=0.14, help="flange -> jaw tip [m] (openarm_left_gripper.json)")
    a = ap.parse_args()
    import msgpack
    import msgpack_numpy as m
    import torch
    import zmq
    from graspnetAPI import GraspGroup
    from collision_detector import ModelFreeCollisionDetector
    from graspnet import GraspNet, pred_decode
    m.patch()

    dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4, cylinder_radius=0.05,
                   hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False).to(dev)
    ck = torch.load(a.checkpoint, map_location=dev)
    net.load_state_dict(ck["model_state_dict"]); net.eval()
    print(f"[graspnet] loaded {a.checkpoint} (epoch {ck['epoch']}) on {dev}; listening on :{a.port}", flush=True)

    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{a.port}")
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            t0 = time.time()
            pts = np.asarray(req["points"], np.float32)
            n = int(req.get("num_point", 20000))
            rng = np.random.default_rng(int(req.get("seed", 0)))
            idx = rng.choice(len(pts), n, replace=len(pts) < n)
            with torch.no_grad():
                ep = net({"point_clouds": torch.from_numpy(pts[idx][None]).to(dev)})
                arr = pred_decode(ep)[0].detach().cpu().numpy()
            gg = GraspGroup(arr)
            n_raw = len(gg)
            ct = float(req.get("collision_thresh", 0.01))
            if ct > 0:
                det = ModelFreeCollisionDetector(pts, voxel_size=float(req.get("voxel_size", 0.01)))
                gg = gg[~det.detect(gg, approach_dist=0.05, collision_thresh=ct)]
            gg = gg.nms().sort_by_score()
            arr = gg.grasp_group_array
            G = to_graspgenx(arr, a.fingertip_depth)
            rep = {"G": G.astype(np.float32), "S": arr[:, 0].astype(np.float32), "W": arr[:, 1].astype(np.float32),
                   "D": arr[:, 3].astype(np.float32)}
            print(f"[graspnet] {len(pts)} pts -> {n_raw} grasps, {len(arr)} after collision+nms "
                  f"({time.time() - t0:.2f}s), top score {arr[0, 0] if len(arr) else float('nan'):.3f}", flush=True)
        except Exception as e:  # never kill the server on one bad request
            rep = {"error": f"{type(e).__name__}: {e}"}
            print(f"[graspnet] error: {rep['error']}", flush=True)
        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    raise SystemExit(main())
