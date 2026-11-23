"""撮った深度から、cuRobo に入れる障害物の候補の点を選ぶ（GraspGenX・lerobot の venv で動く・numpy と scipy と open3d）。

選ぶ点: 腕の付け根の座標で、机の面より 1.5 cm〜60 cm 上・腕の届く辺り・掴む物の点から 3 cm より遠い。1 cm の格子に間引く。
腕自身の点はここでは消さない（撮影時の腕の角度でサーバーが消す: client.set_obstacle_points）。
"""
from __future__ import annotations

import csv
import glob
import json
import os
from typing import List

import numpy as np

H_MIN, H_MAX = 0.015, 0.60
TARGET_CLEAR = 0.03
X_RANGE, Y_RANGE = (-0.25, 0.90), (-0.60, 0.70)
POINTS_FILE = "obstacle_points.npy"
BOXES_FILE = "obstacles.json"


def seen_points(cap_dir: str, T_bc: np.ndarray, obj_cam: np.ndarray, n_b: np.ndarray, d_b: float) -> np.ndarray:
    """障害物の候補の点 (N, 3)（腕の付け根の座標）。n_b・d_b は机の面（高さ = n·p + d）。"""
    import open3d as o3d
    from scipy.spatial import cKDTree
    from grasp.capture import depth_to_points
    K = np.array(json.load(open(os.path.join(cap_dir, "intrinsics.json")))["K"])
    pts, _ = depth_to_points(np.load(os.path.join(cap_dir, "depth.npy")), K, zmax=1.2)
    P = pts @ T_bc[:3, :3].T + T_bc[:3, 3]
    h = P @ n_b + d_b
    P = P[(h > H_MIN) & (h < H_MAX) & (P[:, 0] > X_RANGE[0]) & (P[:, 0] < X_RANGE[1])
          & (P[:, 1] > Y_RANGE[0]) & (P[:, 1] < Y_RANGE[1])]
    if len(P) and len(obj_cam):
        ob = obj_cam @ T_bc[:3, :3].T + T_bc[:3, 3]
        d, _ = cKDTree(ob).query(P, k=1)
        P = P[d > TARGET_CLEAR]
    P = drop_known(P)
    if len(P) == 0:
        return np.zeros((0, 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(float))).voxel_down_sample(0.01)
    return np.asarray(pc.points)


KNOWN_PAD = 0.03


def drop_known(P: np.ndarray) -> np.ndarray:
    """2026-11-22 16:40: since the chest camera looks almost straight down it sees the cart tray, the body and the
    other arm parked at home. Those became 'seen' obstacle boxes around the left arm's own home pose, so cuRobo refused
    every move out of home. They are already modelled (world.yml cuboid / cuboid_extra / mesh, checked by
    grasp/arm_collision.py): drop the points within KNOWN_PAD of them."""
    w_path = os.environ.get("GRASP_WORLD_YML")
    if not w_path or not os.path.exists(w_path) or len(P) == 0:
        return P
    import yaml
    from scipy.spatial.transform import Rotation
    w = yaml.safe_load(open(w_path)) or {}
    keep = np.ones(len(P), bool)
    cubs = dict(w.get("cuboid") or {}); cubs.update(w.get("cuboid_extra") or {})
    for name, c in cubs.items():
        dims, pose = np.array(c["dims"], float), np.array(c["pose"], float)
        R = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
        loc = (P - pose[:3]) @ R
        keep &= ~np.all(np.abs(loc) <= dims / 2 + KNOWN_PAD, axis=1)
    here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for name, c in (w.get("mesh") or {}).items():
        f = os.path.join(here, c["file"])
        if not os.path.exists(f):
            continue
        from scipy.spatial import cKDTree
        from grasp.arm_collision import read_stl, surface_points
        V = surface_points(read_stl(f) * float(c.get("scale", 1.0)))
        pose = np.array(c.get("pose", [0, 0, 0, 1, 0, 0, 0]), float)
        R = Rotation.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
        d, _ = cKDTree(V @ R.T + pose[:3]).query(P, k=1)
        keep &= d > KNOWN_PAD
    return P[keep]


def capture_arm_qs(extra: List[List[float]] = ()) -> List[List[float]]:
    """撮影したときに腕がいそうな姿勢（lerobot の度）: GRASP_REST と、いちばん新しい「休みの姿勢に戻り終えた」traj.csv の最後の姿勢。

    最後の行が back to start・back to rest の記録だけを使う。途中で止めた回の最後の行（例: 置き先の上）を使うと、実際には
    腕がいない所の点を腕として消し、そこにある物を障害物に入れない（2026-11-21 19:00: 非常停止の回の後、段ボール箱が消えた）。"""
    qs = [list(q) for q in extra]
    # 2026-11-22: the arm now waits at home v5 between trials (trial_loop / --hold-home): its own points there are the arm
    import yaml
    hy = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config",
                      f"home_{os.environ.get('GRASP_ARM', 'left')}_cart-sym-v5.yaml")
    if os.path.exists(hy):
        h = yaml.safe_load(open(hy)); qs.append([float(h[f"joint_{i}"]) for i in range(1, 8)])
    if os.environ.get("GRASP_REST"):
        qs.append([float(v) for v in os.environ["GRASP_REST"].split(",")])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for t in sorted(glob.glob(os.path.join(here, "runs", "*", "traj.csv")), key=os.path.getmtime, reverse=True):
        rows = list(csv.DictReader(open(t)))
        if rows and rows[-1]["phase"].startswith(("back to start", "back to rest")):
            qs.append([float(rows[-1][f"joint_{i}.pos"]) for i in range(1, 8)])
            break
    return qs
