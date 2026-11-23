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
    if len(P) == 0:
        return np.zeros((0, 3))
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P.astype(float))).voxel_down_sample(0.01)
    return np.asarray(pc.points)


def capture_arm_qs(extra: List[List[float]] = (), side: str = "left") -> List[List[float]]:
    """撮影したときにその腕（side）がいそうな姿勢（lerobot の度）: 休みの姿勢（左 GRASP_REST・右 GRASP_REST_RIGHT）と、
    その腕で動かした記録のうち、いちばん新しい「休みの姿勢に戻り終えた」traj.csv の最後の姿勢。

    最後の行が back to start・back to rest の記録だけを使う。途中で止めた回の最後の行（例: 置き先の上）を使うと、実際には
    腕がいない所の点を腕として消し、そこにある物を障害物に入れない（2026-11-21 19:00: 非常停止の回の後、段ボール箱が消えた）。
    どちらの腕の記録かは同じフォルダの plan.json の "arm"（無ければ左・2026-11-22 より前の記録は全部左）。"""
    qs = [list(q) for q in extra]
    rest = os.environ.get("GRASP_REST_RIGHT" if side == "right" else "GRASP_REST")
    if rest:
        qs.append([float(v) for v in rest.split(",")])
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for t in sorted(glob.glob(os.path.join(here, "runs", "*", "traj.csv")), key=os.path.getmtime, reverse=True):
        pj = os.path.join(os.path.dirname(t), "plan.json")
        try:
            arm = json.load(open(pj)).get("arm", "left") if os.path.exists(pj) else "left"
        except (OSError, ValueError):
            continue
        if arm != side:
            continue
        rows = list(csv.DictReader(open(t)))
        if rows and rows[-1]["phase"].startswith(("back to start", "back to rest")):
            qs.append([float(rows[-1][f"joint_{i}.pos"]) for i in range(1, 8)])
            break
    return qs
