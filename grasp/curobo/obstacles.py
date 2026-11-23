"""胸のカメラの深度の点から、cuRobo に入れる障害物の箱を作る（2026-11-21 Munenori の依頼「経路に障害物がある状態で試す」）。

numpy と scipy だけを使う。cuRobo の venv（サーバー）で腕の点を消した後に呼ぶ。
- 点（腕の付け根の座標）を 1 cm の格子に入れ、つながった塊ごとに軸に沿った箱にする（点の少ない塊はノイズとして捨てる）
- 箱の下端は机の板まで伸ばす（机に置いた物の下は見えなくても埋まっている）
- カメラから見えない奥の面の分、カメラと反対の向きに BACK だけ伸ばす（段ボール箱などは手前の面と上の面しか写らない）
- 周りに PAD だけ広げる
"""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

VOXEL = 0.01
MIN_VOXELS = 15      # これより少ない塊はノイズとして捨てる（1 cm 角の格子でおよそ 15 cm²）
PAD = 0.01           # 箱の周りに広げる [m]
BACK = 0.05          # カメラから見えない奥の分 [m]
MAX_BOXES = 16       # サーバーの cuRobo の箱の枠（planner の collision_cache）に収まる数


def points_to_boxes(P: np.ndarray, cam_pos: Sequence[float], floor_z: float, pad: float = PAD, back: float = BACK,
                    min_voxels: int = MIN_VOXELS, max_boxes: int = MAX_BOXES, voxel: float = VOXEL,
                    to_floor: bool = True, oriented: bool = True) -> List[Dict]:
    """点 (N, 3) → 箱の一覧 [{"name", "dims", "pose"(xyz + wxyz)}]。大きい塊から max_boxes 個まで。

    oriented: 真上から見た点の広がりの向き（主成分）に合わせて、箱を鉛直の軸のまわりに回す。軸に沿った箱だと、斜めに置いた
    細長い物（段ボール箱）を空いた角ごと囲み、隣の物（掴む物）まで障害物が広がる（2026-11-21 19:3x）。"""
    from scipy import ndimage
    P = np.asarray(P, float).reshape(-1, 3)
    if len(P) == 0:
        return []
    lo = P.min(0)
    idx = np.floor((P - lo) / voxel).astype(int)
    grid = np.zeros(idx.max(0) + 1, bool)
    grid[tuple(idx.T)] = True
    lab, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    cams = np.asarray(cam_pos, float)
    boxes = []
    for k in range(1, n + 1):
        vox = np.argwhere(lab == k)
        if len(vox) < min_voxels:
            continue
        pts = lo + (vox + 0.5) * voxel
        yaw = 0.0
        if oriented and len(pts) >= 3:
            xy = pts[:, :2] - pts[:, :2].mean(0)
            ev, evec = np.linalg.eigh(xy.T @ xy)
            yaw = float(np.arctan2(evec[1, 1], evec[0, 1]))  # 一番広がる向き
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
        loc = pts @ Rz  # 箱の向きの座標
        bmin, bmax = loc.min(0) - voxel / 2 - pad, loc.max(0) + voxel / 2 + pad
        if to_floor:  # 机まで（机に置いた物。宙に浮いた手は伸ばさない）
            bmin[2] = min(bmin[2], floor_z)
        cl = (bmin + bmax) / 2
        cam_l = (cams - Rz @ cl) @ Rz  # 箱の中心から見たカメラ（箱の向きの座標）
        for ax in (0, 1):  # カメラと反対の向きに伸ばす（横の 2 軸）
            if cam_l[ax] <= 0:
                bmax[ax] += back
            else:
                bmin[ax] -= back
        boxes.append((len(vox), bmin, bmax, Rz, yaw))
    boxes.sort(key=lambda b: -b[0])
    out = []
    for i, (_, bmin, bmax, Rz, yaw) in enumerate(boxes[:max_boxes]):
        c = Rz @ ((bmin + bmax) / 2)
        q = [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]  # wxyz（鉛直の軸のまわり）
        out.append({"name": f"seen_{i}", "dims": (bmax - bmin).round(4).tolist(), "pose": c.round(4).tolist() + [round(v, 5) for v in q]})
    return out
