"""cuRobo の MotionPlanner で、置き場の周りの物（本体・カメラ・机）をよけて回り込む関節の経路を作る。cuRobo の venv で動かす。

cuRobo の経路（細かい点の列）は、そのままだと run_grasp.py が点ごとに止まって遅いので、こちらの当たり判定（server.check_path と
client.violations_from。cuRobo とは別の計算）で「直線で動かしても当たらない」所までまとめて、少ない経由点にして返す。
返す経由点は全部、こちらの判定で確かめてある。cuRobo が当たらないと言っても、こちらの判定で当たる経路は返さない。
"""
from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional

import numpy as np

from grasp.curobo.model import ARM_JOINTS, robot_cfg_dict

# 経路計画の最適化が、周りの物にこの距離より近づくと離れる向きに押す [m]。こちらの判定の余裕（client.MARGIN 10 mm）より大きくして、
# まとめた直線でも余裕が残るようにする。腕の球そのものは広げない: 広げると、物の近くの端の姿勢（待機の姿勢など）が
# cuRobo には当たって見えて計画が始まらない（2026-11-21 の試しで 15 mm 広げると 10 組中 5 組が端で弾かれた）
PLAN_ACTIVATION = float(os.environ.get("GRASP_CUROBO_PLAN_ACTIVATION", "0.025"))
# 本体にねじ止めしてあり、いつも本体に触れている link は経路計画の当たり判定から外す
PLAN_DISABLE_LINKS = ["openarm_left_link0"]


def scene_dict(world: Dict) -> Dict:
    """load_world の結果を cuRobo の scene の辞書に。メッシュは頂点と面を渡す（倍率と位置を当てた後）。"""
    import trimesh
    from grasp.curobo.model import _quat_wxyz_to_R
    sc = {"cuboid": {c["name"]: {"dims": c["dims"], "pose": c["pose"]} for c in world["cuboid"]}, "mesh": {}}
    for m in world["mesh"]:
        tm = trimesh.load(m["file"], force="mesh")
        tm.apply_scale(m["scale"])
        T = np.eye(4); T[:3, :3] = _quat_wxyz_to_R(m["pose"][3:7]); T[:3, 3] = m["pose"][:3]
        tm.apply_transform(T)
        sc["mesh"][m["name"]] = {"vertices": tm.vertices.tolist(), "faces": tm.faces.ravel().tolist(),
                                 "pose": [0, 0, 0, 1, 0, 0, 0]}
    return sc


class DetourPlanner:
    """回り込む経路を作る。作るのと温めるのに時間がかかるので、サーバーで 1 度だけ作る。"""

    def __init__(self, world: Dict):
        import torch
        from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
        self.torch = torch
        cfg = MotionPlannerCfg.create(
            robot=robot_cfg_dict(), scene_model=scene_dict(world), collision_cache={"cuboid": 24, "mesh": 2},  # 置き場の箱 + カメラで見た障害物（obstacles.MAX_BOXES）
            use_cuda_graph=True, graph_planner_algorithm="rrt", optimizer_collision_activation_distance=PLAN_ACTIVATION,
        )
        self.mp = MotionPlanner(cfg)
        self.mp.disable_link_collision(PLAN_DISABLE_LINKS)
        self.mp.warmup(enable_graph=True, num_warmup_iterations=2)
        self.joint_names = list(self.mp.joint_names)

    def update_world(self, world: Dict) -> None:
        """周りの物を入れ替える（カメラで見た障害物が変わったとき）。"""
        from curobo.scene import Scene
        self.mp.update_world(Scene.create(scene_dict(world)))

    def dense(self, q0_deg, q1_deg, max_attempts: int = 4) -> Optional[np.ndarray]:
        """cuRobo の経路（(M, 7) [deg]・関節 1〜7）。作れなければ None。"""
        from curobo.types import JointState
        torch = self.torch

        def js(qd):
            full = torch.zeros((1, len(self.joint_names)), device="cuda:0", dtype=torch.float32)
            for k, j in enumerate(ARM_JOINTS):
                full[0, self.joint_names.index(j)] = float(np.deg2rad(qd[k]))
            return JointState.from_position(full, joint_names=self.joint_names)

        res = self.mp.plan_cspace(js(q1_deg), js(q0_deg), max_attempts=max_attempts, enable_graph_attempt=0)
        if res is None or not bool(res.success.any()):
            return None
        tr = res.get_interpolated_plan()
        names = list(tr.joint_names)
        P = tr.position.reshape(-1, len(names)).detach().cpu().numpy()
        return np.rad2deg(P[:, [names.index(j) for j in ARM_JOINTS]])


def shortcut(path_deg: np.ndarray, seg_ok: Callable[[np.ndarray, np.ndarray], bool]) -> Optional[List[List[float]]]:
    """細かい経路を、直線で当たらない限り先まで飛ばして、少ない経由点（始点を除き終点を含む）にする。

    seg_ok(qa, qb) はこちらの当たり判定。隣り合う点どうしでも通らなければ None（その経路は使わない）。
    """
    P = np.asarray(path_deg, float)
    vias, i = [], 0
    while i < len(P) - 1:
        j = len(P) - 1
        while j > i + 1 and not seg_ok(P[i], P[j]):
            j = i + 1 + (j - i - 1) // 2 if j - i > 8 else j - 1  # 遠い所は半分ずつ、近くは 1 点ずつ戻る
        if not seg_ok(P[i], P[j]):
            return None
        vias.append(P[j].tolist())
        i = j
    return vias
