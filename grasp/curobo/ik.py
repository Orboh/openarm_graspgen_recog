"""cuRobo の IK（GPU で多くの初期値をまとめて解く・周りの物と自己干渉も見る）。cuRobo の venv でサーバーの中で動かす。

2026-11-22・作業者: Munenori。plan.py の Pinocchio の IK は初期値 7〜11 個で 1 つずつ解くので、解けるのに「解けない」と出る
候補が混ざり、計算もほとんどの時間を取っていた。掴み方の候補（掴む・手前・持ち上げの link7 の姿勢）をまとめて送って解く。

- 腕のモデルは grasp/curobo/model.py の robot_cfg_dict（左の URDF・link7 が道具の枠）。右腕は左のモデルを鏡に写して解く:
  目標の姿勢を M T M に、初期値の関節角を MIRROR_SIGN 倍にし、答えも MIRROR_SIGN 倍で右に戻す（planner.MirroredPlanner と同じ）
- 1 つの目標に return_seeds 個の解を返す（点数の順）。関節の範囲の絞り（GRASP_IK_LIMITS_RIGHT）と、手前・持ち上げの姿勢と
  同じ枝かは plan.py が選ぶ
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import numpy as np

from grasp.arm import mirror_q, mirror_T
from grasp.curobo.model import ARM_JOINTS, robot_cfg_dict
from grasp.curobo.planner import PLAN_DISABLE_LINKS, scene_dict

BATCH = int(os.environ.get("GRASP_CUROBO_IK_BATCH", "32"))   # 1 回にまとめて解く目標の数（GPU のメモリと相談）
SEEDS = int(os.environ.get("GRASP_CUROBO_IK_SEEDS", "32"))   # 目標 1 つあたりの初期値の数
# L-BFGS の反復回数（cuRobo の既定 80）。解けない目標が 1 つでも混ざると最初の LM の段で終われず、この回数を全部まわす
# （北九州の RTX 2000 Ada で 32 目標 × 32 初期値・80 回が 2.8 秒・2026-11-23 Munenori）。空なら既定
ITERS = os.environ.get("GRASP_CUROBO_IK_ITERS", "")


class GraspIK:
    """掴み方の link7 の姿勢をまとめて解く。作るのに時間がかかるので、サーバーで 1 度だけ作る。"""

    def __init__(self, world: Dict, side: str = "left"):
        import torch
        from curobo.inverse_kinematics import InverseKinematics, InverseKinematicsCfg
        from grasp.curobo.model import mirror_world
        self.torch, self.side = torch, side
        self._mw = mirror_world
        cfg = InverseKinematicsCfg.create(
            robot=robot_cfg_dict(side), scene_model=scene_dict(self._world(world)),
            collision_cache={"cuboid": 40, "mesh": 2}, self_collision_check=True, num_seeds=SEEDS,
            position_tolerance=0.002, orientation_tolerance=0.02, max_batch_size=BATCH, use_cuda_graph=True,
            **({"optimizer_configs": ["ik/particle_ik.yml", _lbfgs_yml(int(ITERS))]} if ITERS else {}),
        )
        self.ik = InverseKinematics(cfg)
        for ln in PLAN_DISABLE_LINKS:  # 本体にねじ止めした付け根は、いつも本体に触れている
            self.ik.kinematics.config.kinematics_config.disable_link_spheres(ln)
        self.joint_names = list(self.ik.joint_names)
        self.tool = self.ik.tool_frames[0]
        self.col = [self.joint_names.index(j) for j in ARM_JOINTS]
        self.solve(np.tile(np.eye(4), (2, 1, 1)) @ self._probe())  # 温める（CUDA graph を作る）

    def _world(self, world: Dict) -> Dict:
        return world if self.side == "left" else self._mw(world)

    @staticmethod
    def _probe() -> np.ndarray:
        T = np.eye(4); T[:3, :3] = np.diag([1.0, -1.0, -1.0]); T[:3, 3] = [0.3, 0.2, 0.45]
        return T

    def update_world(self, world: Dict) -> None:
        from curobo.scene import Scene
        self.ik.update_world(Scene.create(scene_dict(self._world(world))))

    def solve(self, T_link7: np.ndarray, seed_deg: Optional[np.ndarray] = None,
              return_seeds: int = 4, lm_only: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """link7 の姿勢 (N, 4, 4)（その腕の付け根の座標）→ 関節角 (N, k, 7) [deg・その腕の lerobot の度] と成功 (N, k)。
        seed_deg (N, 7) を渡すと、その目標の初期値の 1 つにする（手前・持ち上げを掴む姿勢と同じ枝で解くため）。"""
        from curobo.types import GoalToolPose, Pose
        torch = self.torch
        T = np.asarray(T_link7, float).reshape(-1, 4, 4)
        if self.side == "right":
            T = mirror_T(T)
            seed_deg = None if seed_deg is None else mirror_q(np.asarray(seed_deg, float))
        N, k = len(T), max(1, int(return_seeds))
        Q = np.zeros((N, k, 7)); OK = np.zeros((N, k), bool)
        for s in range(0, N, BATCH):
            Tb = T[s:s + BATCH]; n = len(Tb)
            Tp = np.concatenate([Tb, np.repeat(Tb[:1], BATCH - n, 0)]) if n < BATCH else Tb  # 決まった大きさに埋める（CUDA graph）
            pos = torch.tensor(Tp[:, :3, 3], device="cuda:0", dtype=torch.float32)
            quat = torch.tensor(_quat_wxyz(Tp[:, :3, :3]), device="cuda:0", dtype=torch.float32)
            goal = GoalToolPose.from_poses({self.tool: Pose(position=pos, quaternion=quat)}, num_goalset=1)
            seed = None
            if seed_deg is not None:
                sd = np.asarray(seed_deg, float)[s:s + BATCH]
                sd = np.concatenate([sd, np.repeat(sd[:1], BATCH - n, 0)]) if n < BATCH else sd
                full = torch.zeros((BATCH, 1, len(self.joint_names)), device="cuda:0", dtype=torch.float32)
                for c, j in enumerate(self.col):
                    full[:, 0, j] = torch.tensor(np.deg2rad(sd[:, c]), device="cuda:0", dtype=torch.float32)
                seed = full
            # lm_only: 最初の LM の段（初期値の解き）だけで終える。解けない目標が混ざっても L-BFGS をまわさない（2026-11-23）。
            # cuRobo は「解けた割合 ≥ exit_early_batch_success_threshold（既定 1.0）」なら LM の段で返すので、その間だけ 0 にする
            # （run_optimizer=False は LM の段も飛ばす）
            cfg = self.ik.config
            keep = cfg.exit_early_batch_success_threshold
            cfg.exit_early_batch_success_threshold = 0.0 if lm_only else keep
            try:
                r = self.ik.solve_pose(goal, seed_config=seed, return_seeds=k)
            finally:
                cfg.exit_early_batch_success_threshold = keep
            P = r.js_solution.position.reshape(BATCH, k, -1).detach().cpu().numpy()[:n][:, :, self.col]
            Q[s:s + n] = np.rad2deg(P)
            OK[s:s + n] = r.success.reshape(BATCH, k).detach().cpu().numpy()[:n]
        if self.side == "right":
            Q = mirror_q(Q)
        return Q, OK


def _lbfgs_yml(iters: int) -> str:
    """cuRobo の ik/lbfgs_ik.yml の写しで、反復回数（num_iters・inner_iters。両方そろえないと cuRobo が止める）だけ替えた物の絶対パス。"""
    import tempfile

    import yaml
    from curobo._src.util.config_io import join_path, load_yaml
    from curobo._src.util_file import get_task_configs_path
    d = load_yaml(join_path(get_task_configs_path(), "ik/lbfgs_ik.yml"))
    d["optimizer"]["num_iters"] = d["optimizer"]["inner_iters"] = int(iters)
    f = os.path.join(tempfile.gettempdir(), f"openarm_grasp_lbfgs_ik_{iters}.yml")
    with open(f, "w") as fh:
        yaml.safe_dump(d, fh)
    return f


def _quat_wxyz(R: np.ndarray) -> np.ndarray:
    """回転行列 (N, 3, 3) → 四元数 (N, 4) wxyz。"""
    from scipy.spatial.transform import Rotation
    q = Rotation.from_matrix(R).as_quat()  # xyzw
    return np.concatenate([q[:, 3:4], q[:, :3]], axis=1)
