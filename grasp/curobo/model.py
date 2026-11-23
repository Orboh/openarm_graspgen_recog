"""cuRobo の左腕のモデルと、置き場ごとの周りの物（箱）を読む。cuRobo の venv（~/work/curobo/.venv）で動かす。

- ロボット: grasp/curobo/openarm_left_umi.yml（make_urdf.py の URDF に cuRobo の RobotBuilder が球を当てはめた物）。
  読むときに URDF とメッシュの場所をこのフォルダに向け直し、指の関節を開いた形で止める
- 周りの物: grasp/config/sites/<置き場>.world.yml（GRASP_WORLD_YML）。座標は腕の付け根（URDF の world・plan.py の base と同じ）[m]
  箱（cuboid）とメッシュ（純正の本体など）。机の板は GRASP_FLOOR_Z を上の面にした大きな箱で入れる
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
ROBOT_YML = os.path.join(HERE, "openarm_left_umi.yml")
ARM_JOINTS = [f"openarm_left_joint{i}" for i in range(1, 8)]
FINGER_JOINT = "openarm_left_umi_finger_l_joint"
# 指の関節（片側の開き [m]）。0.055 で指先の間 110 mm（make_umi_urdf.py）。北九州の全開は 80 mm なので 0.04
FINGER_OPEN_M = float(os.environ.get("GRASP_CUROBO_FINGER_OPEN_M", "0.04"))
TABLE_THICK = 0.10


def robot_cfg_dict() -> Dict:
    """cuRobo の RobotCfg.create に渡す辞書。"""
    d = yaml.safe_load(open(ROBOT_YML))
    d = d.get("robot_cfg", d)
    k = d["kinematics"]
    k["urdf_path"] = os.path.join(HERE, "openarm_left_umi.urdf")
    k["asset_root_path"] = HERE
    k["lock_joints"] = {FINGER_JOINT: FINGER_OPEN_M}
    # 手首をはさむ link5（前腕の先）と link7（手）は、球のはみ出し（約 5 mm）で手首を曲げると重なって見える。
    # 2026-11-21 の実機の軌跡 120 姿勢（関節 6 は −44〜+39°）でも最大 10 mm 重なり、実物は当たっていないので自己干渉から外す
    ign = k.setdefault("self_collision_ignore", {})
    ign.setdefault("openarm_left_link5", [])
    if "openarm_left_link7" not in ign["openarm_left_link5"]:
        ign["openarm_left_link5"].append("openarm_left_link7")
    return {"robot_cfg": d}


def load_world(path: Optional[str] = None, floor_z: Optional[float] = None) -> Dict[str, List[Dict]]:
    """置き場の周りの物を読む。返り値は {"cuboid": [{"name", "dims", "pose"(xyz+wxyz)}...],
    "mesh": [{"name", "file"(絶対パス), "scale", "pose"}...]}。

    world.yml の形:
        cuboid:
          <名前>: {dims: [x, y, z], pose: [x, y, z, qw, qx, qy, qz]}
        mesh:
          <名前>: {file: <repo からの相対パス>, scale: 0.001, pose: [x, y, z, qw, qx, qy, qz]}
        table: true      # GRASP_FLOOR_Z を上の面にした机の板を足す（既定 true）
    """
    path = path or os.environ.get("GRASP_WORLD_YML")
    w = yaml.safe_load(open(path)) if path else {}
    w = w or {}
    cubs = [dict(name=n, dims=[float(v) for v in c["dims"]], pose=[float(v) for v in c["pose"]])
            for n, c in (w.get("cuboid") or {}).items()]
    meshes = [dict(name=n, file=os.path.join(REPO, c["file"]), scale=float(c.get("scale", 1.0)),
                   pose=[float(v) for v in c.get("pose", [0, 0, 0, 1, 0, 0, 0])], ignore_links=list(c.get("ignore_links", [])))
              for n, c in (w.get("mesh") or {}).items()]
    if w.get("table", True):
        fz = floor_z if floor_z is not None else os.environ.get("GRASP_FLOOR_Z")
        if fz is None:
            raise SystemExit("[curobo] GRASP_FLOOR_Z が無い。机の板の高さを決めずに当たりを調べない")
        fz = float(fz)
        cubs.append(dict(name="table", dims=[3.0, 3.0, TABLE_THICK], pose=[0.0, 0.0, fz - TABLE_THICK / 2, 1, 0, 0, 0]))
    return {"cuboid": cubs, "mesh": meshes}


def _quat_wxyz_to_R(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def sphere_cuboid_distance(spheres: np.ndarray, cuboid: Dict) -> np.ndarray:
    """球（(..., 4) = x, y, z, r）の表面から箱の表面までの符号付きの距離 [m]。負はめり込み。"""
    c = np.asarray(cuboid["pose"][:3], float)
    R = _quat_wxyz_to_R(cuboid["pose"][3:7])
    h = np.asarray(cuboid["dims"], float) / 2
    p = (spheres[..., :3] - c) @ R  # 箱の座標へ
    q = np.abs(p) - h
    outside = np.linalg.norm(np.maximum(q, 0.0), axis=-1)
    inside = np.minimum(np.max(q, axis=-1), 0.0)
    return outside + inside - spheres[..., 3]


class MeshDistance:
    """動かない物のメッシュ（本体など）と球の符号付きの距離 [m]（Warp の GPU のメッシュの問い合わせ。負はめり込み）。"""

    def __init__(self, mesh: Dict):
        import trimesh
        import warp as wp
        self.wp = wp
        m = trimesh.load(mesh["file"], force="mesh")
        m.apply_scale(mesh["scale"])
        T = np.eye(4); T[:3, :3] = _quat_wxyz_to_R(mesh["pose"][3:7]); T[:3, 3] = mesh["pose"][:3]
        m.apply_transform(T)
        if not m.is_watertight:
            raise SystemExit(f"[curobo] {mesh['file']} が閉じていない（内と外が決まらない）")
        self.name = mesh["name"]
        self.ignore_links = set(mesh.get("ignore_links", []))  # この物に触れていてよい link（本体にねじ止めした付け根）
        self.mesh = wp.Mesh(points=wp.array(m.vertices.astype(np.float32), dtype=wp.vec3, device="cuda:0"),
                            indices=wp.array(m.faces.astype(np.int32).ravel(), dtype=int, device="cuda:0"))
        self._kernel = _mesh_sd_kernel()

    def distance(self, spheres: np.ndarray) -> np.ndarray:
        """球（(..., 4)）の表面からメッシュの表面までの距離。"""
        wp = self.wp
        P = spheres[..., :3].reshape(-1, 3).astype(np.float32)
        pts = wp.array(P, dtype=wp.vec3, device="cuda:0")
        out = wp.zeros(len(P), dtype=float, device="cuda:0")
        wp.launch(self._kernel, dim=len(P), inputs=[self.mesh.id, pts, 10.0, out], device="cuda:0")
        return out.numpy().reshape(spheres.shape[:-1]) - spheres[..., 3]


_KERNEL = None


def _mesh_sd_kernel():
    """点からメッシュの表面までの符号付きの距離（外が正）の Warp の kernel を 1 度だけ作る。"""
    global _KERNEL
    if _KERNEL is None:
        import warp as wp

        @wp.kernel
        def mesh_sd(mesh: wp.uint64, pts: wp.array(dtype=wp.vec3), max_dist: float, out: wp.array(dtype=float)):
            i = wp.tid()
            q = wp.mesh_query_point_sign_winding_number(mesh, pts[i], max_dist)
            if q.result:
                c = wp.mesh_eval_position(mesh, q.face, q.u, q.v)
                out[i] = wp.length(pts[i] - c) * q.sign
            else:
                out[i] = max_dist
        _KERNEL = mesh_sd
    return _KERNEL


def obstacle_distances(S: np.ndarray, world: Dict, meshes: Optional[List[MeshDistance]] = None,
                       sphere_links: Optional[List[str]] = None) -> Dict[str, np.ndarray]:
    """周りの物ごとに、球（(..., S, 4)）の表面からの距離（(..., S)）。メッシュの ignore_links の球は +inf。"""
    out = {c["name"]: sphere_cuboid_distance(S, c) for c in world["cuboid"]}
    for md in meshes or []:
        D = md.distance(S)
        if md.ignore_links and sphere_links:
            D[..., np.array([l in md.ignore_links for l in sphere_links])] = np.inf
        out[md.name] = D
    return out


class ArmSpheres:
    """cuRobo の FK で、関節角（rad・関節 1〜7）から当たり判定の球を出す。"""

    def __init__(self):
        import torch
        from curobo.kinematics import Kinematics, KinematicsCfg
        self.torch = torch
        self.kin = Kinematics(KinematicsCfg.from_data_dict(robot_cfg_dict()))
        self.joint_names = list(self.kin.joint_names)
        idx = [self.joint_names.index(j) for j in ARM_JOINTS]
        self._idx = idx
        self.sphere_links = self._sphere_link_names()

    def _sphere_link_names(self) -> List[str]:
        """球ごとの link の名前（球の並びの順）。"""
        kc = self.kin.kinematics_config
        names = [""] * int(kc.total_spheres)
        for link in robot_cfg_dict()["robot_cfg"]["kinematics"]["collision_link_names"]:
            for i in np.atleast_1d(kc.get_sphere_index_from_link_name(link).cpu().numpy()):
                names[int(i)] = link
        return names

    def spheres(self, q_rad: np.ndarray) -> np.ndarray:
        """q_rad: (N, 7) → 球 (N, S, 4)。半径が負の球（使っていない枠）はそのまま返す。"""
        torch = self.torch
        q = np.atleast_2d(np.asarray(q_rad, float))
        full = np.zeros((len(q), len(self.joint_names)))
        full[:, self._idx] = q
        from curobo.types import JointState
        js = JointState.from_position(torch.as_tensor(full, dtype=torch.float32, device="cuda:0"), joint_names=self.joint_names)
        st = self.kin.compute_kinematics(js)
        s = st.robot_spheres
        return s.reshape(len(q), -1, 4).detach().cpu().numpy()

    def link7_pose(self, q_rad: np.ndarray) -> np.ndarray:
        """FK の確かめ用: link7 の 4x4（N, 4, 4）。"""
        torch = self.torch
        from curobo.types import JointState
        q = np.atleast_2d(np.asarray(q_rad, float))
        full = np.zeros((len(q), len(self.joint_names)))
        full[:, self._idx] = q
        js = JointState.from_position(torch.as_tensor(full, dtype=torch.float32, device="cuda:0"), joint_names=self.joint_names)
        st = self.kin.compute_kinematics(js)
        tp = st.tool_poses
        p = tp.position.reshape(len(q), -1, 3)[:, 0].cpu().numpy()
        qq = tp.quaternion.reshape(len(q), -1, 4)[:, 0].cpu().numpy()
        T = np.tile(np.eye(4), (len(q), 1, 1))
        for i in range(len(q)):
            T[i, :3, :3] = _quat_wxyz_to_R(qq[i]); T[i, :3, 3] = p[i]
        return T
