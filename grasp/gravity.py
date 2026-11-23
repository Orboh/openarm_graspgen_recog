"""腕を動かす script の重力のフィードフォワード（2026-11-23 Munenori・Kota の PC の Claude Code から）。

lerobot の OpenArmGravityModel に、手と爪の質量を入れた URDF（urdf/openarm_v10_bimanual_grav.urdf・
grasp/tools/make_grav_urdf.py が作る）を渡す。元の urdf/openarm_v10_bimanual.urdf は手が 0 kg で、
止めた姿勢の試し（grasp/tools/hold_test.py・2026-11-23 左腕）で TCP が 12〜18 mm 垂れていた。手 0.35 kg・爪 130 g を
足すと、重力による片寄りは 2〜6.5 mm に減った（J1〜J4 のモデルと実機の比 0.94〜1.05）。

GRASP_GRAV_URDF で別の URDF に替えられる（爪を替えたら make_grav_urdf.py の JAW_MASS を直して作り直す）。
"""
from __future__ import annotations

import os
from typing import Callable, Dict

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAV_URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual_grav.urdf")


def grav_urdf_path() -> str:
    """重力のモデルに使う URDF のパス（GRASP_GRAV_URDF があればそれ）。無ければ止める。"""
    path = os.path.expanduser(os.environ.get("GRASP_GRAV_URDF", GRAV_URDF))
    if not os.path.exists(path):
        raise SystemExit(f"[gravity] 重力のモデルの URDF が無い: {path}（python3 grasp/tools/make_grav_urdf.py で作る）")
    return path


def gravity_torque_fn(side: str, scale: float = 1.0) -> Callable[[Dict[str, float]], Dict[str, float]]:
    """{joint_1..7: deg} → {joint_k: Nm} の重力トルクの関数（soft_move_to_position などの torque_ff_fn に渡す）。"""
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    path = grav_urdf_path()
    print(f"[gravity] {side} の腕の重力のモデル: {os.path.basename(path)}", flush=True)
    return OpenArmGravityModel(path, side, scale).torque
