"""どちらの腕で掴むか（左・右）と、腕ごとの設定の読み方。numpy だけで動く（どの venv からも読む）。

2026-11-22・作業者: Munenori。右腕（右リーダー＋左と同じグリッパー）を足すときに作った。

- 今の腕は環境変数 GRASP_ARM（left・right・既定 left）。plan.py --arm both は腕ごとに子のプロセスで GRASP_ARM を変えて計算する
- 腕ごとの値は、左は今までの名前（GRASP_CAN_PORT など）、右は名前の後ろに _RIGHT を付けた名前（GRASP_CAN_PORT_RIGHT）
- 左右の腕の関節と link の枠は、URDF（openarm_v10_bimanual）の上で付け根の座標の y を反転した鏡写し。関節の角度を
  q_右 = MIRROR_SIGN * q_左 にすると、link の枠が T_右 = M T_左 M（M = diag(1, -1, 1)）になる。link4 だけは右の枠が
  肘の軸（自分の y）に沿って 63 mm ずれた所にある（動きは同じ）。指先（hand_tcp）も鏡写し（2026-11-22 に確かめた・tests/test_arm.py）
- ただし URDF の右腕の各 link の形は左の形の鏡写しではなく、同じ部品を右の枠に置いた物。cuRobo の右腕は球を link ごとに
  置き直してから左の運動学で動かし、鏡に写す（grasp/curobo/model.py の right_local_center・MirroredArm）
"""
from __future__ import annotations

import os
from typing import Optional

import numpy as np

ARMS = ("left", "right")
# q_右 = MIRROR_SIGN * q_左（関節 1〜7）。lerobot の度も URDF のラジアンも同じ（どちらも関節の番号どおりに写しているので）
MIRROR_SIGN = np.array([-1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0])
M = np.diag([1.0, -1.0, 1.0])


def current() -> str:
    """今の腕（GRASP_ARM）。知らない値なら止める。"""
    a = os.environ.get("GRASP_ARM", "left")
    if a not in ARMS:
        raise SystemExit(f"[arm] GRASP_ARM={a!r} は left か right")
    return a


def env(name: str, side: Optional[str] = None, default: Optional[str] = None) -> Optional[str]:
    """腕ごとの環境変数。右は name + '_RIGHT'。右の値が無いとき左の値には落とさない（違う腕の CAN や姿勢で動かさない）。"""
    side = side or current()
    return os.environ.get(name + "_RIGHT" if side == "right" else name, default)


def prefix(side: Optional[str] = None) -> str:
    """URDF の名前の頭（openarm_left_ / openarm_right_）。"""
    return f"openarm_{side or current()}_"


def mirror_q(q) -> np.ndarray:
    """左の関節角を、鏡写しの右の関節角に（逆も同じ式）。(..., 7)。"""
    return np.asarray(q, float) * MIRROR_SIGN


def mirror_points(P) -> np.ndarray:
    """付け根の座標の点を y で反転（(..., 3) か、半径の付いた球 (..., 4)）。"""
    P = np.array(P, float, copy=True)
    P[..., 1] *= -1.0
    return P


def mirror_T(T) -> np.ndarray:
    """4x4 の姿勢（(..., 4, 4)）の鏡写し: M T M。向きは右手系のまま。"""
    T = np.asarray(T, float)
    S = np.diag([1.0, -1.0, 1.0, 1.0])
    return S @ T @ S


def mirror_pose_wxyz(pose) -> list:
    """[x, y, z, qw, qx, qy, qz] の鏡写し（箱の位置と向き）。回転 R → M R M は四元数で (w, -x, y, -z)。"""
    x, y, z, w, qx, qy, qz = (float(v) for v in pose)
    return [x, -y, z, w, -qx, qy, -qz]


def default_rest(side: str) -> Optional[list]:
    """休みの姿勢（lerobot の度）。右は GRASP_REST_RIGHT だけ（左の値を鏡に写して勝手に使わない。実機で確かめて置き場の env に書く）。"""
    v = env("GRASP_REST", side)
    return [float(x) for x in v.split(",")] if v else None
