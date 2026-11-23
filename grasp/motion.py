"""関節を滑らかに動かす指令（lerobot の venv）。lerobot の soft_move_to_position と同じ送り方で、速さの付け方だけを変える。

2026-11-22・作業者: Munenori。soft_move_to_position は指令の位置を一定の速さで進めるので、出だしでいきなり最高の速さになり、
着くといきなり止まる。回り込みの経由点ごとにこれをくり返すと、腕がかくかく動いた（北九州の右腕・75 deg/s）。

- 位置: 最小躍度の曲線 s(t) = 10t^3 - 15t^4 + 6t^5（出だしと終わりの速さと加速度が 0）
- 速さの上限: 曲線の一番速い所は平均の 1.875 倍なので、一番大きく動く関節の一番速い所が max_speed_dps になる長さにする
  （soft_move_to_position の一定の速さより 1.875 倍長くかかる）
- 速さの目標（MIT 指令の速度の項）も曲線の微分で送る。soft_move_to_position は 0 を送るので、kd が動きを止める向きに効いて遅れていた
- ゲイン・重力の補正・10 ms ごとの送り方は soft_move_to_position と同じ（OPENARM_ALIGN_KP/KD・torque_ff_fn）
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

PEAK = 1.875  # 最小躍度の曲線の一番速い所 / 平均の速さ


def smooth_move_to_position(bus, goal_pos_deg: Dict[str, float], max_speed_dps: float,
                            torque_ff_fn: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
                            min_duration_s: float = 0.4, velocity_ff: bool = True) -> float:
    """今の位置から goal_pos_deg へ、最小躍度の曲線で動かす（終わるまで戻らない）。かかった長さ [s] を返す。"""
    from lerobot.motors.damiao.damiao_alignment import _ALIGN_STEP_S, OPENARM_ALIGN_KD, OPENARM_ALIGN_KP
    states = bus.sync_read_all_states()
    start: Dict[str, float] = {}
    for m in goal_pos_deg:
        if m not in bus.motors:
            continue
        p = states.get(m, {}).get("position")
        if p is None:  # 位置の分からないモーターは動かさない（soft_move_to_position と同じ）
            raise RuntimeError(f"smooth_move_to_position: {m!r} の位置の返事が無い（CAN の配線・モーターの電源）")
        start[m] = float(p)
    largest = max((abs(goal_pos_deg[m] - p) for m, p in start.items()), default=0.0)
    T = max(min_duration_s, PEAK * largest / max(max_speed_dps, 1e-6))
    n = max(1, round(T / _ALIGN_STEP_S))
    for k in range(n):
        t = (k + 1) / n
        s = t * t * t * (10 - 15 * t + 6 * t * t)
        ds = 30 * t * t * (1 - t) * (1 - t) / T  # ds/dt [1/s]
        target = {m: p + (goal_pos_deg[m] - p) * s for m, p in start.items()}
        ff = torque_ff_fn(target) if torque_ff_fn is not None else {}
        cmd = {m: (OPENARM_ALIGN_KP[m], OPENARM_ALIGN_KD[m], target[m],
                   (goal_pos_deg[m] - p) * ds if velocity_ff else 0.0, float(ff.get(m, 0.0)))
               for m, p in start.items()}
        bus._mit_control_batch(cmd)
        time.sleep(_ALIGN_STEP_S)
    return T


def follow_path(bus, joints, path_deg, max_speed_dps: float, gripper_deg: Optional[float] = None,
                torque_ff_fn: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
                min_duration_s: float = 0.4, start_tol_deg: float = 5.0) -> float:
    """細かい関節の軌道 path_deg（(M, len(joints)) [deg]・cuRobo の plan_grasp など）を、止まらずにたどる（2026-11-23 Munenori）。

    点ごとに止まる go() の代わり。軌道の上の位置を「一番大きく動く関節の角度の道のり」で測り、その上を最小躍度の曲線で進める。
    どの関節も max_speed_dps を超えない（道のりの 1 [deg] で、どの関節も 1 [deg] 以下しか動かないため）。
    出だしは今の角度から始める（軌道の始点と start_tol_deg より離れていれば動かさずに止める）。かかった長さ [s] を返す。"""
    import numpy as np
    from lerobot.motors.damiao.damiao_alignment import _ALIGN_STEP_S, OPENARM_ALIGN_KD, OPENARM_ALIGN_KP
    P = np.asarray(path_deg, float)
    states = bus.sync_read_all_states()
    now = []
    for m in joints:
        p = states.get(m, {}).get("position")
        if p is None:
            raise RuntimeError(f"follow_path: {m!r} の位置の返事が無い（CAN の配線・モーターの電源）")
        now.append(float(p))
    now = np.asarray(now)
    off = float(np.max(np.abs(now - P[0])))
    if off > start_tol_deg:
        raise RuntimeError(f"follow_path: 今の角度が軌道の始点から {off:.1f}° 離れている（{start_tol_deg:.0f}° まで）。動かさない")
    P = P.copy(); P[0] = now
    g0 = None
    if gripper_deg is not None and "gripper" in bus.motors:
        g0 = float(states.get("gripper", {}).get("position", gripper_deg))
    seg = np.max(np.abs(np.diff(P, axis=0)), axis=1) if len(P) > 1 else np.zeros(0)
    L = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(L[-1])
    if g0 is not None:
        total = max(total, abs(gripper_deg - g0))
    T = max(min_duration_s, PEAK * total / max(max_speed_dps, 1e-6))
    n = max(1, round(T / _ALIGN_STEP_S))
    for k in range(n):
        t = (k + 1) / n
        s = t * t * t * (10 - 15 * t + 6 * t * t)
        ds = 30 * t * t * (1 - t) * (1 - t) / T
        l = s * float(L[-1])
        i = int(np.clip(np.searchsorted(L, l, side="right") - 1, 0, max(len(P) - 2, 0)))
        if len(P) == 1 or seg[i] <= 1e-9:
            q, dq = P[min(i + 1, len(P) - 1)], np.zeros(P.shape[1])
        else:
            u = (l - L[i]) / seg[i]
            q = P[i] + (P[i + 1] - P[i]) * u
            dq = (P[i + 1] - P[i]) / seg[i] * float(L[-1]) * ds  # [deg/s]
        target = {m: float(q[c]) for c, m in enumerate(joints)}
        vel = {m: float(dq[c]) for c, m in enumerate(joints)}
        if g0 is not None:
            target["gripper"] = g0 + (gripper_deg - g0) * s
            vel["gripper"] = (gripper_deg - g0) * ds
        ff = torque_ff_fn(target) if torque_ff_fn is not None else {}
        bus._mit_control_batch({m: (OPENARM_ALIGN_KP[m], OPENARM_ALIGN_KD[m], target[m], vel[m], float(ff.get(m, 0.0)))
                                for m in target})
        time.sleep(_ALIGN_STEP_S)
    return T
