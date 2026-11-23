#!/usr/bin/env python3
"""トルクを入れて腕を少しだけ動かす確かめ（左右どちらも）。**--go を付けると腕が動く。** 付けなければ計算だけ。

2026-11-22・作業者: Munenori。右腕を足したとき、指令どおりに動くか・重力の補正が効くか・グリッパーの開閉の指令を確かめる。

段:
  1. 今の角度を読む（休みの姿勢のはず。GRASP_REST(_RIGHT) から 5° 以上ずれていたら動かさない）
  2. 指先（TCP）を向きを保ったまま真上に --lift（既定 8 cm）上げる姿勢を IK で解く（run_grasp.py の raised と同じ）
     途中の 20 点で、手首と指先が机（GRASP_FLOOR_Z）より上か・関節が可動域の中かを確かめる
  3. --dps（既定 10 deg/s）でゆっくり上げる → 3 秒止めて、関節の誤差を読む
  4. --gripper なら、上げたままグリッパーを開く（GRASP_GRIPPER_OPEN(_RIGHT)）→ 閉じる（CLOSE）
  5. 休みの姿勢にゆっくり戻す → トルクを切る（戻した所で切るので腕は落ちない）
止める: Ctrl-C（休みの姿勢に戻ってからトルクを切る）。危ないときは非常停止。

    set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
    ~/projects/lerobot/.venv/bin/python grasp/tools/torque_test.py --arm right                 # 計算だけ
    ~/projects/lerobot/.venv/bin/python grasp/tools/torque_test.py --arm right --go --gripper  # 動かす
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
TCP_LEN, TIP_LEN = 0.2145, 0.2295


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--lift", type=float, default=0.08, help="指先を真上に上げる量 [m]")
    ap.add_argument("--dps", type=float, default=10.0, help="関節の速さ [deg/s]")
    ap.add_argument("--gripper", action="store_true", help="上げたままグリッパーを開いて閉じる")
    ap.add_argument("--go", action="store_true", help="腕を動かす（無ければ計算だけ）")
    ap.add_argument("--to-rest", action="store_true",
                    help="今の角度から休みの姿勢へゆっくり（5 deg/s）合わせてトルクを切るだけ（どの関節も 15° 以内のときだけ・--go と一緒に）")
    a = ap.parse_args()
    side = a.arm
    suf = "_RIGHT" if side == "right" else ""
    need = ["GRASP_CAN_PORT", "GRASP_ROBOT_ID", "GRASP_REST", "GRASP_FLOOR_Z", "GRASP_GRIPPER_OPEN", "GRASP_GRIPPER_CLOSE"]
    env = {k: os.environ.get(k + (suf if k != "GRASP_FLOOR_Z" else "")) for k in need}
    missing = [k + (suf if k != "GRASP_FLOOR_Z" else "") for k, v in env.items() if v is None]
    if missing:
        sys.exit(f"[torque] 置き場の env に無い: {', '.join(missing)}（set -a; source grasp/config/sites/<置き場>.env; set +a）")
    rest = np.array([float(v) for v in env["GRASP_REST"].split(",")])
    fz, g_open, g_close = float(env["GRASP_FLOOR_Z"]), float(env["GRASP_GRIPPER_OPEN"]), float(env["GRASP_GRIPPER_CLOSE"])

    import pinocchio as pin
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(URDF, side=side)

    def heights(qd):
        M = ik.fk(np.deg2rad(qd))  # hand_tcp（link7 と同じ所）
        pin.forwardKinematics(ik.model, ik.data, np.deg2rad(qd))
        l7 = ik.data.oMi[ik.model.getJointId(f"openarm_{side}_joint7")]
        return float(l7.translation[2]), float((l7.translation + l7.rotation[:, 2] * TIP_LEN)[2]), M

    def raised(qd):
        _, _, M = heights(qd)
        T = pin.SE3(M.rotation, M.translation + np.array([0, 0, a.lift]))
        q, ok, _, _ = ik.solve(T, np.deg2rad(qd), position_only=False, max_iter=300)
        if not ok or not ik.within_limits(q):
            sys.exit(f"[torque] {a.lift * 100:.0f} cm 上げる姿勢が解けない。動かさない")
        return np.rad2deg(q)

    def path_ok(q0, q1):
        """関節の直線の途中で、手首と爪の先が机より上（端の低い方 − 1 cm まで許す）・関節が可動域の中。"""
        l7a, tipa, _ = heights(q0); l7b, tipb, _ = heights(q1)
        l7min, tipmin = min(l7a, l7b) - 0.01, min(tipa, tipb) - 0.01
        for s in np.linspace(0, 1, 20):
            q = q0 + (q1 - q0) * s
            l7, tip, _ = heights(q)
            if l7 < l7min or tip < tipmin or tip < fz - 0.005 or not ik.within_limits(np.deg2rad(q)):
                sys.exit(f"[torque] 途中（s={s:.2f}）で手首 {l7 - fz:+.3f}・爪の先 {tip - fz:+.3f} m（机から）か可動域の外。動かさない")

    up = raised(rest)
    path_ok(rest, up)
    l7r, tipr, _ = heights(rest); l7u, tipu, _ = heights(up)
    print(f"[torque] {side} 腕・CAN {env['GRASP_CAN_PORT']}・較正 {env['GRASP_ROBOT_ID']}・{a.dps:.0f} deg/s")
    print("[torque] 休み   " + " ".join(f"{v:+6.1f}" for v in rest) + f"   手首 {l7r - fz:+.3f}・爪の先 {tipr - fz:+.3f} m（机から）")
    print("[torque] 上げる " + " ".join(f"{v:+6.1f}" for v in up) + f"   手首 {l7u - fz:+.3f}・爪の先 {tipu - fz:+.3f} m（机から）")
    print(f"[torque] 関節の動き 最大 {np.max(np.abs(up - rest)):.1f}°" + (f"・グリッパー 開 {g_open:+.0f} → 閉 {g_close:+.0f}" if a.gripper else ""))
    if not a.go:
        print("[torque] DRY RUN（--go で動かす）")
        return 0

    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # grasp/gravity.py を読む
    from grasp.gravity import gravity_torque_fn
    grav = gravity_torque_fn(side)  # 手と爪の質量入り（2026-11-23）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=env["GRASP_CAN_PORT"], side=side, id=env["GRASP_ROBOT_ID"],
                                                  align_on_connect=False, cameras={}))
    robot.connect()

    def now():
        o = robot.get_observation()
        return np.array([float(o[f"{j}.pos"]) for j in J]), float(o["gripper.pos"])

    def go(q, g, phase, speed=a.dps):
        goal = {j: float(q[i]) for i, j in enumerate(J)}
        goal["gripper"] = float(g)
        soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=speed)
        age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
        if age > 0.3:
            raise SystemExit(f"[torque] {phase}: モーターの返事が {age:.1f} 秒来ない（非常停止・電源）。指令をやめる")
        qn, gn = now()
        e = qn - q
        print(f"[torque] {phase}: 関節の誤差 " + " ".join(f"{v:+5.1f}" for v in e) + f"（最大 {np.max(np.abs(e)):.1f}°）・グリッパー {gn:+.1f}（指令 {g:+.0f}）",
              flush=True)
        return qn, gn

    q0, g0 = now()
    if a.to_rest:  # 2026-11-22: 手で置いた姿勢が休みの姿勢から数度ずれていたので、腕で合わせる
        try:
            d = np.max(np.abs(q0 - rest))
            print("[torque] 今     " + " ".join(f"{v:+6.1f}" for v in q0) + f"   休みの姿勢から最大 {d:.1f}°", flush=True)
            if d > 15.0:
                raise SystemExit(f"[torque] 休みの姿勢から {d:.1f}° 離れている（15° まで）。手で近づけてから打つ（動かしていない）")
            path_ok(q0, rest)
            go(rest, g0, "休みの姿勢に合わせた", speed=5.0)
        finally:
            robot.disconnect()
            print("[torque] トルクを切った", flush=True)
        return 0
    print("[torque] 今     " + " ".join(f"{v:+6.1f}" for v in q0) + f"   グリッパー {g0:+.1f}", flush=True)
    if np.max(np.abs(q0 - rest)) > 5.0:
        robot.disconnect()
        sys.exit(f"[torque] 今の角度が休みの姿勢から {np.max(np.abs(q0 - rest)):.1f}° ずれている。休みの姿勢に置いてから打つ（動かしていない）")
    try:
        go(up, g0, "上げた")
        time.sleep(3.0)
        qh, _ = now()
        print(f"[torque] 3 秒止めた後: 上げた姿勢からの誤差 最大 {np.max(np.abs(qh - up)):.1f}°（重力の補正が効いていれば数度以内）", flush=True)
        if a.gripper:
            go(up, g_open, "グリッパーを開いた", speed=a.dps)
            time.sleep(1.0)
            go(up, g_close, "グリッパーを閉じた", speed=a.dps)
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("[torque] Ctrl-C: 休みの姿勢に戻す", flush=True)
    finally:
        try:
            go(q0, g0 if not a.gripper else g_close, "休みの姿勢に戻した", speed=min(a.dps, 10.0))
        finally:
            robot.disconnect()  # トルクを切る（休みの姿勢に戻した後）
            print("[torque] トルクを切った", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
