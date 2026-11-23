#!/usr/bin/env python3
"""Move one arm (GRASP_ARM) from where it is to a home YAML and keep holding (torque on). **The arm moves.** Dry run by default.

2026-11-22 (Sota: グリッパーの初期姿勢の位置を 15 cm 後ろへ): the joint-space straight line is checked first with
grasp/arm_collision.py (link meshes + gripper vs chest camera, clamp, body, the other arm's home box). If it is not clear,
it goes via the current pose raised 8 cm. Slow (15 deg/s), gravity feed-forward, like handeye_near.py.

    GRASP_ARM=left ~/projects/lerobot/.venv/bin/python grasp/tools/move_home.py grasp/config/home_left_cart-sym-v5.yaml [--go]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("home")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--start", default=None, help="dry run: start pose (deg, comma separated)")
    ap.add_argument("--speed-dps", type=float, default=15.0)
    ap.add_argument("--standby", default=None, help="standby YAML: if needed go start -> raised 8-16 cm -> standby -> home (11/22: from a limp pose resting on the cart edge)")
    a = ap.parse_args()
    import pinocchio as pin
    from ik.openarm_right_ik import arm_ik_class
    from grasp.arm_collision import ArmCollision, describe
    ARM = os.environ.get("GRASP_ARM", "left")
    ik = arm_ik_class()(URDF); ac = ArmCollision(ik)
    h = yaml.safe_load(open(a.home)); home = np.array([float(h[j]) for j in J])
    robot = None
    if a.go:
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
        robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_r" if ARM == "right" else "can_fol_l"),
                                                      side=ARM, id=os.environ.get("GRASP_ROBOT_ID", f"follower_{ARM}"),
                                                      align_on_connect=False, cameras={}, disable_torque_on_disconnect=False))
        from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
        from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell
        connect_holding(robot, OpenArmGravityModel(URDF, ARM, 1.0).torque)
        o = robot.get_observation(); start = np.array([float(o[f"{j}.pos"]) for j in J])
    else:
        start = np.array([float(v) for v in (a.start.split(",") if a.start else [0] * 7)])
    print(f"[home] {ARM}: start " + " ".join(f"{v:+.1f}" for v in start) + "  ->  home " + " ".join(f"{v:+.1f}" for v in home))
    path = [home]
    ok, w = ac.check_path(start, home)
    if not ok:
        M = ik.fk(np.deg2rad(start))
        q, s_ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, 0.08])), np.deg2rad(start), position_only=False)
        up = np.rad2deg(q)
        ok1, w1 = ac.check_path(start, up); ok2, w2 = ac.check_path(up, home)
        if s_ok and ok1 and ok2:
            path = [up, home]; print("[home] straight line not clear -> via the start raised 8 cm")
        else:
            path = None
            if a.standby:
                sy = yaml.safe_load(open(a.standby)); sb = np.array([float(sy[j]) for j in J])
                for dz in (0.08, 0.12, 0.16):
                    q, s_ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, dz])), np.deg2rad(start), position_only=False)
                    if not (s_ok and ik.within_limits(q)):
                        continue
                    up = np.rad2deg(q)
                    if ac.check_path(start, up)[0] and ac.check_path(up, sb)[0] and ac.check_path(sb, home)[0]:
                        path = [up, sb, home]; print(f"[home] via the start raised {dz*100:.0f} cm and the standby pose"); break
            if path is None:
                print(f"[home] not clear: {describe(w)} / via raise: {describe(w1 if not ok1 else w2)}. Not moving")
                if robot:
                    robot.disconnect()
                return 1
    print("[home] path clear (arm_collision)")
    if not a.go:
        print("[home] DRY RUN (add --go to move)"); return 0
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    grav = OpenArmGravityModel(URDF, ARM, 1.0).torque
    try:
        for q in path:
            goal = {j: float(q[i]) for i, j in enumerate(J)}
            soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=a.speed_dps)
        o = robot.get_observation()
        print("[home] reached, max joint error %.1f deg; holding (torque on)" % max(abs(float(o[f'{j}.pos']) - home[i]) for i, j in enumerate(J)))
    finally:
        robot.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
