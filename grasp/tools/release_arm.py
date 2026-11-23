#!/usr/bin/env python3
"""Lower the left arm from a held pose (e.g. run_grasp.py --hold-home) to a resting pose, then torque off. **The arm moves.**

    ~/projects/lerobot/.venv/bin/python grasp/release_arm.py            # show the plan
    ~/projects/lerobot/.venv/bin/python grasp/release_arm.py --go
Default rest = the limp resting pose seen on 2026-11-20 (arm folded near the body); slow (20 deg/s).
"""
from __future__ import annotations

import argparse
import os
import sys

URDF = os.path.expanduser("~/projects/openarm-orboh/urdf/openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
REST = [float(v) for v in os.environ.get("GRASP_REST", "20.3,-3.6,-0.9,92.8,-2.5,-11.0,22.7").split(",")]
if os.environ.get("GRASP_SITE") and "GRASP_REST" not in os.environ:  # 東京の台車以外で休みの姿勢を決め忘れたら動かさない
    sys.exit("[grasp] GRASP_REST が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の休みの姿勢へは下ろさない。grasp/config/sites/ で決める")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--rest", default=",".join(str(v) for v in REST))
    ap.add_argument("--speed-dps", type=float, default=20.0)
    a = ap.parse_args()
    rest = [float(v) for v in a.rest.split(",")]
    print("[release] rest " + " ".join(f"{v:+.1f}" for v in rest) + f" at {a.speed_dps} deg/s")
    if not a.go:
        print("[release] DRY RUN (add --go to move)"); return 0
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    grav = OpenArmGravityModel(URDF, "left", 1.0).torque
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from grasp.soft_release import release_and_disconnect  # 2026-11-22 Sota: 力を抜くときはゆっくり（ダンプのように）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_l"), side="left", id=os.environ.get("GRASP_ROBOT_ID", "follower_left"), align_on_connect=False, cameras={}))
    from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell onto the cart
    connect_holding(robot, grav)
    try:
        o = robot.get_observation()
        goal = {j: rest[i] for i, j in enumerate(J)}; goal["gripper"] = float(o["gripper.pos"])
        soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=a.speed_dps)
    finally:
        release_and_disconnect(robot, grav)
        print("[release] at rest, torque off", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
