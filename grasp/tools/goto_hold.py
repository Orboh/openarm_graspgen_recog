#!/usr/bin/env python3
"""Move the left arm slowly to a given pose and hold it. On exit, return to **the pose it started from** before turning torque off (runs on Thor). **The arm moves.**

Purpose: capture with the arm out of the camera's view (Kota's doc: "capture before moving the arm, in a pose where the arm doesn't block the view").
hold_pose.py turns torque off right where it is when there are no waypoints (the arm drops from a raised pose), so this one
always returns to the start pose (= where the arm was resting limp) before turning torque off.

    ~/projects/lerobot/.venv/bin/python goto_hold.py --pose ~/poses/_tmpsets/cart-sym-v4-direct/left.yaml          # show the plan only
    ~/projects/lerobot/.venv/bin/python goto_hold.py --pose ... --go      # move → hold → (on stop file or Ctrl-C) return to the start pose → torque off
Stop: touch ~/grasp_runs/STOP_HOLD (or Ctrl-C)
"""
from __future__ import annotations

import argparse
import os
import sys
import signal
import time

import numpy as np
import yaml

URDF = os.path.expanduser("~/projects/openarm-orboh/urdf/openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
STOP = os.path.expanduser("~/grasp_runs/STOP_HOLD")
LO = np.degrees([-3.45, -3.30, -1.50, -0.01, -1.50, -0.75, -1.50]) + 2
HI = np.degrees([1.35, 0.15, 1.50, 2.40, 1.50, 0.75, 1.50]) - 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pose", required=True)
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--speed-dps", type=float, default=15.0)
    a = ap.parse_args()
    tgt = yaml.safe_load(open(a.pose))
    goal = {j: float(np.clip(tgt[j], LO[i], HI[i])) for i, j in enumerate(J)}
    print("[goto] target " + " ".join(f"{goal[j]:+.1f}" for j in J) + f"  ({a.speed_dps} deg/s)")
    if not a.go:
        print("[goto] DRY RUN (add --go to move)"); return 0

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    grav = OpenArmGravityModel(URDF, "left", 1.0).torque
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from grasp.soft_release import release_and_disconnect  # 2026-11-22 Sota: 力を抜くときはゆっくり（ダンプのように）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_l"), side="left", id=os.environ.get("GRASP_ROBOT_ID", "follower_left"), align_on_connect=False, cameras={}))
    if os.path.exists(STOP):
        os.remove(STOP)
    stop = {"v": False}
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell onto the cart
    connect_holding(robot, grav)
    obs = robot.get_observation(); start = {j: float(obs[f"{j}.pos"]) for j in J}
    print("[goto] start " + " ".join(f"{start[j]:+.1f}" for j in J), flush=True)
    try:
        soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=a.speed_dps)
        obs = robot.get_observation()
        print("[goto] reached " + " ".join(f"{obs[f'{j}.pos']:+.1f}" for j in J) + "  holding (stop: touch ~/grasp_runs/STOP_HOLD)", flush=True)
        while not stop["v"] and not os.path.exists(STOP):
            soft_move_to_position(robot.bus, goal, duration_s=0.5, torque_ff_fn=grav)  # keep sending the hold command
    finally:
        print("[goto] returning to the start pose", flush=True)
        soft_move_to_position(robot.bus, start, torque_ff_fn=grav, max_speed_dps=a.speed_dps)
        time.sleep(0.3)
        release_and_disconnect(robot, grav)
        print("[goto] back at start, torque off", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
