#!/usr/bin/env python3
"""Hand-eye data collection with the robot moving itself (runs on Thor, lerobot venv). **The left arm moves.**

From the current pose, moves slowly (default 15 deg/s ≈ 0.26 rad/s) through a fixed set of small offsets. At each pose it
stops for 1.5 s, then records the joint angles (lerobot logical degrees = URDF radians in degrees) and a chest RGB-D frame.
At the end it returns to the starting pose and turns torque off (the arm goes limp in the pose it started from).

- Dry run by default (only prints the plan, sends nothing to the motors). Moves only with --go.
- Gravity feed-forward: lerobot's OpenArmGravityModel (same mechanism as align_gravity at startup).
- The gripper is never commanded.
- Ctrl-C: tries to return to the starting pose, then turns torque off.

Output: ~/grasp_runs/handeye_auto_<ts>/ (same format as handeye_collect.py → can be fed straight into handeye_icp.py)

    ~/projects/lerobot/.venv/bin/python handeye_auto.py            # show the plan only
    ~/projects/lerobot/.venv/bin/python handeye_auto.py --go       # run it
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

URDF = os.path.expanduser("~/projects/openarm-orboh/urdf/openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
# offsets from the starting pose (deg): J1, J2, J3, J4, J5, J6, J7
# first 1 joint at a time (direction check), then combinations
OFFSETS = [
    (0, 0, 0, 0, 0, 0, 0),
    (15, 0, 0, 0, 0, 0, 0), (-15, 0, 0, 0, 0, 0, 0),
    (0, 0, 0, 20, 0, 0, 0), (0, 0, 0, 40, 0, 0, 0),
    (0, 0, 15, 20, 0, 0, 0),
    (0, 0, 0, 20, 20, 0, 0), (0, 0, 0, 20, -20, 0, 0),
    (0, 0, 0, 20, 0, 15, 0), (0, 0, 0, 20, 0, -15, 0),
    (0, 0, 0, 20, 0, 0, 30),
    (10, 0, -10, 35, 15, 10, -20),
    (-10, 0, 10, 35, -15, -10, 20),
    (15, 0, 0, 30, 0, 10, 0),
    (-15, 0, 0, 30, 0, -10, 0),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually move (dry run if omitted)")
    ap.add_argument("--speed-dps", type=float, default=15.0)
    ap.add_argument("--settle-s", type=float, default=1.5)
    ap.add_argument("--port", default=os.environ.get("GRASP_CAN_PORT", "can_fol_l"))
    a = ap.parse_args()

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel

    cfg = OpenArmFollowerConfig(port=a.port, side="left", id=os.environ.get("GRASP_ROBOT_ID", "follower_left"), align_on_connect=False, cameras={})
    robot = OpenArmFollower(cfg)

    if not a.go:
        print("[auto] DRY RUN: nothing is sent to the motors. Plan (offsets from the start pose, deg):")
        for k, o in enumerate(OFFSETS):
            print(f"  #{k:02d} " + " ".join(f"{v:+4.0f}" for v in o))
        print(f"  speed {a.speed_dps} deg/s ({np.radians(a.speed_dps):.2f} rad/s), stop {a.settle_s} s at each pose, return to the start pose at the end")
        return 0

    import cv2
    import pyrealsense2 as rs
    # Do NOT use cfg.joint_limits (the default is a tiny ±5 deg, so clipping to it would yank the arm toward 0).
    # Use the measured left-arm limits (DimOS openarm adapter, rad) instead.
    LO = np.degrees([-3.45, -3.30, -1.50, -0.01, -1.50, -0.75, -1.50])
    HI = np.degrees([1.35, 0.15, 1.50, 2.40, 1.50, 0.75, 1.50])
    out = os.path.expanduser(f"~/grasp_runs/handeye_auto_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)
    pipe = rs.pipeline(); rc = rs.config(); rc.enable_device(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
    rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(rc); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color); spatial, holes = rs.spatial_filter(), rs.hole_filling_filter(1)
    for _ in range(30):
        pipe.wait_for_frames()

    grav = OpenArmGravityModel(URDF, "left", 1.0).torque

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

    from grasp.soft_release import release_and_disconnect  # 2026-11-22 Sota: 力を抜くときはゆっくり（ダンプのように）
    from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell onto the cart
    connect_holding(robot, grav)
    obs = robot.get_observation()
    start = {j: float(obs[f"{j}.pos"]) for j in J}
    for i, j in enumerate(J):  # don't move if the start pose itself is outside the limits
        if not (LO[i] < start[j] < HI[i]):
            release_and_disconnect(robot, grav); pipe.stop()
            raise SystemExit(f"[auto] start pose {j}={start[j]:.1f} is outside the limits [{LO[i]:.0f},{HI[i]:.0f}]. Stopping")
    print("[auto] start pose " + " ".join(f"{start[j]:+.1f}" for j in J), flush=True)
    qs, K = [], None

    def move(goal):
        g = {j: float(np.clip(goal[j], LO[i] + 2, HI[i] - 2)) for i, j in enumerate(J)}
        soft_move_to_position(robot.bus, g, torque_ff_fn=grav, max_speed_dps=a.speed_dps)

    try:
        for k, o in enumerate(OFFSETS):
            goal = {j: start[j] + o[i] for i, j in enumerate(J)}
            move(goal)
            time.sleep(a.settle_s)
            obs = robot.get_observation()
            q = np.array([obs[f"{j}.pos"] for j in J])
            fs = align.process(pipe.wait_for_frames())
            color = np.asanyarray(fs.get_color_frame().get_data()).copy()
            depth = (np.asanyarray(holes.process(spatial.process(fs.get_depth_frame())).get_data()).astype(np.float32) * scale).astype(np.float16)
            if K is None:
                i = fs.get_color_frame().profile.as_video_stream_profile().intrinsics
                K = [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]
            cv2.imwrite(f"{out}/color_{k:03d}.jpg", color); np.save(f"{out}/depth_{k:03d}.npy", depth)
            cv2.imwrite(f"{out}/preview.jpg", color)
            qs.append(q)
            err = q - np.array([goal[j] for j in J])
            print(f"[auto] #{k:02d} reached  q=" + " ".join(f"{v:+.1f}" for v in q) + f"  max err {np.max(np.abs(err)):.1f} deg", flush=True)
    except KeyboardInterrupt:
        print("[auto] Ctrl-C: returning to the start pose", flush=True)
    finally:
        np.savez(f"{out}/samples.npz", q_deg=np.array(qs), t=np.zeros(len(qs)), K=np.array(K if K else np.eye(3)))
        try:
            move(start)
            time.sleep(0.5)
        finally:
            release_and_disconnect(robot, grav); pipe.stop()
            print(f"[auto] back at start, torque off. {len(qs)} samples -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
