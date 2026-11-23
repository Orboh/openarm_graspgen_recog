#!/usr/bin/env python3
"""Collect hand-eye data in poses near the grasp (runs on Thor, lerobot venv). **The arm moves.** Dry run by default.

Visits small variations of the given anchor poses (e.g. pregrasp / lift from plan.json) and records joint angles + chest RGB-D at each.
Each pose is FK-checked (TCP ≥ floor + 6 cm, link7 ≥ floor + 10 cm); any that fail are skipped.
Path: start → raise straight up 8 cm → each pose → raise → start → torque off. Output format matches handeye_auto.py.

    ~/projects/lerobot/.venv/bin/python handeye_near.py <plan.json>          # print the list of poses
    ~/projects/lerobot/.venv/bin/python handeye_near.py <plan.json> --go
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

REPO = os.path.expanduser("~/projects/openarm-orboh")
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
FLOOR_Z = float(os.environ.get("GRASP_FLOOR_Z", "0.27")) + 0.02  # floor height in the base frame near the grasp area (+ margin)
if os.environ.get("GRASP_SITE") and "GRASP_FLOOR_Z" not in os.environ:  # 東京の台車以外で床の高さを決め忘れたら動かさない
    sys.exit("[grasp] GRASP_FLOOR_Z が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の値のまま床の確認をしない。grasp/config/sites/ で決める")
LO = np.degrees([-3.45, -3.30, -1.50, -0.01, -1.50, -0.75, -1.50]) + 3
HI = np.degrees([1.35, 0.15, 1.50, 2.40, 1.50, 0.75, 1.50]) - 3
# offsets around each anchor (J1..J7, deg). Mostly the wrist (J5–J7), to change how the marker faces the camera
VAR = [(0, 0, 0, 0, 0, 0, 0), (0, 0, 0, 0, 20, 0, 0), (0, 0, 0, 0, -20, 0, 0), (0, 0, 0, 0, 0, 15, 0), (0, 0, 0, 0, 0, -15, 0),
       (0, 0, 0, 0, 0, 0, -25), (0, 0, 0, 0, 0, 0, -50), (10, 0, 0, 0, 0, 0, -20), (-10, 0, 0, 0, 0, 0, -20),
       (0, 0, 0, 15, 15, 0, -30), (0, 0, 0, 15, -15, 10, -40), (0, 0, 10, 0, 0, -10, -15)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("plan", nargs="?", default=None)
    ap.add_argument("--poses", default=None, help="JSON from make_joint_sweep.py: visit exactly these poses (lerobot deg) instead of variations of plan.json")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--speed-dps", type=float, default=15.0)
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    import pinocchio as pin
    from ik.openarm_left_ik import OpenArmLeftIK
    ik = OpenArmLeftIK(URDF)
    poses = []
    anchors = []
    if a.poses:  # explicit list (kinematic-calibration sweep); still floor-checked below
        for q in json.load(open(a.poses))["poses"]:
            q = np.clip(np.array(q, float), LO, HI)
            M = ik.fk(np.deg2rad(q)); tcp = M.translation + M.rotation @ [0, 0, 0.2145]
            if tcp[2] >= FLOOR_Z + 0.06 and M.translation[2] >= FLOOR_Z + 0.10:
                poses.append(q)
    else:
        p = json.load(open(a.plan))
        anchors = [p["q_pre_deg"], p["q_lift_deg"]]
    for an in anchors:
        for v in VAR:
            q = np.clip(np.array(an) + v, LO, HI)
            M = ik.fk(np.deg2rad(q)); tcp = M.translation + M.rotation @ [0, 0, 0.2145]
            if tcp[2] >= FLOOR_Z + 0.06 and M.translation[2] >= FLOOR_Z + 0.10:
                poses.append(q)
    print(f"[near] {len(poses)} poses pass the floor check")
    for k, q in enumerate(poses):
        print(f"  #{k:02d} " + " ".join(f"{v:+6.1f}" for v in q))
    if not a.go:
        print("[near] DRY RUN (add --go to move)"); return 0

    import cv2
    import pyrealsense2 as rs
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # grasp/gravity.py を読む
    from grasp.gravity import gravity_torque_fn
    grav = gravity_torque_fn("left")  # 手と爪の質量入り（2026-11-23）
    out = os.path.expanduser(f"~/grasp_runs/handeye_near_{time.strftime('%Y%m%d_%H%M%S')}"); os.makedirs(out, exist_ok=True)
    pipe = rs.pipeline(); rc = rs.config(); rc.enable_device(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
    rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(rc); scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color); spatial, holes = rs.spatial_filter(), rs.hole_filling_filter(1)
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_l"), side="left", id=os.environ.get("GRASP_ROBOT_ID", "follower_left"), align_on_connect=False, cameras={}))
    robot.connect()
    o = robot.get_observation(); start = np.array([o[f"{j}.pos"] for j in J])
    M = ik.fk(np.deg2rad(start))
    qu, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, 0.08])), np.deg2rad(start), position_only=False)
    up = np.rad2deg(qu) if ok else start

    def go(q, speed=a.speed_dps):
        goal = {j: float(q[i]) for i, j in enumerate(J)}
        soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=speed)
        cmd = dict(goal)
        for _ in range(3):
            o = robot.get_observation(); err = {j: goal[j] - float(o[f"{j}.pos"]) for j in J}
            if max(abs(v) for v in err.values()) < 1.0:
                break
            for j in J:
                cmd[j] = goal[j] + float(np.clip(cmd[j] - goal[j] + err[j], -15, 15))
            soft_move_to_position(robot.bus, cmd, duration_s=1.0, torque_ff_fn=grav)

    qs, K = [], None
    try:
        go(up, 10)
        for k, q in enumerate(poses):
            go(q); time.sleep(1.0)
            o = robot.get_observation(); qa = np.array([o[f"{j}.pos"] for j in J])
            fs = align.process(pipe.wait_for_frames())
            col = np.asanyarray(fs.get_color_frame().get_data()).copy()
            dep = (np.asanyarray(holes.process(spatial.process(fs.get_depth_frame())).get_data()).astype(np.float32) * scale).astype(np.float16)
            if K is None:
                i = fs.get_color_frame().profile.as_video_stream_profile().intrinsics; K = [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]
            cv2.imwrite(f"{out}/color_{k:03d}.jpg", col); np.save(f"{out}/depth_{k:03d}.npy", dep); qs.append(qa)
            print(f"[near] #{k:02d} saved", flush=True)
    except KeyboardInterrupt:
        print("[near] Ctrl-C", flush=True)
    finally:
        np.savez(f"{out}/samples.npz", q_deg=np.array(qs), t=np.zeros(len(qs)), K=np.array(K if K else np.eye(3)))
        try:
            go(up, 12); go(start, 12)
        finally:
            robot.disconnect(); pipe.stop()
            print(f"[near] back at start, torque off. {len(qs)} samples -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
