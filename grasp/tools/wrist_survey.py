#!/usr/bin/env python3
"""Hand-guided wrist-camera survey (runs on Thor). **Does not drive the arm.**

2026-11-21 (Sota: クランプは手首のカメラで回って見ればわかる): the operator holds the left arm (torque OFF, lowered to rest
beforehand with release_arm.py) and points the wrist D405 at the chest-camera clamp from a few sides. Every --every s this
saves the wrist colour + depth and the joint angles read over CAN (read-only, no torque). On the PC,
fit_survey (below, --fit) turns the depth into base-frame points via FK + the wrist camera pose from
grasp/config/wrist_jaw_model.json, and prints the box around everything near the camera mount.

    ~/projects/lerobot/.venv/bin/python grasp/tools/wrist_survey.py --seconds 60 --every 2      # on Thor
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

READ = r'''
import can,time,numpy as np,json,os
'''


def read_q(bus, cal):
    import can
    q = []
    for sid in range(1, 8):
        bus.send(can.Message(arbitration_id=0x7FF, data=bytes([sid, 0, 0xCC, 0, 0, 0, 0, 0]), is_extended_id=False))
        end = time.monotonic() + 0.1
        while time.monotonic() < end:
            m = bus.recv(timeout=0.01)
            if m and m.arbitration_id == 0x10 + sid:
                d = bytes(m.data)
                q.append(np.degrees(((d[1] << 8) | d[2]) / 65535 * 25 - 12.5) - cal[f"joint_{sid}"]["homing_offset"])
                break
    return q


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=60)
    ap.add_argument("--every", type=float, default=2.0)
    a = ap.parse_args()
    import can
    import cv2
    import pyrealsense2 as rs
    cal = json.load(open(os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/follower_left.json")))
    bus = can.Bus(channel=os.environ.get("GRASP_CAN_PORT", "can_fol_l"), interface="socketcan", fd=True)
    out = os.path.expanduser(f"~/grasp_runs/wrist_survey_{time.strftime('%Y%m%d_%H%M%S')}"); os.makedirs(out)
    pipe = rs.pipeline(); rc = rs.config(); rc.enable_device(os.environ.get("GRASP_WRIST_SERIAL", "409122272149"))
    rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(rc); sc = prof.get_device().first_depth_sensor().get_depth_scale(); al = rs.align(rs.stream.color)
    qs, t0, k = [], time.time(), 0
    try:
        while time.time() - t0 < a.seconds:
            fs = al.process(pipe.wait_for_frames())
            q = read_q(bus, cal)
            if len(q) == 7:
                cv2.imwrite(f"{out}/color_{k:03d}.jpg", np.asanyarray(fs.get_color_frame().get_data()))
                np.save(f"{out}/depth_{k:03d}.npy", (np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * sc).astype(np.float16))
                qs.append(q); print(f"[survey] #{k:02d} q " + " ".join(f"{v:+.0f}" for v in q), flush=True); k += 1
            time.sleep(a.every)
    finally:
        np.save(f"{out}/q_deg.npy", np.array(qs)); pipe.stop(); bus.shutdown()
        print(f"[survey] {k} frames -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
