#!/usr/bin/env python3
"""Hand-eye data collection (runs on Thor): read the left arm's joint angles **without powering any motor**, and grab chest D435i RGB-D frames.

The person moves the arm slowly by hand and pauses about 1 s at each pose. Only samples where the arm is still are kept.
Motors are only queried for status (refresh 0x7FF). **Enable / position commands are never sent** (the arm stays limp).

Output: ~/grasp_runs/handeye_<ts>/
  samples.npz   q_deg (N,7) lerobot degrees (raw - homing_offset), t (N,)
  color_XXX.jpg / depth_XXX.npy (m, float16)
  preview.jpg   latest frame + sample count (shown on this PC)

    ~/projects/lerobot/.venv/bin/python handeye_collect.py --seconds 90 --max 40
"""
from __future__ import annotations

import argparse
import json
import os
import time

import can
import cv2
import numpy as np
import pyrealsense2 as rs

CHEST_SERIAL = os.environ.get("GRASP_CHEST_SERIAL", "344422070898")
CAL = os.path.expanduser("~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/" + os.environ.get("GRASP_ROBOT_ID", "follower_left") + ".json")
P_MAX = 12.5  # DM43xx / DM8009 share the 25 rad range


def read_joints(bus, offsets):
    out = []
    for sid in range(1, 8):
        bus.send(can.Message(arbitration_id=0x7FF, data=bytes([sid, 0, 0xCC, 0, 0, 0, 0, 0]), is_extended_id=False))
        d = None
        end = time.monotonic() + 0.03
        while time.monotonic() < end:
            m = bus.recv(timeout=0.005)
            if m is not None and m.arbitration_id == 0x10 + sid:
                d = bytes(m.data)
                break
        if d is None:
            return None
        raw = np.degrees(((d[1] << 8) | d[2]) / 65535 * 2 * P_MAX - P_MAX)
        out.append(raw - offsets[sid - 1])
    return np.array(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default=os.environ.get("GRASP_CAN_PORT", "can_fol_l"))
    ap.add_argument("--seconds", type=float, default=90.0)
    ap.add_argument("--max", type=int, default=40)
    ap.add_argument("--still-deg", type=float, default=0.5, help="treat the arm as still when every joint changed less than this")
    ap.add_argument("--min-move-deg", type=float, default=5.0, help="minimum change from the last saved pose (avoid duplicates)")
    a = ap.parse_args()

    cal = json.load(open(CAL))
    offsets = [cal[f"joint_{i}"]["homing_offset"] for i in range(1, 8)]
    out = os.path.expanduser(f"~/grasp_runs/handeye_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out, exist_ok=True)

    bus = can.Bus(channel=a.iface, interface="socketcan", fd=True)
    pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(CHEST_SERIAL)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    spatial, holes = rs.spatial_filter(), rs.hole_filling_filter(1)
    for _ in range(30):
        pipe.wait_for_frames()
    intr = None
    qs, ts = [], []
    prev = None
    t0 = time.time()
    print(f"[collect] {out}  move the arm slowly by hand and pause ~1 s at each pose (motors stay off)", flush=True)
    try:
        while time.time() - t0 < a.seconds and len(qs) < a.max:
            q = read_joints(bus, offsets)
            fs = align.process(pipe.wait_for_frames())
            q2 = read_joints(bus, offsets)
            color = np.asanyarray(fs.get_color_frame().get_data()).copy()
            if q is None or q2 is None:  # 関節が読めなくてもプレビューは出す（電源・非常停止に気付けるように・2026-11-21）
                img = color.copy()
                cv2.putText(img, "NO JOINT DATA - check arm power / e-stop", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(img, f"samples {len(qs)}/{a.max}  {a.seconds - (time.time() - t0):.0f}s left", (8, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.imwrite(f"{out}/preview.jpg", img)
                continue
            still = prev is not None and np.max(np.abs(q2 - prev)) < a.still_deg and np.max(np.abs(q2 - q)) < a.still_deg
            fresh = not qs or np.max(np.abs(q2 - qs[-1])) > a.min_move_deg
            prev = q2
            status = "MOVING"
            if still and fresh:
                df = holes.process(spatial.process(fs.get_depth_frame()))
                depth = (np.asanyarray(df.get_data()).astype(np.float32) * scale).astype(np.float16)
                k = len(qs)
                cv2.imwrite(f"{out}/color_{k:03d}.jpg", color)
                np.save(f"{out}/depth_{k:03d}.npy", depth)
                qs.append(q2); ts.append(time.time())
                if intr is None:
                    i = fs.get_color_frame().profile.as_video_stream_profile().intrinsics
                    intr = [[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]]
                status = f"SAVED #{k}"
                print(f"[collect] saved #{k}  q=" + " ".join(f"{v:+.0f}" for v in q2), flush=True)
            elif still:
                status = "still (same pose) - move to a new pose"
            prev_img = color.copy()
            cv2.putText(prev_img, f"samples {len(qs)}/{a.max}  {status}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0) if status.startswith("SAVED") else (0, 200, 255), 2)
            cv2.putText(prev_img, f"{a.seconds - (time.time() - t0):.0f}s left", (8, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(f"{out}/preview.jpg", prev_img)
            time.sleep(0.25)
    finally:
        pipe.stop(); bus.shutdown()
        np.savez(f"{out}/samples.npz", q_deg=np.array(qs), t=np.array(ts), K=np.array(intr if intr else np.eye(3)))
        print(f"[collect] done: {len(qs)} samples -> {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
