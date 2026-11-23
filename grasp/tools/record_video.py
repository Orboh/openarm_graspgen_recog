#!/usr/bin/env python3
"""Record the chest camera to <cap_dir>/video_chest.mp4 while a trial moves the arm.

2026-11-23 (Sota: 試行のデータは動画も含めてクラウドに). run_grasp.py starts this as a separate process before the arm
moves and stops it (SIGTERM) in its finally block, before uploading the folder to S3 (grasp/tools/s3_sync.sh).
A separate process so that encoding never slows the motor loop.

    python grasp/tools/record_video.py <cap_dir> [--fps 15] [--serial 344422070898] [--max-min 20]

Writes
  video_chest.mp4      color 640x480, --fps frames/s (every 2nd frame of the 30 fps stream at 15)
  video_chest_ts.csv   frame index, wall time (same clock as traj.csv's t), so the video lines up with the joints
Pause: while <cap_dir>/.video_pause exists the camera is released (another reader, e.g. run_grasp --probe, needs it);
<cap_dir>/.video_paused is written once it is released and removed when recording resumes.
"""
import argparse
import os
import signal
import sys
import time

import cv2
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--serial", default=os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
    ap.add_argument("--max-min", type=float, default=20.0, help="stop by itself after this many minutes")
    a = ap.parse_args()
    import pyrealsense2 as rs

    stop = {"v": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.__setitem__("v", True))
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__("v", True))
    pause, paused = os.path.join(a.cap_dir, ".video_pause"), os.path.join(a.cap_dir, ".video_paused")
    mp4 = os.path.join(a.cap_dir, "video_chest.mp4")
    vw = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (640, 480))
    ts = open(os.path.join(a.cap_dir, "video_chest_ts.csv"), "w")
    ts.write("frame,t\n")
    step = max(1, round(30 / a.fps))
    n = k = 0
    pipe = None
    t_end = time.time() + a.max_min * 60
    try:
        while not stop["v"] and time.time() < t_end:
            if os.path.exists(pause):
                if pipe is not None:
                    pipe.stop(); pipe = None
                    open(paused, "w").close()
                    print("[video] paused (camera released)", flush=True)
                time.sleep(0.05)
                continue
            if pipe is None:
                if os.path.exists(paused):
                    os.remove(paused)
                pipe = rs.pipeline(); cfg = rs.config(); cfg.enable_device(a.serial)
                cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                pipe.start(cfg)
                print(f"[video] recording {a.serial} -> {mp4}", flush=True)
            try:
                fs = pipe.wait_for_frames(2000)
            except RuntimeError as e:
                print(f"[video] no frame ({e}); reopening", flush=True)
                try:
                    pipe.stop()
                except Exception:  # noqa: BLE001
                    pass
                pipe = None
                time.sleep(0.5)
                continue
            k += 1
            if k % step:
                continue
            c = fs.get_color_frame()
            if not c:
                continue
            vw.write(np.asanyarray(c.get_data()))
            ts.write(f"{n},{time.time():.3f}\n")
            n += 1
    finally:
        if pipe is not None:
            pipe.stop()
        vw.release(); ts.close()
        for p in (paused,):
            if os.path.exists(p):
                os.remove(p)
        print(f"[video] stopped: {n} frames ({n / a.fps:.0f} s) in {mp4}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
