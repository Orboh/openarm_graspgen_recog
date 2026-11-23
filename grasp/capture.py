#!/usr/bin/env python3
"""Step 1: capture a point cloud with the chest D435i (aligned color+depth + post-processing filters).

The pipeline/align setup comes from ik/detect_okra.py:make_pipeline. Here spatial/temporal/hole-filling
filters are added and depth is turned into a point cloud (camera optical frame, metres).

Usage (on the collection PC, with no other process holding the camera):
    python grasp/capture.py --out grasp/runs/cap_$(date +%H%M%S)
    → writes <out>/scene.ply  color.png  depth.npy(m)  intrinsics.json

Notes:
- The D435i is exclusive to one process. Stop record / stream_chest first.
- Keep the object at least ~30 cm from the camera (depth isn't returned below the minimum range).
- Capture with the arm out of the field of view (before moving the arm).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

CHEST_SERIAL = os.environ.get("GRASP_CHEST_SERIAL", "344422070898")  # D435i (chest), same as ik/detect_okra.py


def capture(serial: str = CHEST_SERIAL, w: int = 640, h: int = 480, fps: int = 30,
            warmup: int = 30, temporal_frames: int = 10, preset: str | None = None, hole_fill: bool = True,
            tries: int = 4):
    """Return (color_bgr[H,W,3] uint8, depth_m[H,W] float32, K[3,3], intr_dict).
    2026-11-22 21:41: since a third RealSense (right wrist D405) was plugged in, some chest captures came back with
    more than half of the depth at 65.5 m (an invalid frame): every capture is checked and taken again (up to `tries`)."""
    for k in range(tries):
        out = _capture_once(serial, w, h, fps, warmup, temporal_frames, preset, hole_fill)
        d = out[1]; v = d[d > 0]
        bad = float((v > 5.0).mean()) if len(v) else 1.0
        if bad < 0.05 and len(v) > 0.3 * d.size:
            return out
        print(f"[capture] depth frame looks broken ({bad*100:.0f} % beyond 5 m, {len(v)/d.size*100:.0f} % valid): again ({k + 1}/{tries})", flush=True)
    raise SystemExit("[capture] the chest depth stays broken: check the camera cable / USB")


def _capture_once(serial, w, h, fps, warmup, temporal_frames, preset, hole_fill):
    import pyrealsense2 as rs

    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg)
    try:
        ds = prof.get_device().first_depth_sensor()
        depth_scale = ds.get_depth_scale()
        if preset:  # e.g. "High Accuracy" / "Hand": drop low-confidence depth (shiny aluminium) instead of reporting floor depth there
            r = ds.get_option_range(rs.option.visual_preset)
            names = {ds.get_option_value_description(rs.option.visual_preset, i): i for i in range(int(r.min), int(r.max) + 1)}
            if preset not in names:
                raise SystemExit(f"[capture] unknown preset {preset!r}; have {list(names)}")
            for k in range(5):  # set_option right after start sometimes fails with "Resource temporarily unavailable"
                try:
                    ds.set_option(rs.option.visual_preset, names[preset]); break
                except RuntimeError:
                    if k == 4:
                        raise
                    time.sleep(0.3)
        align = rs.align(rs.stream.color)
        # Order follows the Intel recommendation: (decimation skipped to keep resolution) → depth→disparity
        # → spatial → temporal → disparity→depth → hole filling.
        to_disp, to_depth = rs.disparity_transform(True), rs.disparity_transform(False)
        spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)
        for _ in range(warmup):  # auto-exposure settling
            pipe.wait_for_frames()
        depth_f = color_f = None
        for _ in range(temporal_frames):  # let the temporal filter accumulate a few frames
            fs = align.process(pipe.wait_for_frames())
            d = fs.get_depth_frame()
            color_f = fs.get_color_frame()
            d = to_disp.process(d)
            d = spatial.process(d)
            d = temporal.process(d)
            d = to_depth.process(d)
            depth_f = holes.process(d) if hole_fill else d  # hole filling would paint floor depth into the dropped pixels
        intr = color_f.profile.as_video_stream_profile().intrinsics
        color = np.asanyarray(color_f.get_data()).copy()
        depth_m = np.asanyarray(depth_f.get_data()).astype(np.float32) * depth_scale
    finally:
        pipe.stop()
    K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], dtype=np.float64)
    intr_d = {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
              "ppx": intr.ppx, "ppy": intr.ppy, "model": str(intr.model), "coeffs": list(intr.coeffs),
              "depth_scale": depth_scale, "serial": serial}
    return color, depth_m, K, intr_d


def depth_to_points(depth_m: np.ndarray, K: np.ndarray, color_bgr: np.ndarray | None = None,
                    zmin: float = 0.25, zmax: float = 1.2):
    """Depth image → (N,3) points in the camera optical frame (+x right, +y down, +z forward) and (N,3) RGB 0-1."""
    h, w = depth_m.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_m
    valid = np.isfinite(z) & (z > zmin) & (z < zmax)
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], axis=-1)[valid].astype(np.float32)
    rgb = None
    if color_bgr is not None:
        rgb = color_bgr[valid][:, ::-1].astype(np.float32) / 255.0
    return pts, rgb


def to_o3d(pts: np.ndarray, rgb: np.ndarray | None = None):
    import open3d as o3d

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    if rgb is not None:
        pc.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    return pc


def save_capture(out_dir: str, color, depth_m, K, intr_d):
    """color.png / depth.npy / intrinsics.json are always written. scene.ply only when open3d is available (Thor doesn't have it)."""
    import cv2

    os.makedirs(out_dir, exist_ok=True)
    pts, rgb = depth_to_points(depth_m, K, color)
    cv2.imwrite(os.path.join(out_dir, "color.png"), color)
    np.save(os.path.join(out_dir, "depth.npy"), depth_m)
    with open(os.path.join(out_dir, "intrinsics.json"), "w") as f:
        json.dump({**intr_d, "K": K.tolist(), "t_unix": time.time()}, f, indent=2)
    try:
        import open3d as o3d
        o3d.io.write_point_cloud(os.path.join(out_dir, "scene.ply"), to_o3d(pts, rgb))
    except ImportError:
        print("[capture] no open3d, so scene.ply was skipped (depth.npy is enough to rebuild it)")
    return pts


def load_capture(cap_dir: str):
    """Read a saved capture back → (color_bgr, depth_m, K)."""
    import cv2

    color = cv2.imread(os.path.join(cap_dir, "color.png"))
    depth_m = np.load(os.path.join(cap_dir, "depth.npy"))
    with open(os.path.join(cap_dir, "intrinsics.json")) as f:
        K = np.array(json.load(f)["K"])
    return color, depth_m, K


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=f"grasp/runs/cap_{time.strftime('%Y%m%d_%H%M%S')}")
    ap.add_argument("--serial", default=CHEST_SERIAL)
    ap.add_argument("--view", action="store_true", help="show the result in the Open3D viewer")
    ap.add_argument("--warmup", type=int, default=30, help="frames dropped while auto-exposure settles")
    ap.add_argument("--temporal", type=int, default=10, help="frames accumulated by the temporal filter")
    ap.add_argument("--preset", default=None, help="D435 visual preset, e.g. 'High Accuracy' or 'Hand' (2026-11-21: on the bare aluminium bar "
                    "the default gives floor depth on half the top face; 'Hand' keeps 22%% of it at the right height)")
    ap.add_argument("--no-hole-fill", action="store_true", help="skip the hole-filling filter (keep dropped pixels empty)")
    a = ap.parse_args()
    color, depth_m, K, intr_d = capture(a.serial, warmup=a.warmup, temporal_frames=a.temporal,
                                        preset=a.preset, hole_fill=not a.no_hole_fill)
    pts = save_capture(a.out, color, depth_m, K, intr_d)
    print(f"saved {a.out}  points={len(pts)}  valid_depth={np.mean(depth_m > 0):.1%}")
    if a.view:
        import open3d as o3d
        o3d.visualization.draw_geometries([o3d.io.read_point_cloud(os.path.join(a.out, "scene.ply"))])


if __name__ == "__main__":
    main()
