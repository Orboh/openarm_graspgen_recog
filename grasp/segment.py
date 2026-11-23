#!/usr/bin/env python3
"""Step 2: cut out only the object from the scene point cloud (RANSAC plane removal → DBSCAN → largest cluster).

No recognition (YOLO etc.) is used. Assumes one object on the table.

Usage (works offline on a saved capture):
    python grasp/segment.py grasp/runs/cap_xxx            # → writes object.ply next to it
    python grasp/segment.py grasp/runs/cap_xxx --view
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from ultralytics import YOLOE
import numpy as np


@dataclass
class SegParams:
    voxel: float = 0.004          # downsampling [m]
    zmax: float = 1.2             # drop points farther than this from the camera [m]
    plane_dist: float = 0.008     # RANSAC inlier distance [m]. Raise if the table is uneven or depth is noisy
    plane_iters: int = 2000
    above_plane_min: float = 0.010  # keep only points at least this far above the plane (sensor noise right on the plane)
    above_plane_max: float = 0.40   # drop points too high above the plane (arm, torso)
    dbscan_eps: float = 0.015
    dbscan_min_pts: int = 30
    min_cluster_pts: int = 150
    sor_nb: int = 20              # statistical outlier removal
    sor_std: float = 2.0


def segment_largest_object(pts: np.ndarray, p: SegParams = SegParams(), return_debug: bool = False):
    """pts (N,3) camera frame → object point cloud (M,3). Raises RuntimeError on failure.

    debug: plane model (a,b,c,d; normal oriented toward the camera side), cluster sizes.
    """
    import open3d as o3d
    pts = pts[np.isfinite(pts).all(1) & (pts[:, 2] > 0) & (pts[:, 2] < p.zmax)]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64)))
    pc = pc.voxel_down_sample(p.voxel)
    if len(pc.points) < 500:
        raise RuntimeError(f"too few points after voxel ({len(pc.points)})")

    plane, inl = pc.segment_plane(p.plane_dist, 3, p.plane_iters)
    a, b, c, d = plane
    n = np.array([a, b, c])
    # Orient the normal toward the camera (origin), so "above the table" = positive distance
    if d < 0:
        n, d = -n, -d
    q = np.asarray(pc.points)
    h = q @ n + d  # signed height above the plane
    keep = (h > p.above_plane_min) & (h < p.above_plane_max)
    obj_pc = pc.select_by_index(np.where(keep)[0])
    if len(obj_pc.points) < p.min_cluster_pts:
        raise RuntimeError(f"nothing above plane ({len(obj_pc.points)} pts)")

    labels = np.asarray(obj_pc.cluster_dbscan(p.dbscan_eps, p.dbscan_min_pts))
    if labels.max() < 0:
        raise RuntimeError("DBSCAN found no cluster")
    sizes = np.bincount(labels[labels >= 0])
    best = int(np.argmax(sizes))
    if sizes[best] < p.min_cluster_pts:
        raise RuntimeError(f"largest cluster too small ({sizes[best]})")
    cl = obj_pc.select_by_index(np.where(labels == best)[0])
    cl, _ = cl.remove_statistical_outlier(p.sor_nb, p.sor_std)
    out = np.asarray(cl.points, dtype=np.float32)
    if return_debug:
        return out, {"plane": [*n.tolist(), float(d)], "plane_inliers": len(inl),
                     "cluster_sizes": sorted(sizes.tolist(), reverse=True)[:5]}
    return out

def segment_object_frames(cap_dir: str, p: SegParams = SegParams(), return_debug: bool = False):
    model = YOLOE("yoloe-26l-seg-pf.pt")
    color_path = os.path.join(cap_dir, "color.png")
    segmented_objects = model.predict(source=color_path, save=True, save_txt=True, conf=0.25, device="cuda", project=cap_dir, name="segmentation")
    r = segmented_objects[0]
    print(f"masks found: {0 if r.masks is None else len(r.masks.data)}")
    return segmented_objects

def main():
    import open3d as o3d

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir", help="output directory of capture.py (needs scene.ply)")
    ap.add_argument("--view", action="store_true")
    ap.add_argument("--plane-dist", type=float, default=SegParams.plane_dist)
    ap.add_argument("--eps", type=float, default=SegParams.dbscan_eps)
    a = ap.parse_args()
    scene = o3d.io.read_point_cloud(os.path.join(a.cap_dir, "scene.ply"))
    p = SegParams(plane_dist=a.plane_dist, dbscan_eps=a.eps)
    obj, dbg = segment_largest_object(np.asarray(scene.points), p, return_debug=True)
    obj_seg=segment_object_frames(a.cap_dir, p, return_debug=True)
    out = os.path.join(a.cap_dir, "object.ply")
    o3d.io.write_point_cloud(out, o3d.geometry.PointCloud(o3d.utility.Vector3dVector(obj.astype(np.float64))))
    ext = obj.max(0) - obj.min(0)
    print(f"object pts={len(obj)} extent[m]={np.round(ext, 3)} centroid={np.round(obj.mean(0), 3)} {dbg}")
    print(f"saved {out}")
    if a.view:
        o_pc = o3d.io.read_point_cloud(out)
        o_pc.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([scene, o_pc])


if __name__ == "__main__":
    main()
