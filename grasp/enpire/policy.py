"""Policy side of the ENPIRE loop: the ONLY knobs auto-research may change.

A "policy" here is the training-free perception -> grasp-selection pipeline
(segment.py + GraspGenX + selection rule). Params are a flat dict saved with every iteration.
"""
from __future__ import annotations

import copy
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GRASP = os.path.dirname(HERE)

DEFAULT_PARAMS = {
    # ROI on the chest camera image (cuts the robot's own hands at the bottom) and depth window
    "rows": [0, 480], "cols": [0, 640], "zmin": 0.30, "zmax": 0.70,
    # segment.SegParams
    "voxel": 0.004, "plane_dist": 0.008, "above_plane_min": 0.010, "above_plane_max": 0.04,
    "dbscan_eps": 0.015, "dbscan_min_pts": 30, "min_cluster_pts": 150,
    # GraspGenX
    "sweep_json": os.environ.get("GRASP_SWEEP_JSON", os.path.join(GRASP, "config", "openarm_custom_hand_sweep.json")),
    "gripper_name": None, "num_grasps": 200, "topk": 50, "max_points": 3500,
    # selection among top-k: prefer grasps whose approach is within this tilt of straight down
    "select_max_tilt_deg": 25.0, "select_min_score": 0.5,
    # with a hand-eye: try up to this many filtered candidates (score order) and take the first IK-reachable one
    "select_ik_tries": 1,
}


def load(path: str | None) -> dict:
    p = copy.deepcopy(DEFAULT_PARAMS)
    if path:
        with open(path) as f:
            p.update(json.load(f))
    return p


def select_grasp(grasps: np.ndarray, scores: np.ndarray, plane, p: dict):
    """Pick the best grasp: highest score among those within select_max_tilt_deg of top-down;
    fall back to the global best. Returns (index, info)."""
    if len(scores) == 0:
        return None, {"reason": "no grasps"}
    if plane is None or p["select_max_tilt_deg"] is None:
        return 0, {"reason": "best score"}
    n = np.asarray(plane[:3], float)
    tilt = np.degrees(np.arccos(np.clip(-(grasps[:, :3, 2] @ n), -1, 1)))
    ok = np.where((tilt <= p["select_max_tilt_deg"]) & (scores >= p["select_min_score"]))[0]
    if len(ok):
        i = int(ok[np.argmax(scores[ok])])
        return i, {"reason": "top-down filter", "candidates": int(len(ok)), "tilt_deg": float(tilt[i])}
    return 0, {"reason": "fallback best score (no top-down candidate)", "tilt_deg": float(tilt[0])}


# Offline auto-research proposals: one-knob perturbations around the current best (no LLM needed).
SEARCH_SPACE = {
    "plane_dist": [0.005, 0.008, 0.012],
    "dbscan_eps": [0.010, 0.015, 0.025],
    "above_plane_max": [0.04, 0.10, 0.25],
    "select_max_tilt_deg": [15.0, 25.0, 40.0],
    "zmax": [0.70, 0.90],
}


def proposals(best: dict):
    """Yield (hypothesis, params) one-knob changes from `best`, skipping no-ops."""
    for k, vals in SEARCH_SPACE.items():
        for v in vals:
            if best.get(k) == v:
                continue
            q = copy.deepcopy(best)
            q[k] = v
            yield f"changing {k}: {best.get(k)} -> {v} raises success rate", q


def grid(keys):
    vals = [SEARCH_SPACE[k] for k in keys]
    for combo in itertools.product(*vals):
        yield dict(zip(keys, combo))
