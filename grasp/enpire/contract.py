"""ENPIRE-style frozen verifier for the OpenArm top-down grasp task.

READ-ONLY for the auto-research loop (ENPIRE autoresearch contract: the policy side may change
`policy.py` params only; never the verifier, thresholds, trial budget or artifact schema).
Changing anything here invalidates comparison with earlier iterations -> bump VERIFIER_VERSION.

Success is a conjunction of predicates. Each predicate reports PASS / FAIL / SKIPPED (not
applicable at this level, e.g. `reachable` without a hand-eye calibration, `lifted` without motion).
`level` says how far the trial got:
    perception : capture -> segment -> grasp, judged on the image/point cloud only
    plan       : + transformed to the arm base and IK-reachable (needs T_base_cam)
    real       : + executed on the arm and the object was lifted (needs motion, not implemented yet)
"""
from __future__ import annotations

import json
import os

import numpy as np

VERIFIER_VERSION = "2026-11-22.5"  # .3: top_down 30 -> 45 deg (operator decision 2026-11-20 19:55: nothing reachable+visible within 30 deg)
# .4: max object extent 0.25 -> 0.35 m (operator decision 2026-11-21 22:55, Sota: the aluminium bars / square pipe in the
#     mock pile are 26 cm; a planned and dry-run-checked grasp was refused only on length)
# .5: top_down 45 -> 60 deg, min grasp score 0.5 -> 0 (Sota 2026-11-22: loosen the strict gates; match plan.py defaults)

# Task constants (Kota's doc: wood block + PET bottle, grasp from above, >=70% over 10 placements each)
SUCCESS_GATE = 0.70
MIN_OBJ_EXTENT_M = 0.02
MAX_OBJ_EXTENT_M = 0.35         # was 0.25 (blocks only); bars/pipes up to 35 cm (.4)
MIN_GRASP_SCORE = 0.0 
MIN_PTS_BETWEEN_JAWS = 20        # object points inside the closing volume between the open jaws
JAW_LEN_M = 0.045                # finger length along approach (config/openarm_custom_hand_sweep.json)
JAW_HALF_THICK_M = 0.019         # half finger thickness (sweep extents_open[1] / 2)
MAX_TOPDOWN_TILT_DEG = 60.0      # "from above": approach within 60 deg of straight down (was 30; see VERIFIER_VERSION)
MAX_OPENING_M = 0.110            # measured jaw tip gap, config/openarm_left_gripper.json
FINGERTIP_DEPTH_M = 0.140        # gripper base (flange) -> jaw tip
# 上の爪の寸法は東京の左フォロワーの爪。置き場の爪の設定（grasp/config/sites/）を環境変数で指したら、その寸法で判定する
if os.environ.get("GRASP_GRIPPER_JSON"):
    _g = json.load(open(os.environ["GRASP_GRIPPER_JSON"]))
    MAX_OPENING_M, FINGERTIP_DEPTH_M = float(_g["max_opening_m"]), float(_g["flange_to_tip_m"])
if os.environ.get("GRASP_SWEEP_JSON"):
    _s = json.load(open(os.environ["GRASP_SWEEP_JSON"]))
    JAW_LEN_M, JAW_HALF_THICK_M = float(_s["extents_open"][2]), float(_s["extents_open"][1]) / 2
LIFT_MIN_M = 0.05                # real level: object must leave the table by this much

PREDICATES = ["object_found", "grasp_found", "grasp_on_object", "top_down", "width_ok",
              "reachable", "lifted"]

# Failure attribution in pipeline order (first failing predicate = attributed cause)
CAUSE = {
    "object_found": "(a) segmentation: no plausible object cluster",
    "grasp_found": "(b) grasp net: no grasp above score threshold",
    "grasp_on_object": "(c) chosen grasp not on the object",
    "top_down": "(d) chosen grasp not from above",
    "width_ok": "(e) object wider than the jaw opening along the closing axis",
    "reachable": "(f) IK: grasp not reachable by the left arm",
    "lifted": "(g) execution: object not lifted",
}


def _p(status, metric=None, threshold=None, note=""):
    return {"status": status, "metric": None if metric is None else float(metric),
            "threshold": threshold, "note": note}


def evaluate(obs: dict) -> dict:
    """obs keys (all optional; missing -> predicate SKIPPED or FAIL as documented):
        obj (M,3) camera-frame object points or None, plane [nx,ny,nz,d] (normal toward camera),
        grasp 4x4 chosen grasp (camera frame, GraspGenX convention: +Z approach, +X closing) or None,
        score float, ik {"converged": bool, "pos_err_m": float, "rot_err_deg": float} or None,
        lift_m float or None.
    """
    pr = {}
    obj = obs.get("obj")
    if obj is None or len(obj) == 0:
        pr["object_found"] = _p("FAIL", note=obs.get("seg_error", "no cluster"))
    else:
        ext = float(np.max(np.ptp(obj, axis=0)))
        ok = MIN_OBJ_EXTENT_M <= ext <= MAX_OBJ_EXTENT_M
        pr["object_found"] = _p("PASS" if ok else "FAIL", ext, [MIN_OBJ_EXTENT_M, MAX_OBJ_EXTENT_M],
                                "max extent [m]")

    G, score = obs.get("grasp"), obs.get("score")
    if G is None:
        pr["grasp_found"] = _p("FAIL", note=obs.get("infer_error", "no grasp"))
    else:
        pr["grasp_found"] = _p("PASS" if score >= MIN_GRASP_SCORE else "FAIL", score, MIN_GRASP_SCORE)

    if G is not None and obj is not None and len(obj):
        G = np.asarray(G)
        q = (obj - G[:3, 3]) @ G[:3, :3]            # object points in the grasp frame (+Z approach, +X closing)
        slab = (np.abs(q[:, 1]) <= JAW_HALF_THICK_M) & (q[:, 2] >= FINGERTIP_DEPTH_M - JAW_LEN_M) \
            & (q[:, 2] <= FINGERTIP_DEPTH_M + 0.01)
        inside = slab & (np.abs(q[:, 0]) <= MAX_OPENING_M / 2)
        n_in = int(inside.sum())
        pr["grasp_on_object"] = _p("PASS" if n_in >= MIN_PTS_BETWEEN_JAWS else "FAIL", n_in, MIN_PTS_BETWEEN_JAWS,
                                   "object points between the open jaws")
        plane = obs.get("plane")
        if plane is not None:
            n = np.asarray(plane[:3], float)  # points toward the camera = "up" from the table
            tilt = float(np.degrees(np.arccos(np.clip(-G[:3, 2] @ n, -1, 1))))
            pr["top_down"] = _p("PASS" if tilt <= MAX_TOPDOWN_TILT_DEG else "FAIL", tilt,
                                MAX_TOPDOWN_TILT_DEG, "approach vs table-down [deg]")
        else:
            pr["top_down"] = _p("SKIPPED", note="no table plane")
        # width of what the jaws actually close on: object in the jaw slab, searched wider than the opening
        w = float(np.ptp(q[slab & (np.abs(q[:, 0]) <= 0.15), 0])) if (slab & (np.abs(q[:, 0]) <= 0.15)).any() else 0.0
        pr["width_ok"] = _p("PASS" if 0 < w <= MAX_OPENING_M else "FAIL", w, MAX_OPENING_M,
                            "object width in the jaw slab along closing axis [m]")
    else:
        for k in ("grasp_on_object", "top_down", "width_ok"):
            pr[k] = _p("FAIL", note="no object or no grasp")

    ik = obs.get("ik")
    if ik is None:
        pr["reachable"] = _p("SKIPPED", note=obs.get("ik_note", "no T_base_cam (hand-eye not validated)"))
    else:
        pr["reachable"] = _p("PASS" if ik["converged"] else "FAIL", ik.get("pos_err_m"), 0.005,
                             f"rot_err={ik.get('rot_err_deg', float('nan')):.1f}deg")

    lift = obs.get("lift_m")
    pr["lifted"] = (_p("SKIPPED", note="no motion at this level") if lift is None
                    else _p("PASS" if lift >= LIFT_MIN_M else "FAIL", lift, LIFT_MIN_M))

    applicable = [k for k in PREDICATES if pr[k]["status"] != "SKIPPED"]
    success = all(pr[k]["status"] == "PASS" for k in applicable)
    level = "real" if lift is not None else ("plan" if ik is not None else "perception")
    cause = "success" if success else next(CAUSE[k] for k in PREDICATES if pr[k]["status"] == "FAIL")
    return {"verifier_version": VERIFIER_VERSION, "level": level, "success": success,
            "cause": cause, "predicates": pr}
