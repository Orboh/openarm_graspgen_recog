#!/usr/bin/env python3
"""Chest-camera height vs grasp success: offline study. **Does not move the arm or the camera.**

2026-11-22 (Sota: カメラの高さによって把持の成功率がどれくらい変わるのか、その理由を調査). Inputs are records only:
  grasp/runs/study_camera_height/trials.csv   one row per real grasp attempt (ENPIRE result.json + manual list)
  grasp/runs/study_camera_height/caps/<cap>/   color / depth / intrinsics / object.npy / mask.png / plan.json
  the four chest-camera calibrations E0..E3 (T_base_cam*.json)
Outputs (grasp/runs/study_camera_height/out/):
  rates.csv / rates.md      success rate by camera height x object x arm x tilt band (Wilson 95 %)
  geometry.csv              per camera pose (E0..E3 + hypothetical heights/pitches): floor footprint, overlap with
                            each arm's reach, where the reach band sits in the image, px per cm, range, incidence
  reach_<arm>.npz           reach maps (fraction of headings with an IK grasp per floor cell)
  fov_reach_<pose>.png      footprint vs reach map
  percap.csv                per trial capture: floor flatness / holes in the reach zone, object points, object
                            height/extent, mask touching the image border, range, expected D435i depth noise
Reuses: grasp.plan (T_L7_G, CFG, se3), ik.openarm_left_ik / openarm_right_ik, grasp.capture.depth_to_points.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/camera_height_study.py [--skip-reach]
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
STUDY = os.path.join(REPO, "grasp", "runs", "study_camera_height")
OUT = os.path.join(STUDY, "out")
FLOOR = json.load(open(os.path.join(REPO, "grasp", "config", "floor_plane.json")))
N_F, D_F = np.array(FLOOR["n"], float), float(FLOOR["d"])  # height above floor = n.p + d (base frame)
EPOCHS = {
    "E0": os.path.expanduser("~/work/openarm-orboh/grasp/config/T_base_cam.bak-20261120-5frames.json"),
    "E1": os.path.join(REPO, "grasp", "config", "T_base_cam.bak-20261121-before-camera-move.json"),
    "E2": os.path.join(REPO, "grasp", "config", "T_base_cam.bak-20261121-before-camera-raise.json"),
    "E3": os.path.join(REPO, "grasp", "config", "T_base_cam.json"),
}
W, H = 640, 480
K = np.array([[607.36, 0, 318.03], [0, 607.42, 243.84], [0, 0, 1.0]])  # chest D435i color 640x480 (all captures)
# D435i stereo: baseline 50 mm, depth focal ~ 386 px at 640x480 (848 native scaled), subpixel 0.08 (Intel datasheet)
D435_BASELINE, D435_FDEPTH, D435_SUBPIX = 0.050, 386.0, 0.08


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def height(P):
    return P @ N_F + D_F


def cam_pose(T):
    t = T[:3, 3]; z = T[:3, 2]
    h = float(height(t[None])[0]); pitch = math.degrees(math.asin(-z[2])); yaw = math.degrees(math.atan2(z[1], z[0]))
    return h, pitch, yaw


def pix_to_floor(T, u, v, h_obj=0.0):
    """Rays through pixels → points on the plane h_obj above the floor (base frame); NaN where the ray misses."""
    u = np.atleast_1d(u).astype(float); v = np.atleast_1d(v).astype(float)
    r = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)], -1) @ T[:3, :3].T
    o = T[:3, 3]
    t = (h_obj - (o @ N_F + D_F)) / (r @ N_F)
    P = o + t[:, None] * r
    P[t <= 0] = np.nan
    return P


def floor_to_pix(T, P):
    Ti = np.linalg.inv(T); c = P @ Ti[:3, :3].T + Ti[:3, 3]
    uv = np.stack([K[0, 0] * c[:, 0] / c[:, 2] + K[0, 2], K[1, 1] * c[:, 1] / c[:, 2] + K[1, 2]], -1)
    uv[c[:, 2] <= 0] = np.nan
    return uv, c[:, 2]


def hypo_T(T_ref, h, pitch_deg):
    """Camera at height h above the floor, pitched down pitch_deg, same xy and yaw as T_ref (no roll)."""
    _, _, yaw = cam_pose(T_ref)
    y = math.radians(yaw); p = math.radians(pitch_deg)
    z = np.array([math.cos(p) * math.cos(y), math.cos(p) * math.sin(y), -math.sin(p)])
    x = np.cross(z, [0, 0, 1.0]); x /= np.linalg.norm(x)  # image x to the right
    yv = np.cross(z, x)
    T = np.eye(4); T[:3, :3] = np.c_[x, yv, z]
    t = T_ref[:3, 3].copy(); t[2] += h - (t @ N_F + D_F)
    T[:3, 3] = t
    return T


# ---------------------------------------------------------------- reach maps
def reach_map(arm, xs, ys, tip_h=0.02):
    os.environ["GRASP_ARM"] = arm
    import importlib
    import grasp.plan as plan
    importlib.reload(plan)
    from ik.openarm_right_ik import OpenArmRightIK
    from ik.openarm_left_ik import OpenArmLeftIK
    ik = OpenArmRightIK() if arm == "right" else OpenArmLeftIK()
    T_L7_G, tipz, se3 = plan.T_L7_G, plan.CFG["flange_to_tip_m"], plan.se3
    home = [53.9, 0, -4, 118.7, -1, 0.2, -33] if arm == "left" else [-53.9, 0, 4, 118.7, 1, 0.2, 33]
    rng = np.random.default_rng(0)
    seeds = [np.deg2rad(home), np.deg2rad([0, 0, 0, 60, 0, 0, 0])] + [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(1)]
    headings = np.arange(0, 180, 45)
    tilts = [(0, 0), (30, 0), (30, 180), (30, 90), (30, 270)]

    def frame(hd, tilt, tdir):
        h = np.deg2rad(hd); xg = np.array([math.cos(h), math.sin(h), 0])
        td = np.deg2rad(tdir); z = math.cos(math.radians(tilt)) * np.array([0, 0, -1.0]) + math.sin(math.radians(tilt)) * np.array([math.cos(td), math.sin(td), 0])
        xg = xg - xg @ z * z; xg /= np.linalg.norm(xg)
        return np.c_[xg, np.cross(z, xg), z]

    def solve(T, prev=None):
        for sd in ([prev] if prev is not None else []) + seeds:
            q, ok, _, _ = ik.solve(se3(T @ np.linalg.inv(T_L7_G)), sd, position_only=False, max_iter=120)
            if ok and ik.within_limits(q):
                return q
        return None

    M = np.zeros((len(xs), len(ys)))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            p = np.array([x, y, 0.0]); p[2] = (tip_h - D_F - N_F[:2] @ p[:2]) / N_F[2]
            okh = 0
            for hd in headings:
                good = False
                for tilt, tdir in tilts:
                    for flip in (1, -1):
                        R = frame(hd, tilt, tdir) @ np.diag([flip, flip, 1])
                        Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = p - tipz * R[:, 2]
                        q = solve(Tg)
                        if q is None:
                            continue
                        Tp = Tg.copy(); Tp[:3, 3] -= 0.08 * R[:, 2]
                        Tl = Tg.copy(); Tl[:3, 3] += [0, 0, 0.10]
                        if solve(Tp, q) is not None and solve(Tl, q) is not None:
                            good = True; break
                    if good:
                        break
                okh += good
            M[i, j] = okh / len(headings)
        print(f"[reach] {arm} x={x:.2f} done", flush=True)
    return M


# ---------------------------------------------------------------- geometry per camera pose
def geometry(name, T, reach, xs, ys):
    h, pitch, yaw = cam_pose(T)
    corners = pix_to_floor(T, [0, W - 1, W - 1, 0], [0, 0, H - 1, H - 1])
    X, Y = np.meshgrid(xs, ys, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), np.zeros(X.size)], -1)
    P[:, 2] = (0.0 - D_F - P[:, :2] @ N_F[:2]) / N_F[2]
    uv, rng_ = floor_to_pix(T, P)
    m = 20  # px margin from the image edge
    inview = (uv[:, 0] >= m) & (uv[:, 0] <= W - 1 - m) & (uv[:, 1] >= m) & (uv[:, 1] <= H - 1 - m)
    out = {"pose": name, "height_m": round(h, 3), "pitch_deg": round(pitch, 1), "yaw_deg": round(yaw, 1),
           "img_top_x": round(float(np.nanmin(corners[:2, 0])), 3), "img_bottom_x": round(float(np.nanmax(corners[2:, 0])), 3)
           if np.isfinite(corners[2:, 0]).all() else float("inf")}
    cell = (xs[1] - xs[0]) * (ys[1] - ys[0])
    for arm, M in reach.items():
        r = M.ravel() > 0
        rr = r & inview
        out[f"{arm}_reach_m2"] = round(r.sum() * cell, 4)
        out[f"{arm}_reach_inview_frac"] = round(rr.sum() / max(1, r.sum()), 3)
        # orientation-weighted: headings available x visible
        out[f"{arm}_dexterous_inview_frac"] = round(float((M.ravel() * inview).sum() / max(1e-9, M.ravel().sum())), 3)
        if rr.any():
            v = uv[rr, 1]
            out[f"{arm}_reach_img_rows"] = f"{int(np.min(v))}-{int(np.max(v))}"
            # px per cm on the floor at the reach cells (finite difference 1 cm in x and y)
            d = np.array([0.01, 0, 0]); uvx, _ = floor_to_pix(T, P[rr] + d); uvy, _ = floor_to_pix(T, P[rr] + d[[1, 0, 2]])
            ppc = np.sqrt(np.linalg.norm(uvx - uv[rr], axis=1) * np.linalg.norm(uvy - uv[rr], axis=1))
            out[f"{arm}_px_per_cm"] = round(float(np.median(ppc)), 1)
            out[f"{arm}_range_m"] = round(float(np.median(rng_[rr])), 3)
            ray = P[rr] - T[:3, 3]; ray /= np.linalg.norm(ray, axis=1)[:, None]
            out[f"{arm}_incidence_deg"] = round(float(np.median(np.degrees(np.arcsin(np.clip(-ray @ N_F, -1, 1))))), 1)
            z = rng_[rr]
            out[f"{arm}_d435_rms_mm"] = round(float(np.median(z * z * D435_SUBPIX / (D435_FDEPTH * D435_BASELINE)) * 1000), 2)
        else:
            out[f"{arm}_reach_img_rows"] = "-"
    return out, corners, uv, inview


# ---------------------------------------------------------------- point-cloud quality per capture
def percap(cap, T, reach_left):
    d = os.path.join(STUDY, "caps", cap)
    dep = np.load(os.path.join(d, "depth.npy")).astype(np.float32)
    v, u = np.mgrid[0:H, 0:W]
    z = dep
    Pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], -1).reshape(-1, 3)
    Pb = Pc @ T[:3, :3].T + T[:3, 3]
    # reach-zone pixels: rays hitting the floor inside the left arm's reach box x .28-.52, y -.12..+.24
    Fl = pix_to_floor(T, u.ravel(), v.ravel())
    inz = (Fl[:, 0] > 0.28) & (Fl[:, 0] < 0.52) & (Fl[:, 1] > -0.12) & (Fl[:, 1] < 0.24)
    res = {"cap": cap, "reach_px": int(inz.sum())}
    if inz.sum() < 200:
        return res
    valid = inz & (z.ravel() > 0.1)
    res["holes_frac"] = round(1 - valid.sum() / inz.sum(), 4)
    hgt = height(Pb[valid])
    fl = np.abs(hgt) < 0.015  # floor-ish points (sack / carpet)
    if fl.sum() > 100:
        # local plane per capture (removes the calibration offset) → residual std
        A = np.c_[Pb[valid][fl][:, :2], np.ones(fl.sum())]; c, *_ = np.linalg.lstsq(A, Pb[valid][fl][:, 2], rcond=None)
        r = Pb[valid][fl][:, 2] - A @ c
        r = r[np.abs(r) < 0.01]
        res["floor_std_mm"] = round(float(np.std(r)) * 1000, 2)
        res["floor_offset_mm"] = round(float(np.median(hgt[fl])) * 1000, 1)  # vs the fixed floor plane (calibration)
    # calibration consistency: the floor seen through this epoch's T_base_cam should be level in the base frame
    # (the cart floor is level). Its tilt is (camera rotation error + base tilt); at the object's range it moves the
    # object by about range * tan(tilt) (mostly along the viewing direction / height)
    allv = (z.ravel() > 0.2) & (z.ravel() < 2.0)
    Pa = Pb[allv]; ha = height(Pa)
    for _ in range(3):  # robust plane over the floor-ish points of the whole image
        sel = np.abs(ha) < 0.03
        if sel.sum() < 500:
            break
        A = np.c_[Pa[sel][:, :2], np.ones(sel.sum())]; c, *_ = np.linalg.lstsq(A, Pa[sel][:, 2], rcond=None)
        nrm = np.array([-c[0], -c[1], 1.0]); nrm /= np.linalg.norm(nrm)
        ha = (Pa - np.array([0, 0, c[2]])) @ nrm
    if sel.sum() >= 500:
        res["floor_tilt_deg"] = round(float(np.degrees(np.arccos(nrm[2]))), 2)
        res["floor_n"] = [round(float(v), 5) for v in nrm]
    res["range_m"] = round(float(np.median(z.ravel()[valid])), 3)
    if "floor_tilt_deg" in res:
        res["pos_err_from_tilt_mm"] = round(res["range_m"] * math.tan(math.radians(res["floor_tilt_deg"])) * 1000, 1)
    res["d435_rms_mm"] = round(float(res["range_m"] ** 2 * D435_SUBPIX / (D435_FDEPTH * D435_BASELINE) * 1000), 2)
    op = os.path.join(d, "object.npy")
    if os.path.exists(op):
        O = np.load(op) @ T[:3, :3].T + T[:3, 3]
        ho = height(O)
        res["obj_pts"] = int(len(O))
        res["obj_top_mm"] = round(float(np.percentile(ho, 90)) * 1000, 1)
        e = O[:, :2] - O[:, :2].mean(0); ev = np.linalg.eigh(np.cov(e.T))[1]
        res["obj_len_cm"] = round(float(np.ptp(e @ ev[:, -1])) * 100, 1)
        res["obj_wid_cm"] = round(float(np.ptp(e @ ev[:, 0])) * 100, 1)
        uvo, _ = floor_to_pix(T, O)
        res["obj_row_min"] = int(np.nanmin(uvo[:, 1]))
        # distance of the object to the nearest image border [px] (E0/E1 near edge = image bottom, E2/E3 = image top)
        mg = np.nanmin(np.minimum.reduce([uvo[:, 0], W - 1 - uvo[:, 0], uvo[:, 1], H - 1 - uvo[:, 1]]))
        res["obj_edge_margin_px"] = int(mg)
        res["obj_at_edge"] = int(mg < 8)
        res["obj_x_m"] = round(float(O[:, 0].mean()), 3)
        res["obj_y_m"] = round(float(O[:, 1].mean()), 3)
    mp = os.path.join(d, "mask.png")
    if os.path.exists(mp):
        import cv2
        m = cv2.imread(mp, 0) > 127
        if m.any():
            b = np.zeros_like(m); b[:3] = b[-3:] = True; b[:, :3] = b[:, -3:] = True
            res["mask_touches_border"] = int((m & b).any())
            res["mask_px"] = int(m.sum())
    return res


# failure cause per miss, from the two Vault reports and this session (11/20 report §8/§3, 11/21 report §3, 11/22 log)
CAUSE_BY_ITER = {
    "iter_016": "camera calibration", "iter_017": "camera calibration",          # E0 hand-eye 2.4 deg off (11/20 §8)
    "iter_021": "low object / floor", "iter_027": "low object / floor",          # jaw corner on the floor
    **{f"iter_0{n}": "arm/TCP lateral error" for n in (31, 33, 34, 35, 36, 37)},  # 3-7 cm across the jaws (11/21 §3.1)
    **{f"iter_0{n}": "grip slip (friction / end grasp)" for n in (40, 42, 43, 44, 45)},
    "iter_049": "pile: other object in the jaws", "iter_051": "pile: other object in the jaws",
    "iter_052": "arm/TCP lateral error",                                          # bar pushed aside, 11 mm margin
    "iter_054": "grasp too shallow (code change)", "iter_055": "wrong target",
}
CAUSE_GROUP = {"camera calibration": "camera", "view: cut at image edge / mis-measured": "camera",
               "low object / floor": "depth/height", "depth short (jaws above the bar)": "depth/height",
               "arm/TCP lateral error": "arm/TCP", "grip slip (friction / end grasp)": "gripper",
               "pile: other object in the jaws": "pile/target", "wrong target": "pile/target",
               "grasp too shallow (code change)": "code", "reach (far thin bar, steep grasp)": "reach",
               "human hand in the way": "other", "unknown": "other"}


def cause_of(r):
    if r["result"] in ("held", "lift_judged_empty", "aborted"):
        return ""
    if r["source"] in CAUSE_BY_ITER:
        return CAUSE_BY_ITER[r["source"]]
    c = r["cause"].lower()
    if "report-11/20" in r["source"]:
        return "camera calibration"
    if "out of chest view" in c or "mis-measured" in c:
        return "view: cut at image edge / mis-measured"
    if "above bar" in c:
        return "depth short (jaws above the bar)"
    if "beside the bar" in c:
        return "arm/TCP lateral error"
    if "far" in c and "steep" in c:
        return "reach (far thin bar, steep grasp)"
    if "hand" in c:
        return "human hand in the way"
    return "unknown"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-reach", action="store_true", help="reuse out/reach_*.npz")
    a = ap.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    Ts = {e: np.array(json.load(open(p))["T_base_cam"]) for e, p in EPOCHS.items()}

    # 1. success rates
    rows = list(csv.DictReader(open(os.path.join(STUDY, "trials.csv"))))
    for r in rows:
        r["ok"] = r["result"] in ("held", "lift_judged_empty")  # lifted (by eye) counts as a grasp success
        r["hgrp"] = "0.47m" if r["epoch"] in ("E0", "E1", "E2") else "0.69m"
        t = r["tilt"]; r["tiltb"] = "?" if t == "" else ("0-19" if float(t) < 20 else "20-34" if float(t) < 35 else "35+")
    grasp_rows = [r for r in rows if r["result"] != "aborted"]
    keys = [("hgrp",), ("epoch",), ("hgrp", "object"), ("hgrp", "arm"), ("hgrp", "method"), ("hgrp", "tiltb"),
            ("hgrp", "object", "arm"), ("hgrp", "object", "method")]
    rates = []
    for key in keys:
        g = defaultdict(list)
        for r in grasp_rows:
            g[tuple(r[k] for k in key)].append(r["ok"])
        for kv, v in sorted(g.items()):
            k, n = sum(v), len(v); lo, hi = wilson(k, n)
            rates.append({"by": "+".join(key), "group": " / ".join(kv), "n": n, "held": k, "rate": round(k / n, 2),
                          "ci95": f"{lo:.2f}-{hi:.2f}"})
    causes = defaultdict(lambda: defaultdict(int))
    for r in grasp_rows:
        c = cause_of(r); r["cause_cat"] = c
        if c:
            causes[r["hgrp"]][CAUSE_GROUP[c]] += 1; causes[r["hgrp"] + " detail"][c] += 1
    with open(os.path.join(OUT, "causes.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["height", "cause", "n"])
        for hg, d in sorted(causes.items()):
            for c, k in sorted(d.items(), key=lambda kv: -kv[1]):
                w.writerow([hg, c, k])
    for hg in ("0.47m", "0.69m"):
        print(f"[causes] {hg}: " + ", ".join(f"{c} {k}" for c, k in sorted(causes[hg].items(), key=lambda kv: -kv[1])))
    with open(os.path.join(STUDY, "trials_with_cause.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(grasp_rows[0].keys())); w.writeheader(); [w.writerow(x) for x in grasp_rows]
    with open(os.path.join(OUT, "rates.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rates[0])); w.writeheader(); [w.writerow(x) for x in rates]
    ab = sum(r["result"] == "aborted" for r in rows)
    print(f"[rates] {len(grasp_rows)} grasp attempts (+{ab} aborted before closing)")
    for x in rates:
        print(f"  {x['by']:22s} {x['group']:40s} {x['held']:3d}/{x['n']:<3d} {x['rate']:.2f}  [{x['ci95']}]")

    # 2. reach maps + geometry
    xs = np.round(np.arange(0.18, 0.66, 0.03), 3); ys = np.round(np.arange(-0.39, 0.42, 0.03), 3)
    reach = {}
    for arm in ("left", "right"):
        fp = os.path.join(OUT, f"reach_{arm}.npz")
        if a.skip_reach and os.path.exists(fp):
            reach[arm] = np.load(fp)["M"]
        else:
            reach[arm] = reach_map(arm, xs, ys); np.savez(fp, M=reach[arm], xs=xs, ys=ys)
        M = reach[arm]; r = M > 0
        print(f"[reach] {arm}: {r.sum()} cells; x {xs[r.any(1)].min():.2f}-{xs[r.any(1)].max():.2f}, "
              f"y {ys[r.any(0)].min():.2f}-{ys[r.any(0)].max():.2f}")
    poses = dict(Ts)
    for hh in (0.47, 0.55, 0.69, 0.80):
        for pp in (45, 55, 65, 71):
            poses[f"h{hh:.2f}_p{pp}"] = hypo_T(Ts["E3"], hh, pp)
    geo = []
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for name, T in poses.items():
        g, corners, uv, inview = geometry(name, T, reach, xs, ys); geo.append(g)
        if name in Ts or name in ("h0.69_p65", "h0.80_p71", "h0.55_p55"):
            fig, ax = plt.subplots(figsize=(5.5, 5))
            for arm, cm in (("left", "Blues"), ("right", "Oranges")):
                M = np.ma.masked_equal(reach[arm], 0)
                ax.pcolormesh(ys, xs, M, cmap=cm, vmin=0, vmax=1, alpha=0.55, shading="nearest")
            # footprint: image border sampled
            uu = np.r_[np.linspace(0, W - 1, 40), np.full(30, W - 1), np.linspace(W - 1, 0, 40), np.zeros(30)]
            vv = np.r_[np.zeros(40), np.linspace(0, H - 1, 30), np.full(40, H - 1), np.linspace(H - 1, 0, 30)]
            Fp = pix_to_floor(T, uu, vv)
            ax.plot(Fp[:, 1], Fp[:, 0], "g-", lw=2, label="camera view (floor)")
            ax.set_xlim(0.40, -0.40); ax.set_ylim(0.15, 1.0); ax.set_aspect("equal")
            ax.set_xlabel("y [m] (left +)"); ax.set_ylabel("x [m] (forward)")
            ax.set_title(f"{name}: h {g['height_m']:.2f} m, pitch {g['pitch_deg']:.0f} deg\n"
                         f"left reach in view {g['left_reach_inview_frac']*100:.0f}%, right {g['right_reach_inview_frac']*100:.0f}%")
            ax.plot([], [], "s", color="tab:blue", alpha=.6, label="left reach"); ax.plot([], [], "s", color="tab:orange", alpha=.6, label="right reach")
            ax.legend(loc="upper right", fontsize=7)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, f"fov_reach_{name}.png"), dpi=110); plt.close(fig)
    with open(os.path.join(OUT, "geometry.csv"), "w", newline="") as f:
        cols = sorted({k for g in geo for k in g}, key=lambda k: (k != "pose", k))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(g) for g in geo]
    print("[geometry]")
    for g in geo[:4]:
        print("  ", g)

    # 3. per-capture point-cloud quality
    pc = []
    for r in rows:
        if r["cap"] and os.path.exists(os.path.join(STUDY, "caps", r["cap"], "depth.npy")):
            x = percap(r["cap"], Ts[r["epoch"]], reach["left"]); x.update(epoch=r["epoch"], hgrp=r["hgrp"], result=r["result"],
                                                                         object=r["object"], arm=r["arm"]); pc.append(x)
    # the base itself may not be level (all epochs show ~2.5 deg): compare each epoch's floor normal with the best
    # calibrated epoch (E1, 42 marker detections, 2.2 mm) -> the angle is the camera-rotation disagreement
    nref = np.median([x["floor_n"] for x in pc if x["epoch"] == "E1" and "floor_n" in x], axis=0); nref /= np.linalg.norm(nref)
    for x in pc:
        if "floor_n" in x:
            x["rot_vs_E1_deg"] = round(float(np.degrees(np.arccos(np.clip(np.dot(x["floor_n"], nref), -1, 1)))), 2)
            x["pos_err_rot_mm"] = round(x["range_m"] * math.tan(math.radians(x["rot_vs_E1_deg"])) * 1000, 1)
    for ep in ("E0", "E1", "E2", "E3"):
        ns = [x["floor_n"] for x in pc if x["epoch"] == ep and "floor_n" in x]
        if ns:
            nm = np.median(ns, axis=0); nm /= np.linalg.norm(nm)
            ang = float(np.degrees(np.arccos(np.clip(nm @ nref, -1, 1))))
            rg = np.median([x["range_m"] for x in pc if x["epoch"] == ep and "range_m" in x])
            sp = np.median([x["rot_vs_E1_deg"] for x in pc if x["epoch"] == ep and "rot_vs_E1_deg" in x])
            print(f"[calib] {ep}: floor normal vs E1 {ang:.2f} deg (per-capture median {sp:.2f}); at range {rg:.2f} m -> ~{rg*math.tan(math.radians(ang))*1000:.0f} mm")
    with open(os.path.join(OUT, "percap.csv"), "w", newline="") as f:
        cols = sorted({k for g in pc for k in g}, key=lambda k: (k != "cap", k))
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(g) for g in pc]
    print(f"[percap] {len(pc)} captures")
    for e in ("0.47m", "0.69m"):
        s = [x for x in pc if x["hgrp"] == e]
        def med(k):
            v = [x[k] for x in s if k in x]
            return (round(float(np.median(v)), 3), len(v)) if v else None
        v = [x["floor_offset_mm"] for x in s if "floor_offset_mm" in x]
        print(f"  {e}: floor offset vs the fixed floor plane: median {np.median(v):.1f} mm, spread (std) {np.std(v):.1f} mm, range {np.min(v):.1f}..{np.max(v):.1f}")
        print(f"  {e}: object at the image edge {sum(x.get('obj_at_edge', 0) for x in s)}/{sum('obj_at_edge' in x for x in s)}; "
              f"success with object at edge {sum(x['result'] in ('held','lift_judged_empty') for x in s if x.get('obj_at_edge'))}/{sum(1 for x in s if x.get('obj_at_edge'))}, "
              f"not at edge {sum(x['result'] in ('held','lift_judged_empty') for x in s if 'obj_at_edge' in x and not x['obj_at_edge'])}/{sum(1 for x in s if 'obj_at_edge' in x and not x['obj_at_edge'])}")
        print(f"  {e}: floor tilt in base {med('floor_tilt_deg')} deg -> position error at the object ~{med('pos_err_from_tilt_mm')} mm")
        for ep in ("E0", "E1", "E2", "E3"):
            t = [x["floor_tilt_deg"] for x in s if x["epoch"] == ep and "floor_tilt_deg" in x]
            pe = [x["pos_err_from_tilt_mm"] for x in s if x["epoch"] == ep and "pos_err_from_tilt_mm" in x]
            if t:
                print(f"     {ep}: floor tilt median {np.median(t):.2f} deg (n {len(t)}), position error ~{np.median(pe):.0f} mm")
        print(f"  {e}: floor_std_mm {med('floor_std_mm')} holes {med('holes_frac')} range {med('range_m')} d435_rms {med('d435_rms_mm')} "
              f"obj_pts {med('obj_pts')} border {med('mask_touches_border')} floor_offset_mm {med('floor_offset_mm')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
