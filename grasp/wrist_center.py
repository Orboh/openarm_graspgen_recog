#!/usr/bin/env python3
"""Where is the bar between the open jaws? Measured with the LEFT wrist D405 right before closing. **No motion.**

2026-11-22 (Sota: 手首カメラでずれ直しを入れて、つかめるまでやって). With the jaws tilted 10-14 deg the bar sat 15-27 mm
toward the right jaw although the chest camera + arm model put the jaw centre on it. Right before closing, the wrist
camera sees both jaws and the bar from 12-14 cm: the bar's top stands ~7 mm above the sack (D405 depth at this range:
well under 1 mm), so it is found in depth, not colour.

    measure(depth_at_grasp, depth_ref_open) -> dict(dc_m, ...)   dc_m > 0: the bar is toward the RIGHT jaw
                                                                  (= +closing in the camera = +Xg of plan.py's grasp frame)
The jaws are found in the empty-jaw reference taken in the air with the same opening (run_grasp.py wrist_ref_open).
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = json.load(open(os.path.join(HERE, "config", "wrist_jaw_model.json")))


def _intr():
    I = MODEL["intrinsics"]
    if isinstance(I, str):
        import ast
        I = ast.literal_eval(I)
    return float(I["fx"]), float(I["ppx"])


def measure_edges(bgr, ref_open, dep=None, debug=False):
    """Bar centre from its long straight edges in the colour image (the depth version below misses the shiny bar).
    Near-vertical Hough segments between / beyond the open jaws (jaws masked with the empty-jaw reference), a line
    x = a*y + b through their midpoints, read at the jaw-tip row; minus the gap centre there. Returns dict like measure()."""
    fx, ppx = _intr()
    ref = np.asarray(ref_open, np.float32)
    jawm = (ref > 0.03) & (ref < 0.20)
    rows = np.nonzero(jawm.sum(1) > 10)[0]
    out = {"ok": False, "method": "edges"}
    if len(rows) == 0:
        out["why"] = "jaws not found in the reference"; return out
    W = ref.shape[1]; tip = None
    for r in range(int(rows.min()), int(rows.min()) + 120):  # first row where both jaw tips are seen, +6 rows
        jc = np.nonzero(jawm[r])[0]
        lft, rgt = jc[jc < W // 2 + 20], jc[jc >= W // 2 + 20]
        if len(lft) >= 3 and len(rgt) >= 3:
            tip = r + 6; break
    if tip is None:
        out["why"] = "both jaws not seen"; return out
    jc = np.nonzero(jawm[tip])[0]; lft, rgt = jc[jc < W // 2 + 20], jc[jc >= W // 2 + 20]
    if not len(lft) or not len(rgt):
        out["why"] = "both jaws not seen at the tip row"; return out
    L, R = int(lft.max()), int(rgt.min()); gc = (L + R) / 2.0
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    e = cv2.Canny(cv2.GaussianBlur(g, (5, 5), 0), 40, 120)
    e[cv2.dilate(jawm.astype(np.uint8), np.ones((11, 11), np.uint8)).astype(bool)] = 0
    e[:20] = 0; e[tip + 160:] = 0
    e[:, :max(0, L - 60)] = 0; e[:, min(W, R + 60):] = 0
    lines = cv2.HoughLinesP(e, 1, np.pi / 180, 40, minLineLength=50, maxLineGap=8)
    mids = []
    if lines is not None:
        for l in lines.reshape(-1, 4):
            x1, y1, x2, y2 = [float(v) for v in l]
            ang = abs(np.degrees(np.arctan2(x2 - x1, y2 - y1)))
            if ang < 30 or ang > 150:
                n = int(max(2, abs(y2 - y1) // 10))
                for t in np.linspace(0, 1, n):
                    mids.append((x1 + (x2 - x1) * t, y1 + (y2 - y1) * t))
    if len(mids) < 6:
        out["why"] = f"bar edges not found ({len(mids)} points)"; return out
    M = np.array(mids)
    med = np.median(M[:, 0]); M = M[np.abs(M[:, 0] - med) < 45]
    if len(M) < 6:
        out["why"] = "bar edges scattered"; return out
    A = np.c_[M[:, 1], np.ones(len(M))]
    (a, b), *_ = np.linalg.lstsq(A, M[:, 0], rcond=None)
    x_tip = a * tip + b
    z = 0.128
    if dep is not None:
        d = np.asarray(dep, np.float32)[max(0, tip - 10):tip + 30, int(x_tip) - 6:int(x_tip) + 7]
        d = d[(d > 0.05) & (d < 0.30)]
        if len(d) > 10:
            z = float(np.median(d))
    dc_px = float(x_tip - gc)
    out.update(ok=True, dc_px=dc_px, dc_m=dc_px * z / fx, z=z, tip_row=tip, gap=[L, R], n_pts=int(len(M)),
               slope=float(a))
    if not (L + 5 < x_tip < R - 5):
        out["outside_gap"] = True
    if debug:
        im = bgr.copy()
        im[jawm] = (im[jawm] * 0.5 + np.array([0, 0, 128])).astype(np.uint8)
        for x, y in M:
            cv2.circle(im, (int(x), int(y)), 2, (0, 255, 0), -1)
        cv2.line(im, (int(a * 20 + b), 20), (int(a * (tip + 150) + b), tip + 150), (0, 255, 255), 1)
        cv2.line(im, (int(gc), tip - 40), (int(gc), tip + 120), (255, 0, 0), 2)
        cv2.putText(im, f"bar {out['dc_m']*1000:+.0f} mm (+ = right jaw)", (8, 24), 0, 0.7, (0, 255, 255), 2)
        out["debug"] = im
    return out


def measure(dep, ref_open, bar_min=0.003, debug_bgr=None):
    """dep / ref_open: HxW depth [m]. Returns {"ok", "dc_m", "dc_px", "rows", "z", "why"} (+ "debug" image)."""
    fx, ppx = _intr()
    dep = np.asarray(dep, np.float32); ref = np.asarray(ref_open, np.float32)
    jaw = (ref > 0.03) & (ref < 0.20)
    jaw = cv2.morphologyEx(jaw.astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    jaw_d = cv2.dilate(jaw.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(bool)
    H, W = dep.shape
    cols = np.arange(W)
    tip_row = None
    rows = []
    for r in range(20, H - 20):
        jc = cols[jaw[r]]
        if len(jc) < 10:
            continue
        mid = W // 2 + 20
        left, right = jc[jc < mid], jc[jc >= mid]
        if len(left) < 3 or len(right) < 3:
            continue
        L, R = int(left.max()), int(right.min())
        if R - L < 30:
            continue
        if tip_row is None:
            tip_row = r
        if r > tip_row + 140:
            break
        seg = dep[r, L + 4:R - 3]; segc = cols[L + 4:R - 3]
        ok = (seg > 0.05) & (seg < 0.30) & ~jaw_d[r, L + 4:R - 3]
        if ok.sum() < 10:
            continue
        sack = np.percentile(seg[ok], 80)
        bar = ok & (seg < sack - bar_min)
        if bar.sum() < 3:
            rows.append((r, (L + R) / 2.0, None, sack)); continue
        bc = segc[bar]
        # the largest connected run of bar columns
        runs = np.split(bc, np.where(np.diff(bc) > 2)[0] + 1)
        run = max(runs, key=len)
        rows.append((r, (L + R) / 2.0, float(run.mean()), float(np.median(seg[bar]))))
    out = {"ok": False, "tip_row": tip_row, "n_rows": len(rows)}
    found = [x for x in rows if x[2] is not None]
    if tip_row is None:
        out["why"] = "jaws not found in the reference"; return out
    # bar beyond the tips (no jaws there): rows above tip_row, relative to the gap centre at the tips
    above = []
    if rows:
        gc_tip = np.median([x[1] for x in rows[:10]])
        for r in range(max(10, tip_row - 90), tip_row - 5):
            seg = dep[r]; ok = (seg > 0.05) & (seg < 0.30) & ~jaw_d[r]
            lo, hi = int(gc_tip - 120), int(gc_tip + 120)
            ok[:max(lo, 0)] = False; ok[min(hi, W):] = False
            if ok.sum() < 20:
                continue
            sack = np.percentile(seg[ok], 80); bar = ok & (seg < sack - bar_min)
            if bar.sum() < 3:
                continue
            bc = cols[bar]; runs = np.split(bc, np.where(np.diff(bc) > 2)[0] + 1); run = max(runs, key=len)
            if 4 <= len(run) <= 80:
                above.append((r, gc_tip, float(run.mean()), float(np.median(seg[bar]))))
    use = found if len(found) >= 8 else found + above
    if len(use) < 6:
        out["why"] = f"bar not found (between the jaws {len(found)} rows, beyond the tips {len(above)} rows)"; return out
    d_px = np.array([x[2] - x[1] for x in use]); z = float(np.median([x[3] for x in use]))
    dc_px = float(np.median(d_px))
    out.update(ok=True, dc_px=dc_px, z=z, dc_m=dc_px * z / fx, rows_between=len(found), rows_beyond=len(above),
               spread_px=float(np.percentile(d_px, 75) - np.percentile(d_px, 25)))
    if debug_bgr is not None:
        im = debug_bgr.copy()
        im[jaw] = (im[jaw] * 0.5 + np.array([0, 0, 128])).astype(np.uint8)
        for r, gc, bc, _ in use:
            cv2.circle(im, (int(gc), r), 1, (255, 0, 0), -1); cv2.circle(im, (int(bc), r), 1, (0, 255, 0), -1)
        cv2.putText(im, f"bar {out['dc_m']*1000:+.0f} mm (+ = right jaw)", (8, 24), 0, 0.7, (0, 255, 255), 2)
        out["debug"] = im
    return out


if __name__ == "__main__":
    import sys
    d = sys.argv[1]
    dep = np.load(os.path.join(d, "wrist_at_grasp_depth.npy")); ref = np.load(os.path.join(d, "wrist_ref_open_depth.npy"))
    col = cv2.imread(os.path.join(d, "wrist_at_grasp.jpg"))
    r = measure_edges(col, ref, dep, debug=True)
    if "debug" in r:
        cv2.imwrite(os.path.join(d, "wrist_center.jpg"), r.pop("debug"))
    print(os.path.basename(d.rstrip("/")), {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
