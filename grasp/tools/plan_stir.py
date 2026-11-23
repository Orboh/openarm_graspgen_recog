#!/usr/bin/env python3
"""Plan a "stir" stroke that loosens a pile where the pieces touch. **Does not move the arm.**

2026-11-21 (Sota chose A: 閉じた爪で山の上をかき混ぜてばらす). In the mock pile the target bar touched a second bar and a
wood block along its whole length (0.4-2 cm); no grasp or push that touches nothing exists. Instead, contact on
purpose: the closed jaws are put down into the seam between the target bar and its nearest rigid neighbour and dragged
along the seam, tips a little below the lower of the two tops. The jaws wedge the two apart (the bar rolls / slides
onto the soft side). After the stroke the pile is captured again and the usual "bar with room for the jaws" pick runs.

  1. target = object.npy, neighbours = rigid.png points (thor_snap --perceive).
  2. Bar axis; along the bar every 1 cm: the seam point (middle of the gap to the nearest rigid thing on the neighbour
     side, if one is within --seam-max) and the lower of the two local tops.
  3. Longest run of seam points the arm reaches with the jaws pointing down (tilt <= --max-tilt): stroke from the end
     farther from the robot toward the robot (a pull, so the arm does not reach further out while pushing).
  4. Waypoints: above the start (along the gripper axis), down into the seam, 5 steps along it, rise.

Writes push.json (mode "stir") in the push format, run by run_grasp.py --push (closed jaws, near-object speed, stops
when a joint falls more than --max-track-err behind: a jam in the pile). Usage:

    python grasp/tools/plan_stir.py grasp/runs/<cap> --q <home deg>
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from grasp.plan import CFG, T_L7_G, floor_plane_base, in_keepout, se3  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True, help="start (home) pose, lerobot deg")
    ap.add_argument("--seam-max", type=float, default=0.04, help="a rigid neighbour this close [m] makes a seam")
    ap.add_argument("--below", type=float, default=0.015, help="tips this far below the lower of the two tops [m]")
    ap.add_argument("--min-tip", type=float, default=0.012, help="never lower than this above the floor [m]")
    ap.add_argument("--min-len", type=float, default=0.05, help="shortest useful stroke [m]")
    ap.add_argument("--max-len", type=float, default=0.12)
    ap.add_argument("--above", type=float, default=0.06, help="approach / leave distance along the gripper axis [m]")
    ap.add_argument("--max-tilt", type=float, default=45.0)
    ap.add_argument("--across", type=float, default=0.0, help="instead of along the seam: push this far [m] across it, away "
                    "from the target bar (moves the neighbours off it)")
    a = ap.parse_args(argv)
    d = a.cap_dir
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    n_b, d_b = floor_plane_base(d, T_bc)
    hgt = lambda P: P @ n_b + d_b  # noqa: E731
    obj = np.load(os.path.join(d, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    import cv2
    from grasp.capture import depth_to_points
    K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    rig = cv2.imread(os.path.join(d, "rigid.png"), 0)
    if rig is None:
        raise SystemExit("[stir] no rigid.png (run thor_snap --perceive first)")
    rp, rg = depth_to_points(np.load(os.path.join(d, "depth.npy")), K, color_bgr=np.repeat(rig[..., None], 3, -1), zmax=1.0)
    rp = rp[rg[:, 0] > 0.5] @ T_bc[:3, :3].T + T_bc[:3, 3]
    rp = rp[hgt(rp) > 0.008]
    if len(rp) < 50:
        raise SystemExit("[stir] no rigid neighbours next to the bar: nothing to loosen, pick it directly")
    mu = obj[:, :2].mean(0)
    _, V = np.linalg.eigh(np.cov((obj[:, :2] - mu).T)); ax = V[:, 1]; nx = np.array([-ax[1], ax[0]])
    al, ac = (obj[:, :2] - mu) @ ax, (obj[:, :2] - mu) @ nx
    ra, rc = (rp[:, :2] - mu) @ ax, (rp[:, :2] - mu) @ nx
    side = 1.0 if np.median(rc[np.abs(rc) < 0.12]) > 0 else -1.0
    seam = []
    for s in np.arange(al.min() + 0.01, al.max() - 0.005, 0.01):
        sel = np.abs(al - s) < 0.01
        if sel.sum() < 20:
            continue
        edge = float(np.percentile(side * ac[sel], 95))
        near = (np.abs(ra - s) < 0.012) & (side * rc > edge - 0.005) & (side * rc < edge + a.seam_max)
        if near.sum() < 10:
            continue
        nb = side * rc[near]
        gap = max(0.0, float(np.percentile(nb, 5)) - edge)
        top = min(float(np.percentile(hgt(obj[sel]), 90)), float(np.percentile(hgt(rp[near]), 90)))
        xy = mu + s * ax + side * nx * (edge + gap / 2)
        h = max(a.min_tip, top - a.below)
        p = np.r_[xy, 0.0]; p[2] = (h - d_b - n_b[:2] @ xy) / n_b[2]
        seam.append((s, p, gap, top))
    print(f"[stir] bar {np.ptp(al)*100:.0f} cm, seam found at {len(seam)} of the 1 cm slices: "
          + " ".join(f"{x[0]*100:+.0f}:{x[2]*100:.1f}/{x[3]*100:.1f}" for x in seam) + "  (along: gap/top cm)")
    if len(seam) < 3:
        raise SystemExit("[stir] no seam along the bar")
    print("[stir] seam xy: " + " ".join(f"({x[1][0]:.2f},{x[1][1]:.2f},{x[1][2]:.2f})" for x in seam[::3]))
    stat = {"down": 0, "path": 0, "above": 0}
    ik = OpenArmLeftIK()
    q0 = np.deg2rad([float(v) for v in a.q.split(",")])
    rng = np.random.default_rng(0)
    rand_seeds = [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(8)]
    tipz = CFG["flange_to_tip_m"]

    def frame(yaw_deg, tilt, tdir):
        # closed jaws are a ~44 x 38 mm block, so their heading hardly matters for the stroke: any yaw the arm reaches
        # (near the robot only closing-axis headings around 210-270 deg from base x solve, 11/21 probe)
        y_ = np.deg2rad(yaw_deg); h = np.array([np.cos(y_), np.sin(y_), 0.0])
        hd = np.array([np.cos(y_ + np.deg2rad(tdir)), np.sin(y_ + np.deg2rad(tdir)), 0.0])
        t = np.deg2rad(tilt); z = np.cos(t) * np.array([0, 0, -1.0]) + np.sin(t) * hd
        x = h - np.dot(h, z) * z; x /= np.linalg.norm(x)
        return np.c_[x, np.cross(z, x), z]

    def solve(Tg, seeds):
        for sd in seeds:
            q, ok, _, _ = ik.solve(se3(Tg @ np.linalg.inv(T_L7_G)), sd, position_only=False, max_iter=300)
            if ok and ik.within_limits(q) and not in_keepout(ik, q):
                return q
        return None

    # candidate strokes: contiguous seam runs, start at the end farther from the robot (pull toward it)
    best = None
    pts = [x[1] for x in seam]
    if a.across:
        # 11/21 20:43: strokes along the seam barely touched anything (joint lag <= 3.4 deg). Across: the closed jaws
        # go down into the seam and move sideways AWAY from the target bar, pushing the neighbours off it.
        out_dir = side * nx
        cands = [(None, [p + np.r_[out_dir * a.across * k / 5, 0.0] for k in range(6)]) for p in
                 sorted(pts, key=lambda p: p[0])]  # nearest the robot first (reach)
        L_iter = [a.across]
    else:
        cands = None
        L_iter = np.arange(a.max_len, a.min_len - 1e-9, -0.02)
    for L in L_iter:
        n = int(round(L / 0.01))
        windows = cands if cands is not None else [(i0, None) for i0 in range(0, len(pts) - n)]
        for i0, way_ in windows:
          if way_ is not None:
            way = way_; dirv = side * nx
          else:
            run = pts[i0:i0 + n + 1]
            if np.linalg.norm(run[-1][:2] - run[0][:2]) < 0.8 * L:
                continue  # gaps in the seam
            if run[0][0] < run[-1][0]:
                run = run[::-1]  # far -> near
            dirv = (run[-1][:2] - run[0][:2]); dirv /= np.linalg.norm(dirv)
            way = [run[0] + (run[-1] - run[0]) * k / 5 for k in range(6)]
          if True:
            for tilt in (0.0, 15.0, 30.0, 45.0):
                if tilt > a.max_tilt:
                    continue
                for tdir in (0, 90, 180, 270):
                    for yaw in (270, 240, 210, 300, 180, 150, 120, 330, 0, 30, 60, 90):
                        R = frame(yaw, tilt, tdir)
                        Ts = []
                        for p in [way[0] - R[:, 2] * a.above] + way + [way[-1] - R[:, 2] * a.above]:
                            T = np.eye(4); T[:3, :3] = R; T[:3, 3] = p - tipz * R[:, 2]; Ts.append(T)
                        qs = []
                        q_ = solve(Ts[1], [q0] + rand_seeds)  # the down pose first (hardest)
                        if q_ is None:
                            stat["down"] += 1
                            continue
                        qs = [None, q_]
                        okk = True
                        for T in Ts[2:]:
                            q_ = solve(T, [qs[-1], qs[1]])
                            if q_ is None or np.max(np.abs(q_ - qs[-1])) > np.deg2rad(20):
                                okk = False; break
                            qs.append(q_)
                        if not okk:
                            stat["path"] += 1
                            continue
                        qa = solve(Ts[0], [qs[1], q0])
                        if qa is None:
                            stat["above"] += 1
                            continue
                        qs[0] = qa
                        best = dict(L=L, way=way, dirv=dirv, tilt=tilt, qs=qs)
                        break
                    if best:
                        break
                if best:
                    break
            if best:
                break
        if best:
            break
    if best is None:
        raise SystemExit(f"[stir] no reachable stroke along the seam (IK; failures {stat})")
    qs = [list(np.rad2deg(q)) for q in best["qs"]]
    out = {"mode": "stir", "q_start_deg": list(np.rad2deg(q0)), "q_above_deg": qs[0], "q_down_deg": qs[1],
           "q_path_deg": qs[2:-1], "q_rise_deg": qs[-1], "contact_base_m": best["way"][0].tolist(),
           "end_base_m": best["way"][-1].tolist(), "push_dir_base": list(best["dirv"]), "push_dist_m": float(best["L"]),
           "tilt_deg": best["tilt"]}
    json.dump(out, open(os.path.join(d, "push.json"), "w"), indent=1)
    print(f"[stir] stroke {best['L']*100:.0f} cm along the seam from {np.round(best['way'][0], 3).tolist()} "
          f"to {np.round(best['way'][-1], 3).tolist()}, tips {(hgt(best['way'][0][None])[0])*100:.1f} cm above the floor, tilt {best['tilt']:.0f}°")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
