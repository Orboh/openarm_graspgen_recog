#!/usr/bin/env python3
"""Plan a sideways push that opens a gap beside an aluminium bar in a pile. **Does not move the arm.**

2026-11-21 (Sota: "今の状態で取れるようにならないと" → 押して隙間を作る). In the mock pile the target bar touched a second
bar and a wood block along almost its whole length (0.4-2 cm), so no collision-free grasp exists. Pushing the bar from
outside toward its neighbours only packs them tighter; what opens a gap is pushing the part of the bar that sticks out
past its neighbours sideways, *away from* them (the bar turns / slides onto the free, soft side). Steps:

  1. target = object.npy (thor_snap --perceive), rigid neighbours = rigid.png pixels (other bars, wood) with depth.
  2. Bar axis (PCA in the base xy plane). Along the bar, every 1 cm: the free distance to rigid things on each side.
  3. Contact points: on the neighbour side of the bar (pusher between bar and neighbour needs room), where the free
     distance is at least --need (closed fingers, narrow side along the push) -> push direction = away from the
     neighbours. If no such spot, push the bar's end that sticks out past the neighbours lengthwise toward its free end.
  4. Pose: closed jaws pointing down (tilt <= --max-tilt), fingertips at the bar's local mid height, approach from
     --above straight down, push --dist along the floor, rise straight up. IK for all four, nearest to home.

Writes push.json next to plan.json: q_above_deg, q_down_deg, q_push_deg, q_rise_deg, q_start_deg, contact/dir, which
run_grasp.py --push executes (closed gripper, near-object speed). Usage:

    python grasp/tools/plan_push.py grasp/runs/<cap> --q <home deg> [--dist 0.06]
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

FX, FY = 0.022, 0.019  # one finger thickness along the closing axis, half the finger width (plan.py)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True, help="start (home) pose, lerobot deg")
    ap.add_argument("--dist", type=float, default=0.06, help="push length [m]")
    ap.add_argument("--need", type=float, default=0.030, help="free room the closed fingertips need beside the bar [m] "
                    "(fingers' narrow side 2x19 mm along the push, minus what pushes into the soft sack)")
    ap.add_argument("--above", type=float, default=0.05, help="start the straight-down approach this high above the contact [m]")
    ap.add_argument("--depth", type=float, default=0.5, help="fingertip height as a share of the bar's local height above the floor")
    ap.add_argument("--max-tilt", type=float, default=30.0)
    ap.add_argument("--no-collide", dest="collide", action="store_false", help="skip the gripper-vs-scene check")
    a = ap.parse_args(argv)
    d = a.cap_dir
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
    n_b, d_b = floor_plane_base(d, T_bc)
    obj = np.load(os.path.join(d, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    import cv2
    from grasp.capture import depth_to_points
    K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    rig = cv2.imread(os.path.join(d, "rigid.png"), 0)
    if rig is None:
        raise SystemExit("[push] no rigid.png (run thor_snap --perceive first)")
    rp, rg = depth_to_points(np.load(os.path.join(d, "depth.npy")), K, color_bgr=np.repeat(rig[..., None], 3, -1), zmax=1.0)
    rp = rp[rg[:, 0] > 0.5] @ T_bc[:3, :3].T + T_bc[:3, 3]
    rp = rp[(rp @ n_b + d_b) > 0.008]
    mu = obj[:, :2].mean(0)
    w, V = np.linalg.eigh(np.cov((obj[:, :2] - mu).T)); ax = V[:, 1]; nx = np.array([-ax[1], ax[0]])
    al, ac = (obj[:, :2] - mu) @ ax, (obj[:, :2] - mu) @ nx
    ra, rc = (rp[:, :2] - mu) @ ax, (rp[:, :2] - mu) @ nx
    side = 1.0 if np.median(rc[np.abs(rc) < 0.12]) > 0 else -1.0  # neighbours are on this side of the bar
    print(f"[push] bar length {np.ptp(al)*100:.0f} cm, {len(rp)} rigid points, neighbours on the {'+' if side > 0 else '-'}n side")
    cands = []
    for s in np.arange(al.min() + 0.01, al.max() - 0.005, 0.01):
        sel = np.abs(al - s) < 0.01
        if sel.sum() < 20:
            continue
        edge = np.percentile(side * ac[sel], 95)  # bar edge on the neighbour side
        hb = float(np.median(obj[sel] @ n_b + d_b))
        near = np.abs(ra - s) < FY + 0.01
        gaps = side * rc[near] - edge
        room = float(gaps[gaps > -0.005].min()) if np.any(gaps > -0.005) else 0.20
        cands.append((s, edge, hb, room))
    ok = [c for c in cands if c[3] >= a.need]
    print("[push] free room on the neighbour side along the bar: " + " ".join(f"{c[0]*100:+.0f}:{c[3]*100:.1f}" for c in cands))
    if not ok:
        raise SystemExit(f"[push] no spot with {a.need*100:.0f} cm room beside the bar on the neighbour side")
    ik = OpenArmLeftIK()
    q0 = np.deg2rad([float(v) for v in a.q.split(",")])
    tipz, tcpz = CFG["flange_to_tip_m"], CFG["flange_to_tip_m"] - 0.015
    ends = sorted(ok, key=lambda c: -abs(c[0]))  # a push near an end turns the bar most per cm
    best = None
    for s, edge, hb, room in ends:
        dirv = -side * nx  # away from the neighbours
        cxy = mu + s * ax + side * nx * (edge + min(room, 0.04) / 2)  # middle of the free room
        tip_h = max(0.006, a.depth * hb)
        combos = [(yaw, tilt, tdir) for tilt in (0.0, 15.0, 30.0, 45.0) if tilt <= a.max_tilt
                  for yaw in (0, 90, 180, 270) for tdir in (0, 90, 180, 270)][: None]
        fails = {}
        for yaw, tilt, tdir in combos:
            if True:
                # gripper frame: z = approach (down, tilted by `tilt` toward compass direction tdir relative to the push);
                # y = the push direction turned by `yaw` about z (0/180: fingers' narrow side leads, 90/270: wide side)
                c_, s_ = np.cos(np.deg2rad(tdir)), np.sin(np.deg2rad(tdir))
                hdir = np.r_[c_ * dirv - s_ * np.array([-dirv[1], dirv[0]]), 0]
                t = np.deg2rad(tilt)
                z = np.cos(t) * np.array([0, 0, -1.0]) + np.sin(t) * hdir
                c2, s2 = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
                p_ = np.r_[dirv, 0]; p_ = p_ - np.dot(p_, z) * z; p_ /= np.linalg.norm(p_)
                y = c2 * p_ + s2 * np.cross(z, p_)
                x = np.cross(y, z)
                R = np.c_[x, y, z]
                tip = np.r_[cxy, 0.0]
                tip[2] = (tip_h - d_b - n_b[:2] @ cxy) / n_b[2]  # point at height tip_h above the floor plane
                poses = {}
                # approach and leave along the gripper's own axis (like a pregrasp): straight up is out of reach at the far end
                steps = [("above", -z * a.above), ("down", np.zeros(3))] + \
                        [(f"push{k}", np.r_[dirv * a.dist * k / 4, 0]) for k in (1, 2, 3, 4)] + \
                        [("rise", np.r_[dirv * a.dist, 0] - z * a.above)]
                rng = np.random.default_rng(0)
                rand_seeds = [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(6)]
                order = [steps[1], steps[0]] + steps[2:]  # solve the contact pose first (hardest), seed the rest from it
                for name, off in order:
                    Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = tip + off - tipz * z
                    seed = list(poses.values())[-1] if poses else q0  # straight-line push: seed from the previous step
                    if name.startswith("push") or name == "rise":
                        seed = poses[f"push{int(name[4:]) - 1}"] if name.startswith("push") and name != "push1" else (poses["down"] if name == "push1" else poses["push4"])
                    qs = None
                    for sd in [seed, q0, np.deg2rad([0, 0, 0, 60, 0, 0, 0])] + (rand_seeds if not poses else []):
                        q, okk, _, _ = ik.solve(se3(Tg @ np.linalg.inv(T_L7_G)), sd, position_only=False, max_iter=300)
                        if okk and ik.within_limits(q) and not in_keepout(ik, q):
                            qs = q; break
                    if qs is None:
                        fails[name] = fails.get(name, 0) + 1
                        break
                    poses[name] = qs
                if len(poses) == len(steps):
                    poses = {k: poses[k] for k, _ in steps}
                if len(poses) == len(steps) and a.collide:
                    # gripper vs scene for down + push steps: closed fingers must not touch rigid things; the hand
                    # body (palm and the camera bracket, wider and higher up) must not touch rigid things or the bar
                    hit = 0
                    for name, off in steps[1:-1]:
                        Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = tip + off - tipz * z
                        for pts, boxes in ((rp, [(FX + 0.003, FY + 0.003, 0.055), (0.050, 0.040, tipz)]), (obj, [(0.050, 0.040, tipz)])):
                            q_ = (pts - Tg[:3, 3]) @ R
                            for hx, hy, back in boxes:
                                zlo = tipz - back if back < tipz else 0.0
                                zhi = tipz if back < tipz else tipz - 0.055
                                hit += int(np.sum((np.abs(q_[:, 0]) <= hx) & (np.abs(q_[:, 1]) <= hy) & (q_[:, 2] >= zlo) & (q_[:, 2] <= zhi)))
                    if hit > 15:
                        fails["collides"] = fails.get("collides", 0) + 1
                        continue
                if len(poses) == len(steps):
                    seq = list(poses.values())
                    jump = max(np.max(np.abs(q2 - q1)) for q1, q2 in zip(seq[1:-1], seq[2:-1]))
                    if jump < np.deg2rad(12):  # each 1.5 cm push step: small joint change = near-straight line
                        best = dict(s=s, room=room, tilt=tilt, tip=tip, dirv=dirv, poses=poses, hb=hb)
                        break
        if best:
            break
        print(f"[push] {s*100:+.0f} cm: no IK ({len(combos)} poses; first failing step counts {fails})")
    if best is None:
        raise SystemExit("[push] no reachable push pose (IK) at any spot with room")
    out = {"mode": "push", "q_start_deg": list(np.rad2deg(q0)),
           **{f"q_{k}_deg": list(np.rad2deg(v)) for k, v in best["poses"].items() if not k.startswith("push")},
           "q_path_deg": [list(np.rad2deg(best["poses"][f"push{k}"])) for k in (1, 2, 3, 4)],
           "contact_base_m": best["tip"].tolist(), "push_dir_base": list(best["dirv"]), "push_dist_m": a.dist,
           "along_bar_m": float(best["s"]), "room_m": best["room"], "tilt_deg": best["tilt"], "bar_height_m": best["hb"]}
    json.dump(out, open(os.path.join(d, "push.json"), "w"), indent=1)
    print(f"[push] contact {np.round(best['tip'], 3).tolist()} at {best['s']*100:+.0f} cm along the bar (room {best['room']*100:.1f} cm), "
          f"push {a.dist*100:.0f} cm toward {np.round(best['dirv'], 2).tolist()}, tilt {best['tilt']:.0f}°")
    for k, q in [("above", out["q_above_deg"]), ("down", out["q_down_deg"])] + [(f"push{i+1}", q) for i, q in enumerate(out["q_path_deg"])] + [("rise", out["q_rise_deg"])]:
        print(f"  q_{k:<6}" + " ".join(f"{v:+6.1f}" for v in q))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
