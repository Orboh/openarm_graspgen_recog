#!/usr/bin/env python3
"""Plan where to drop the object after the lift: carry it to the robot's left, beside the left shoulder, and open
the jaws there (pick -> lift -> throw aside, like sorting). **Does not move the arm.**

2026-11-21 (Sota): "持ち上げたあと、真横に落とすようにして ... ロボットの左へ ... 肩の真横まで". Putting the bare aluminium bar
back down at the grasp spot pushed it a little further left every trial until it was out of reach.

The drop pose keeps the lift pose's hand orientation turned about the vertical by one of --yaws (jaws still pointing
down; beside the shoulder the arm cannot keep the forward-facing lift orientation, it needs ~60 deg outward) and puts the
jaw TCP at --at (base frame). IK is
seeded from the lift pose and nearby seeds; the solution must stay within the joint limits and within --max-change of
the lift pose. The straight joint path lift -> drop must keep link7 and the TCP >= floor + --clear (the held object
hangs below the TCP). Writes q_drop_deg and drop_point_m into <cap_dir>/plan.json.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/plan_drop.py grasp/runs/cap_xxx [--at 0.08,0.38,0.40]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import pinocchio as pin  # noqa: E402

from grasp.plan import floor_plane_base  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402
from ik.openarm_right_ik import arm_ik_class  # noqa: E402

TCP_OFF = np.array([0.0, 0.0, 0.2145])  # IK frame (URDF hand_tcp = flange) -> jaw TCP, same as run_grasp.py check()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--at", default="0.05,0.45,0.42", help="drop point x,y,z [m] in the base frame (left shoulder is at about -0.02,0.15,0.72)")
    ap.add_argument("--home", default=os.path.join(REPO, "grasp", "config", f"home_{os.environ.get('GRASP_ARM', 'left')}_cart-sym-v4.yaml"),
                    help="start pose YAML: the arm goes lift -> pregrasp -> raised home (run_grasp's checked return path) -> drop, "
                         "so the drop pose is planned from home (its J1 is already turned left), not from the lift")
    ap.add_argument("--yaws", default="0,30,60,90,-30", help="hand turns about the vertical to try [deg]")
    ap.add_argument("--carry-up", type=float, default=0.08, help="carry via the drop point raised by this much [m]")
    ap.add_argument("--max-change", type=float, default=90.0, help="max joint change from the lift pose [deg]")
    ap.add_argument("--clear", type=float, default=0.08, help="link7 and TCP stay this high above the floor along lift->drop [m]")
    a = ap.parse_args(argv)
    plan_path = os.path.join(a.cap_dir, "plan.json")
    plan = json.load(open(plan_path))
    T_bc = np.array(json.load(open(os.path.join(REPO, "grasp", "config", "T_base_cam.json")))["T_base_cam"])
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    ik = arm_ik_class()()
    import yaml
    hy = yaml.safe_load(open(a.home))
    q_lift = np.deg2rad([float(hy[f"joint_{i}"]) for i in range(1, 8)])  # reference = home (variable name kept)
    M_lift = ik.fk(np.deg2rad(plan["q_lift_deg"]))  # hand orientation reference: the lift (jaws down, object as grasped)
    at = np.array([float(v) for v in a.at.split(",")])

    R_home = ik.fk(q_lift).rotation  # q_lift holds home here: home's hand points forward, roughly horizontal

    def target(yaw_deg, dz=0.0, ref=None):  # IK frame is the flange; put the jaw TCP at --at (+dz up)
        c, s_ = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        R = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]]) @ (M_lift.rotation if ref is None else ref)
        return pin.SE3(R, at + np.array([0, 0, dz]) - R @ TCP_OFF)

    def height(q):
        M = ik.fk(q)
        return float(n_b @ (M.translation + M.rotation @ TCP_OFF) + d_b)

    def path_ok(qa, qb):
        for s in np.linspace(0, 1, 25):
            q = qa + (qb - qa) * s
            M = ik.fk(q)
            tcp = M.translation + M.rotation @ TCP_OFF
            if min(float(n_b @ M.translation + d_b), float(n_b @ tcp + d_b)) < a.clear:
                return False
        return True

    rng = np.random.default_rng(0)
    best = None
    # hand orientations tried: the lift's (jaws down) and home's (jaws forward; 11/21: beside the shoulder only this one is
    # reachable close to home, ~37 deg at +60 deg yaw), each turned about the vertical
    for ref_name, ref in (("lift", M_lift.rotation), ("home", R_home)):
      for yaw in [float(v) for v in a.yaws.split(",")]:
        for sd in [q_lift] + [q_lift + rng.normal(0, 0.5, 7) for _ in range(12)]:
            q, ok, _, _ = ik.solve(target(yaw, ref=ref), np.clip(sd, ik.lo, ik.hi), position_only=False, max_iter=300)
            if not (ok and ik.within_limits(q)):
                continue
            dmax = float(np.max(np.abs(np.rad2deg(q - q_lift))))
            if dmax > a.max_change:
                continue
            # carry high: lift -> (drop point + --carry-up) -> drop point; a straight joint path lift -> drop dips toward the floor
            qc, okc, _, _ = ik.solve(target(yaw, a.carry_up, ref=ref), q, position_only=False, max_iter=300)
            if not (okc and ik.within_limits(qc)) or np.max(np.abs(np.rad2deg(qc - q))) > 30:
                continue
            if not (path_ok(q_lift, qc) and path_ok(qc, q)):
                continue
            if best is None or dmax < best[0]:
                best = (dmax, q, f"{ref_name}{yaw:+.0f}", qc)
    if best is None:
        print(f"[drop] no drop pose at {a.at} (hand turned by --yaws, <= {a.max_change:.0f} deg, path >= {a.clear:.2f} m above the floor)")
        return 1
    dmax, q, yaw, qc = best
    # before turning the hand toward the shoulder, raise the held object straight up (sim 11/21: turning right after the
    # lift swung the bar's lower end to 0.8 cm above the floor)
    ql = np.deg2rad(plan["q_lift_deg"]); Ml = ik.fk(ql); qh = None
    for dz in (0.15, 0.12, 0.10, 0.08, 0.06):
        for sd in [ql] + [ql + rng.normal(0, 0.3, 7) for _ in range(10)]:
            qh_, okh, _, _ = ik.solve(pin.SE3(Ml.rotation, Ml.translation + np.array([0, 0, dz])), np.clip(sd, ik.lo, ik.hi),
                                      position_only=False, max_iter=300)
            if okh and ik.within_limits(qh_) and np.max(np.abs(np.rad2deg(qh_ - ql))) < 60:
                qh = qh_; break
        if qh is not None:
            print(f"[drop] raise the held object {dz * 100:.0f} cm straight up first"); break
    if qh is None:  # lift pose near a joint limit: let the hand tilt a little while going up (position only, small change)
        for dz in (0.10, 0.08):
            for sd in [ql] + [ql + rng.normal(0, 0.2, 7) for _ in range(10)]:
                qh_, okh, _, _ = ik.solve(pin.SE3(Ml.rotation, Ml.translation + np.array([0, 0, dz])), np.clip(sd, ik.lo, ik.hi),
                                          position_only=True, max_iter=300)
                if okh and ik.within_limits(qh_) and np.max(np.abs(np.rad2deg(qh_ - ql))) < 30:
                    tilt = np.degrees(np.arccos(np.clip(ik.fk(qh_).rotation[:, 2] @ Ml.rotation[:, 2], -1, 1)))
                    if tilt <= 25:
                        qh = qh_; break
            if qh is not None:
                print(f"[drop] raise the held object {dz * 100:.0f} cm up first (hand tilts {tilt:.0f} deg)"); break
    if qh is None:
        print("[drop] cannot raise the held object above the lift pose"); return 1
    plan.update(q_lift_high_deg=np.rad2deg(qh).tolist(), q_drop_carry_deg=np.rad2deg(qc).tolist(), q_drop_deg=np.rad2deg(q).tolist(), drop_point_m=at.tolist(), drop_yaw_deg=yaw)
    json.dump(plan, open(plan_path, "w"), indent=2)
    print(f"[drop] drop at {at.tolist()} (TCP {height(q) * 100:.0f} cm above the floor, hand {yaw} deg), max joint change from home "
          f"{dmax:.0f} deg; q_drop {np.round(np.rad2deg(q), 1).tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
