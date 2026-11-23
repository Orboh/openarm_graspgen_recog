#!/usr/bin/env python3
"""記録の軌跡（traj.csv）か計画（plan.json）を、cuRobo の左腕の球と置き場の周りの物（箱とメッシュ）で調べる。**腕は動かさない。**

traj.csv の隣り合う行の間を関節の直線で補間し（run_grasp.py の動かし方と同じ）、周りの物ごとに一番近づいた所を出す。
traj.csv が無ければ plan.json の start → pre → grasp → lift → pre → start を調べる。

    set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
    ~/work/curobo/.venv/bin/python grasp/curobo/check_runs.py grasp/runs/cap_resident_20261121_160044 [...]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from grasp.curobo.model import ArmSpheres, MeshDistance, load_world, obstacle_distances  # noqa: E402

J = [f"joint_{i}.pos" for i in range(1, 8)]


def waypoints(cap_dir: str):
    """(名前, 関節角 [deg]) の並び。"""
    tp = os.path.join(cap_dir, "traj.csv")
    if os.path.exists(tp):
        rows = list(csv.DictReader(open(tp)))
        return [(r["phase"], [float(r[j]) for j in J]) for r in rows]
    p = json.load(open(os.path.join(cap_dir, "plan.json")))
    seq = [("start", p["q_start_deg"]), ("pre", p["q_pre_deg"]), ("grasp", p["q_grasp_deg"]),
           ("lift", p["q_lift_deg"]), ("pre", p["q_pre_deg"]), ("start", p["q_start_deg"])]
    return seq


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dirs", nargs="+")
    ap.add_argument("--n", type=int, default=30, help="1 区間を何点で調べるか")
    ap.add_argument("--world", default=None, help="置き場の箱の YAML（無ければ GRASP_WORLD_YML）")
    a = ap.parse_args(argv)
    arm = ArmSpheres()
    world = load_world(a.world)
    meshes = [MeshDistance(m) for m in world["mesh"]]
    print("[check] 周りの物: " + ", ".join([c["name"] for c in world["cuboid"]] + [m.name for m in meshes]))
    for d in a.cap_dirs:
        wp = waypoints(d)
        Q, lab = [], []
        for (n0, q0), (n1, q1) in zip(wp[:-1], wp[1:]):
            for s in np.linspace(0, 1, a.n):
                Q.append(np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * s); lab.append(f"{n0}→{n1}")
        Q = np.deg2rad(np.array(Q))
        S = arm.spheres(Q)
        out = []
        for name, D in obstacle_distances(S, world, meshes, arm.sphere_links).items():  # D: (N, S)
            k = np.unravel_index(np.argmin(D), D.shape)
            out.append(f"{name} {D[k]*1000:+6.0f} mm（{arm.sphere_links[k[1]].replace('openarm_left_', '')}・{lab[k[0]]}）")
        print(f"{os.path.basename(d.rstrip('/'))}: " + " / ".join(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
