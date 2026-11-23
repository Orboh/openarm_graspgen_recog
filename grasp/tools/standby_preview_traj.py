#!/usr/bin/env python3
"""Make the joint trajectories (and arm_collision clearances) for the standby-pose preview. **Does not move the arm.**
2026-11-22 (Sota: 初期位置と作業の間に待機位置を置き、台車にぶつからないよう手をまず前に出す。MuJoCo で見せて): home v5 →
standby → an example pregrasp → grasp → back, for each arm. Output: grasp/runs/sim/standby_traj.json (render with
grasp/tools/mujoco_preview.py)."""
import json, os, sys
import numpy as np, yaml
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, REPO)
os.environ.setdefault("GRASP_WORLD_YML", os.path.join(REPO, "grasp/config/sites/tokyo-thor.world.yml"))
L = lambda f: [float(yaml.safe_load(open(f))[f"joint_{i}"]) for i in range(1, 8)]
SB = {"left": [60.5, -0.0, -5.1, 136.2, -2.1, -0.0, -22.1], "right": [-61.8, -0.1, 5.1, 136.3, 2.1, 0.5, 23.3]}
EX = {"right": ([69.4, 15.2, -75.9, 73.4, 65.7, 2.2, -80.2], [46.6, -3.0, -46.0, 48.1, 25.2, 22.3, -73.1]),
      "left": ([-70.7, -20.9, 65.4, 42.8, -76.4, 18.9, 84.5], [-65.0, -2.1, 64.6, 6.8, -74.2, 1.3, 67.4])}
out = {"frames": []}
home = {a: L(os.path.join(REPO, f"grasp/config/home_{a}_cart-sym-v5.yaml")) for a in ("left", "right")}
from grasp.arm_collision import ArmCollision
for arm in ("right", "left"):
    os.environ["GRASP_ARM"] = arm
    import importlib, ik.openarm_right_ik as m; importlib.reload(m)
    ik = m.arm_ik_class()(); ac = ArmCollision(ik)
    pre, gr = EX[arm]
    import pinocchio as pin
    def shifted(qd, off):
        M = ik.fk(np.deg2rad(qd)); q, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array(off)), np.deg2rad(qd), position_only=False, max_iter=300)
        return np.rad2deg(q).tolist() if ok and ik.within_limits(q) else None
    # like run_grasp.route(): a leg that comes within 1.5 cm of the clamp / camera / body goes via the standby moved forward
    via = None
    okl, w = ac.check_path(SB[arm], pre)
    if not okl:
        for off in ((0.10, 0, 0), (0.15, 0, 0), (0.10, 0, 0.05)):
            v = shifted(SB[arm], off)
            if v and ac.check_path(SB[arm], v)[0] and ac.check_path(v, pre)[0]:
                via = v; print(f"[{arm}] standby -> pregrasp goes via the standby moved {off} m"); break
    mid = [("standby + forward", via)] if via else []
    seq = [("home (v5)", home[arm]), ("standby", SB[arm])] + mid + [("pregrasp", pre), ("grasp", gr), ("pregrasp", pre)] + mid[::-1] + [("standby", SB[arm]), ("home (v5)", home[arm])]
    for (n0, a), (n1, b) in zip(seq[:-1], seq[1:]):
        a, b = np.array(a), np.array(b); k = max(2, int(np.ceil(np.abs(b - a).max() / 2.0)))
        for s in np.linspace(0, 1, k):
            q = a + (b - a) * s
            cl = ac.clearances(np.deg2rad(q))
            other = "left" if arm == "right" else "right"
            out["frames"].append({"arm": arm, "q": {arm: q.tolist(), other: home[other]}, "label": f"{arm}: {n0} -> {n1}",
                                  "clear_cm": {k2: round(v[0] * 100, 1) for k2, v in cl.items()}})
        for _ in range(12):  # pause at each pose
            out["frames"].append(dict(out["frames"][-1], label=f"{arm}: {n1}"))
os.makedirs(os.path.join(REPO, "grasp/runs/sim"), exist_ok=True)
json.dump(out, open(os.path.join(REPO, "grasp/runs/sim/standby_traj.json"), "w"))
mins = {}
for f in out["frames"]:
    for k2, v in f["clear_cm"].items():
        mins[k2] = min(mins.get(k2, 99), v)
print(len(out["frames"]), "frames; min clearance over all [cm]:", mins)
