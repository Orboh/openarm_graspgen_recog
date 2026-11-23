#!/usr/bin/env python3
"""Left-arm poses for a hand-eye run after the chest camera was moved. **Does not move the arm.**

2026-11-22 夜 (the chest camera went onto the Kitakyushu mount). Random hand poses over the grasp area (jaws down,
tilted up to 40 deg, any yaw) are kept when a wrist marker (markers_link7_m) falls inside the first-guess camera's
image and no other arm point covers it, the arm stays >= 4 cm from everything in world.yml (cart tray, body, camera,
clamp, parked right arm) and the jaws stay >= 8 cm above the floor. The route (home v5 -> raised 8 cm -> standby ->
poses -> standby -> raised -> home) is a shortest path over clear joint-space legs (grasp/arm_collision.py).

    GRASP_ARM=left ~/work/GraspGenX/.venv/bin/python grasp/tools/handeye_marker_poses.py <T_guess.json> <intrinsics.json> \
        <out_poses.json> <out_traj.json> [N]
"""
import heapq
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("GRASP_ARM", "left")
import pinocchio as pin  # noqa: E402

from grasp.arm_collision import ArmCollision  # noqa: E402
from grasp.plan import CFG, T_L7_G, se3  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402

J = [f"joint_{i}" for i in range(1, 8)]
FLOOR = 0.28


def yq(p):
    y = yaml.safe_load(open(os.path.join(REPO, p)))
    return np.array([float(y[j]) for j in J])


def main():
    cam_file, intr, out_poses, out_traj = sys.argv[1:5]
    N = int(sys.argv[5]) if len(sys.argv) > 5 else 30
    Jc = json.load(open(cam_file)); T = np.array(Jc["T_base_cam"]); Ti = np.linalg.inv(T); cam = T[:3, 3]
    Mk = [np.array(m) for m in Jc["markers_link7_m"]]
    I = json.load(open(intr)); K = np.array([[I["fx"], 0, I["ppx"]], [0, I["fy"], I["ppy"]], [0, 0, 1]])
    ik = OpenArmLeftIK(); tipz = CFG["flange_to_tip_m"]
    AC = ArmCollision(ik, "left", os.path.join(REPO, "grasp/config/sites/tokyo-thor.world.yml"))
    home = yq("grasp/config/home_left_cart-sym-v5.yaml"); sb = yq("grasp/config/standby_left_cart-v1.yaml")
    right_home = yq("grasp/config/home_right_cart-sym-v5.yaml")
    lo, hi = np.degrees(ik.lo) + 3, np.degrees(ik.hi) - 3
    seeds = [sb] + [np.array(v, float) for v in ([-25, -30, 40, 85, 0, -35, 40], [-3, -31, 20, 96, 10, -44, 34],
                                                 [-50, -15, 80, 60, -20, -20, 40], [-37, -28, 41, 63, -11, -39, 52])]

    def marker_seen(qd):
        M = ik.fk(np.deg2rad(qd)); P, lab = AC.points(np.deg2rad(qd)); n = 0
        for m in Mk:
            p = M.rotation @ m + M.translation; c = Ti[:3, :3] @ p + Ti[:3, 3]
            if c[2] <= 0.1:
                continue
            u, v = K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]
            if not (40 < u < 600 and 40 < v < 440):
                continue
            d = p - cam; L = np.linalg.norm(d); d /= L
            sx = (P - cam) @ d; perp = np.linalg.norm((P - cam) - np.outer(sx, d), axis=1)
            if not ((sx > 0.05) & (sx < L - 0.04) & (perp < 0.02)).any():
                n += 1
        return n

    rng = np.random.default_rng(7)
    cands = []
    for it in range(4000):
        if len(cands) >= 4 * N:
            break
        x, y, h = rng.uniform(0.22, 0.50), rng.uniform(-0.10, 0.30), rng.uniform(0.10, 0.30)
        tilt, phi, yaw = np.deg2rad(rng.uniform(0, 40)), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi)
        a = np.array([np.sin(tilt) * np.cos(phi), np.sin(tilt) * np.sin(phi), -np.cos(tilt)])
        xx = np.array([np.cos(yaw), np.sin(yaw), 0.0]); xx -= a * (xx @ a)
        if np.linalg.norm(xx) < 0.3:
            continue
        xx /= np.linalg.norm(xx); Rg = np.c_[xx, np.cross(a, xx), a]
        Tg = np.eye(4); Tg[:3, :3] = Rg; Tg[:3, 3] = np.array([x, y, FLOOR + h]) - tipz * a
        sd = seeds[it % len(seeds)]
        q, ok, _, _ = ik.solve(se3(Tg @ np.linalg.inv(T_L7_G)), np.deg2rad(sd), position_only=False, max_iter=200)
        qd = np.degrees(q)
        if not (ok and np.all(qd > lo) and np.all(qd < hi)):
            continue
        if marker_seen(qd) == 0:
            continue
        cl = AC.clearances(np.deg2rad(qd))
        if min(c for k, (c, _) in cl.items() if k != "chest_camera") < 0.04 or cl.get("chest_camera", (1, ""))[0] < 0.02:
            continue  # the camera box is already 4 cm larger than the camera until the refit
        P, lab = AC.points(np.deg2rad(qd))
        if P[:, 2].min() < FLOOR + 0.06:
            continue
        cands.append(qd)
    print(f"[he] {len(cands)} candidate poses with a marker in view")
    # spread them: greedy farthest-point in joint space
    pick = []
    if cands:
        C = np.array(cands); pick = [0]
        while len(pick) < min(N, len(C)):
            d = np.min(np.abs(C[:, None, :] - C[pick][None, :, :]).max(2), axis=1)
            pick.append(int(np.argmax(d)))
        pick = [C[i] for i in pick]
    nodes = [sb] + pick; n = len(nodes)
    clear = {}

    def ok_leg(i, j):
        k = (min(i, j), max(i, j))
        if k not in clear:
            clear[k] = AC.check_path(nodes[k[0]], nodes[k[1]], margin=0.02)[0]
        return clear[k]
    w = lambda i, j: float(np.abs(nodes[i] - nodes[j]).max())

    def shortest(src, dst):
        dist, prev, hq = {src: 0.0}, {}, [(0.0, src)]
        while hq:
            dd, u = heapq.heappop(hq)
            if u == dst:
                break
            if dd > dist.get(u, 1e9):
                continue
            for v in range(n):
                if v != u and ok_leg(u, v) and dd + w(u, v) < dist.get(v, 1e9):
                    dist[v] = dd + w(u, v); prev[v] = u; heapq.heappush(hq, (dist[v], v))
        if dst not in dist:
            return None
        p = [dst]
        while p[-1] != src:
            p.append(prev[p[-1]])
        return p[::-1]
    order, cur, todo = [0], 0, set(range(1, n))
    while todo:
        t = min(todo, key=lambda t: w(cur, t))
        sp = shortest(cur, t)
        if sp is None:
            todo.discard(t); print(f"[he] pose {t} unreachable, dropped"); continue
        order += sp[1:]; cur = t; todo -= set(sp)
    back = shortest(cur, 0) or [cur, 0]
    order += back[1:]
    M = ik.fk(np.deg2rad(home))
    qu, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, 0.08])), np.deg2rad(home), position_only=False)
    up = np.rad2deg(qu) if ok else home
    route = [home, up] + [nodes[i] for i in order] + [up, home]
    frames, bad = [], []
    for a_, b_ in zip(route[:-1], route[1:]):
        okp, wst = AC.check_path(a_, b_, margin=0.02)
        if not okp:
            bad.append(wst)
        for s in np.linspace(0, 1, max(2, int(np.abs(b_ - a_).max() / 3))):
            qq = a_ + (b_ - a_) * s; cl = AC.clearances(np.deg2rad(qq))
            frames.append({"q": {"left": [round(float(v), 1) for v in qq], "right": right_home.tolist()},
                           "label": "hand-eye after the camera move", "clear_cm": {k: round(v[0] * 100, 1) for k, v in cl.items()}})
    worst = {}
    for f in frames:
        for k, v in f["clear_cm"].items():
            worst[k] = min(worst.get(k, 99), v)
    print(f"[he] route: {len(order)} stops ({n - 1} poses), legs not clear: {len(bad)}; worst clearance cm {worst}")
    json.dump({"poses": [[round(float(v), 1) for v in nodes[i]] for i in order],
               "note": "2026-11-22 hand-eye poses for the chest camera on the Kitakyushu mount (grasp/tools/handeye_marker_poses.py)"},
              open(out_poses, "w"), indent=1)
    json.dump({"frames": frames}, open(out_traj, "w"))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
