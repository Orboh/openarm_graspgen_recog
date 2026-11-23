#!/usr/bin/env python3
"""Poses to measure where the LEFT arm's fingertips really are when the jaws are tilted. **Does not move the arm.**

2026-11-22 (Sota: 左手を使えるようにしよ). Tilted left-arm grasps missed sideways by 2-3 cm; the left fingertip was only
ever checked with the jaws pointing straight down. This picks poses whose fingertips (12 cm above the floor) the chest
camera sees, with the jaws tilted 0 / 30 / 45 deg in several directions, and checks every leg of the route with
grasp/arm_collision.py (cart tray, body, camera, clamp, other arm):

    home v5 -> raised 8 cm (what handeye_near.py does first) -> standby v1 -> poses ... -> standby -> raised -> home

Writes the pose list for handeye_near.py --poses and a MuJoCo preview trajectory (grasp/tools/mujoco_preview.py).

    GRASP_ARM=left ~/work/GraspGenX/.venv/bin/python grasp/tools/left_tcp_poses.py <intrinsics.json> <out.json> <traj.json>
"""
import json
import os
import sys

import numpy as np
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
os.environ.setdefault("GRASP_ARM", "left")
import pinocchio as pin  # noqa: E402

from grasp.arm_collision import ArmCollision, describe  # noqa: E402
from grasp.plan import CFG, T_L7_G, se3  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402

J = [f"joint_{i}" for i in range(1, 8)]
FLOOR = 0.28
TIP_H = 0.12          # fingertips this far above the floor
MARGIN = 0.02


def yq(p):
    y = yaml.safe_load(open(os.path.join(REPO, p)))
    return np.array([float(y[j]) for j in J])


def main():
    intr, out_poses, out_traj = sys.argv[1:4]
    I = json.load(open(intr)); K = np.array([[I["fx"], 0, I["ppx"]], [0, I["fy"], I["ppy"]], [0, 0, 1]])
    cam_file = os.environ.get("LTCP_CAM", os.path.join(REPO, "grasp/config/T_base_cam.json"))  # e.g. a first guess after moving the camera
    T = np.array(json.load(open(cam_file))["T_base_cam"]); Ti = np.linalg.inv(T)
    cam = T[:3, 3]
    ik = OpenArmLeftIK(); tipz = CFG["flange_to_tip_m"]
    AC = ArmCollision(ik, "left", os.path.join(REPO, "grasp/config/sites/tokyo-thor.world.yml"))
    home = yq("grasp/config/home_left_cart-sym-v5.yaml"); sb = yq("grasp/config/standby_left_cart-v1.yaml")
    right_home = yq("grasp/config/home_right_cart-sym-v5.yaml")
    lo, hi = np.degrees(ik.lo) + 3, np.degrees(ik.hi) - 3

    def grip(qd):
        M = ik.fk(np.deg2rad(qd)); G = np.eye(4); G[:3, :3] = M.rotation; G[:3, 3] = M.translation
        return G @ T_L7_G

    def proj(p):
        c = Ti[:3, :3] @ p + Ti[:3, 3]
        return K[0, 0] * c[0] / c[2] + K[0, 2], K[1, 1] * c[1] / c[2] + K[1, 2]

    def tip_visible(qd):
        """Fingertip inside the image and no other arm point within 2.5 cm of the camera->tip ray (in front of the tip)."""
        G = grip(qd); tip = G[:3, 3] + G[:3, 2] * tipz
        u, v = proj(tip)
        if not (40 < u < 600 and 40 < v < 440):
            return False, tip, (u, v)
        P, lab = AC.points(np.deg2rad(qd))
        far = lab != "gripper"  # the fingers themselves are allowed; the hand body / links must not cover the tips
        far |= np.isin(lab, ["gripper"]) & ((P - tip) @ G[:3, 2] < -0.07)  # hand body (> 7 cm behind the tip)
        d = tip - cam; L = np.linalg.norm(d); d /= L
        s = (P[far] - cam) @ d; perp = np.linalg.norm((P[far] - cam) - np.outer(s, d), axis=1)
        block = (s > 0.05) & (s < L - 0.03) & (perp < 0.025)
        return not block.any(), tip, (u, v)

    # up = home raised 8 cm (as handeye_near.py computes it)
    M = ik.fk(np.deg2rad(home))
    qu, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, 0.08])), np.deg2rad(home), position_only=False)
    up = np.rad2deg(qu) if ok else home

    # IK seeds: today's real left-arm grasp postures (plan.json q_grasp_deg on Thor, 11/21 22:46-23:05, 11/22 02:08)
    seeds = [np.array(v, float) for v in ([-25, -30, 40, 85, 0, -35, 40], [-3, -31, 20, 96, 10, -44, 34],
                                          [-50, -15, 80, 60, -20, -20, 40], [-12, -14, 31, 89, 12, -36, 48],
                                          [-1, -19, 15, 85, 24, -13, 28], [-37, -28, 41, 63, -11, -39, 52])]
    # (x, y, tilt, direction): left J6 (+-42 deg) cannot point straight down this close, so: tilts forward (+x, 0)
    # and inward (-y, 270), as the real left grasps do, plus a near-vertical 15 deg as far out as the camera sees
    spots = [(x, y, t, f) for (x, y) in ((0.30, 0.10), (0.34, 0.16), (0.28, 0.20), (0.34, 0.04), (0.30, 0.24))
             for t in (30, 45) for f in (0, 270)] + [(0.38, 0.18, 15, f) for f in (0, 270)] + [(0.40, 0.20, 15, 0)]
    cands = []
    from collections import Counter
    why = Counter()
    spots = sorted(spots, key=lambda sp: sp[3] == 270)  # inward last: they may start from an accepted forward pose
    for (x, y, tilt, phi) in spots:
        for H in (TIP_H, 0.18):
            f, t = np.deg2rad(phi), np.deg2rad(tilt)
            a = np.array([np.sin(t) * np.cos(f), np.sin(t) * np.sin(f), -np.cos(t)])  # approach (gripper z)
            done = False
            for hd in range(0, 180, 30):
                h = np.deg2rad(hd); xx = np.array([np.cos(h), np.sin(h), 0.0]); xx -= a * (xx @ a)
                if np.linalg.norm(xx) < 0.3:
                    continue
                xx /= np.linalg.norm(xx); R0 = np.c_[xx, np.cross(a, xx), a]
                for flip in (1, -1):
                    R = R0 @ np.diag([flip, flip, 1])
                    Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = np.array([x, y, FLOOR + H]) - tipz * a
                    for sd in seeds:
                        q, okk, _, _ = ik.solve(se3(Tg @ np.linalg.inv(T_L7_G)), np.deg2rad(sd), position_only=False, max_iter=300)
                        qd = np.degrees(q)
                        if not (okk and np.all(qd > lo) and np.all(qd < hi)):
                            why[(tilt, phi, "ik/limits")] += 1; continue
                        vis, tip, uv = tip_visible(qd)
                        if not vis:
                            why[(tilt, phi, "tip hidden / out of view")] += 1; continue
                        cl = AC.clearances(np.deg2rad(qd))
                        if min(c for c, _ in cl.values()) < 0.04:
                            why[(tilt, phi, "close to " + min(cl, key=lambda k: cl[k][0]))] += 1; continue
                        if not (AC.check_path(sb, qd, margin=MARGIN)[0] or
                                any(AC.check_path(np.array(c["q"]), qd, margin=MARGIN)[0] for c in cands if c["phi"] != 270)):
                            why[(tilt, phi, "no clear path (standby / forward poses)")] += 1; continue
                        cands.append({"q": [round(float(v), 1) for v in qd], "tilt": tilt, "phi": phi, "xy": [x, y], "h": H,
                                      "uv": [int(uv[0]), int(uv[1])], "travel": float(np.abs(qd - sb).max())})
                        done = True; break
                    if done:
                        break
                if done:
                    break
    # pick: per (tilt, direction) up to 2 different spots (shortest travel from standby first), at most 10
    pick = []
    for tilt, phi in ((15, 0), (15, 270), (30, 0), (30, 270), (45, 0), (45, 270)):
        c = sorted([c for c in cands if c["tilt"] == tilt and c["phi"] == phi], key=lambda c: c["travel"])
        seen = set()
        for cc in c:
            if tuple(cc["xy"]) in seen:
                continue
            seen.add(tuple(cc["xy"])); pick.append(cc)
            if len(seen) == 2:
                break
    pick = pick[:10]
    if os.environ.get("LTCP_SWEEP"):
        # 16:12 run: the wrist markers showed the left FK 21-29 mm off at these poses, but only 6 marker views. For a
        # joint-offset fit: wrist variations of each pose (like handeye_near.py VAR) in which a wrist marker
        # (markers_link7_m of T_base_cam.json) is in the image and not covered by the arm.
        Mk = [np.array(m) for m in json.load(open(cam_file))["markers_link7_m"]]
        VAR = [(0, 0, 0, 0, 20, 0, 0), (0, 0, 0, 0, -20, 0, 0), (0, 0, 0, 0, 0, 15, 0), (0, 0, 0, 0, 0, -15, 0),
               (0, 0, 0, 0, 0, 0, 30), (0, 0, 0, 0, 0, 0, -30), (0, 0, 0, 0, 0, 0, 60), (0, 0, 0, 0, 0, 0, -60),
               (8, 0, 0, 0, 0, 0, 0), (-8, 0, 0, 0, 0, 0, 0)]

        def marker_seen(qd):
            M = ik.fk(np.deg2rad(qd)); P, lab = AC.points(np.deg2rad(qd))
            for m in Mk:
                p = M.rotation @ m + M.translation; u, v = proj(p)
                if not (30 < u < 610 and 30 < v < 450):
                    continue
                d = p - cam; L = np.linalg.norm(d); d /= L
                sx = (P - cam) @ d; perp = np.linalg.norm((P - cam) - np.outer(sx, d), axis=1)
                if not ((sx > 0.05) & (sx < L - 0.04) & (perp < 0.02)).any():
                    return True
            return False
        extra = []
        for c in list(pick):
            for v in VAR:
                qd = np.array(c["q"]) + v
                if not (np.all(qd > lo) and np.all(qd < hi)):
                    continue
                G = grip(qd)
                if (G[:3, 3] + G[:3, 2] * tipz)[2] < FLOOR + 0.07 or not marker_seen(qd):
                    continue
                cl = AC.clearances(np.deg2rad(qd))
                if min(cc for cc, _ in cl.values()) < 0.04:
                    continue
                extra.append(dict(c, q=[round(float(x), 1) for x in qd], var=list(v)))
        pick = [c for c in pick] + extra
        print(f"[ltcp] sweep: {len(extra)} wrist variations with a marker in view")
    for k in sorted(why):
        print('   rejected', k, why[k])
    print(f"[ltcp] {len(cands)} candidate poses, picked {len(pick)}")
    for c in pick:
        print(f"  tilt {c['tilt']:2d} dir {c['phi']:3d}  xy {c['xy']} h {c['h']}  px {c['uv']}  q {c['q']}")
    # route over a graph: nodes = standby + picked poses, edge = a clear joint-space leg. Visit greedily (nearest
    # reachable first, travelling through already visited nodes when needed), then return to standby the same way
    import heapq
    nodes = [sb] + [np.array(c["q"]) for c in pick]
    n = len(nodes)
    edge = {(i, j): AC.check_path(nodes[i], nodes[j], margin=MARGIN)[0] for i in range(n) for j in range(n) if i < j}
    clear = lambda i, j: i != j and edge[(min(i, j), max(i, j))]
    w = lambda i, j: float(np.abs(nodes[i] - nodes[j]).max())

    def shortest(src, dst):
        dist, prev, h = {src: 0.0}, {}, [(0.0, src)]
        while h:
            d, u = heapq.heappop(h)
            if u == dst:
                break
            for v in range(n):
                if clear(u, v) and d + w(u, v) < dist.get(v, 1e9):
                    dist[v] = d + w(u, v); prev[v] = u; heapq.heappush(h, (dist[v], v))
        if dst not in dist:
            return None
        p = [dst]
        while p[-1] != src:
            p.append(prev[p[-1]])
        return p[::-1]
    order, cur, todo = [0], 0, set(range(1, n))
    while todo:
        best = None
        for t in todo:
            sp = shortest(cur, t)
            if sp and (best is None or sum(w(a, b) for a, b in zip(sp[:-1], sp[1:])) < best[0]):
                best = (sum(w(a, b) for a, b in zip(sp[:-1], sp[1:])), sp)
        if best is None:
            print("[ltcp] unreachable poses dropped:", sorted(todo)); break
        order += best[1][1:]; cur = best[1][-1]; todo -= set(best[1])
    back = shortest(cur, 0) or [cur, 0]
    order += back[1:]
    route = [home, up] + [nodes[i] for i in order] + [up, home]
    bad = []
    frames = []
    for a_, b_ in zip(route[:-1], route[1:]):
        ok_, w = AC.check_path(a_, b_, margin=MARGIN)
        if not ok_:
            bad.append(describe(w, MARGIN))
        n = max(2, int(np.abs(b_ - a_).max() / 3))
        for s in np.linspace(0, 1, n):
            qq = a_ + (b_ - a_) * s
            cl = AC.clearances(np.deg2rad(qq))
            frames.append({"q": {"left": [round(float(v), 1) for v in qq], "right": right_home.tolist()},
                           "label": "left fingertip measurement (tilt 0/30/45)",
                           "clear_cm": {k: round(v[0] * 100, 1) for k, v in cl.items()}})
    print("[ltcp] route legs not clear:", bad if bad else "none")
    worst = {}
    for f in frames:
        for k, v in f["clear_cm"].items():
            worst[k] = min(worst.get(k, 99), v)
    print("[ltcp] worst clearance along the route (cm):", worst)
    # handeye_near.py visits the listed poses after raising 8 cm from home; list: standby, poses (standby between when needed), standby
    plist = [q for q in route[2:-2]]  # standby ... standby
    json.dump({"poses": [[round(float(v), 1) for v in q] for q in plist], "meta": pick,
               "note": "2026-11-22 left fingertip measurement at tilted jaws; route checked with grasp/arm_collision.py"},
              open(out_poses, "w"), indent=1)
    json.dump({"frames": frames}, open(out_traj, "w"))
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
