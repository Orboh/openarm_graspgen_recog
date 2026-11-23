#!/usr/bin/env python3
"""Arm-shape collision check without cuRobo (the right arm has no cuRobo model yet). **Does not move the arm.**

2026-11-22 (Sota, right arm): "掴みに行くときにまたクランプの方にぶつかっている。左手のデータを元にクランプも避けるように、
ちゃんと右手も" / "アルミフレームを持った状態で左にアームが当たっている" / "左右に振るときに台車にぶつかっている".
The old right-arm check looked only at 12 points on the arm's centre line against a box around the clamp, so the
arm's thickness, the gripper body/fingers and a held bar were never checked.

What is checked (all in the base frame of the URDF, the same frame as plan.py / T_base_cam):
  arm   links 1..7 of GRASP_ARM: surface points of the URDF collision meshes (grasp/curobo/meshes, the same meshes
        Kitakyushu's cuRobo model was built from), 1 cm apart; the gripper body box and the two fingers
        (grasp/curobo/make_urdf.py's GRIPPER_BODY_*); the right arm's measured base correction (ik.C) is applied.
  held  optional: the bar between the jaws, a 2 cm-radius segment along the jaws' thickness axis (Yg), --length long.
  world the cuboids in GRASP_WORLD_YML (chest camera, camera clamp measured with the LEFT wrist camera on 11/21);
        the box "<this arm>_arm_home" is skipped and the other arm's home box is mirrored in (y -> -y) when it is not
        listed; the robot body mesh (link0 / link1 are allowed to touch it, they are bolted to it).
Each obstacle reports a clearance [m] (negative = inside). check_path() samples a joint-space straight line.

    python grasp/arm_collision.py --q "<deg x7>" [--held 0.28]       # clearances at one pose
    python grasp/arm_collision.py --traj grasp/runs/<cap>/traj.csv    # replay a recorded run
"""
from __future__ import annotations

import os
import struct
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
MESH_DIR = os.path.join(HERE, "curobo", "meshes")
URDF_COLL = os.path.join(REPO, "urdf", "openarm_v10_bimanual_umi.urdf")
GRIPPER_BODY_XYZ, GRIPPER_BODY_SIZE = (0.0, 0.0, 0.107), (0.060, 0.100, 0.035)  # grasp/curobo/make_urdf.py (link7 frame)
FINGER_Z = (0.120, 0.2295)  # finger root .. jaw tip along link7 z (openarm_left_gripper.json jaw_tip_z_in_link7_m)
FINGER_T, FINGER_W = 0.022, 0.038  # finger thickness (closing axis = link7 y) and width (link7 x), plan.py fx / 2*fy
PT_R = 0.007  # each surface sample stands for a ball this big (1 cm spacing)
MARGIN = float(os.environ.get("GRASP_COLL_MARGIN", "0.015"))


def read_stl(path: str) -> np.ndarray:
    """Triangles (N, 3, 3) of a binary or ASCII STL."""
    data = open(path, "rb").read()
    n = struct.unpack("<I", data[80:84])[0] if len(data) >= 84 else 0
    if len(data) == 84 + 50 * n:
        a = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n)
        return a["v"].astype(float)
    vs = [list(map(float, l.split()[1:4])) for l in data.decode(errors="ignore").splitlines() if l.strip().startswith("vertex")]
    return np.array(vs, float).reshape(-1, 3, 3)


def surface_points(tris: np.ndarray, step: float = 0.01) -> np.ndarray:
    """Points on the triangles about `step` apart (area-weighted samples, then one per voxel)."""
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    n = np.maximum(1, np.ceil(area / (step * step) * 4).astype(int))
    rng = np.random.default_rng(0)
    idx = np.repeat(np.arange(len(tris)), n)
    u, v = rng.random(len(idx)), rng.random(len(idx))
    flip = u + v > 1; u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    P = a[idx] + u[:, None] * (b - a)[idx] + v[:, None] * (c - a)[idx]
    P = np.vstack([P, tris.reshape(-1, 3)])
    _, keep = np.unique(np.floor(P / step).astype(int), axis=0, return_index=True)
    return P[keep]


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def box_points(center, size, step=0.01) -> np.ndarray:
    ax = [np.linspace(c - s / 2, c + s / 2, max(2, int(np.ceil(s / step)) + 1)) for c, s in zip(center, size)]
    G = np.stack(np.meshgrid(*ax, indexing="ij"), -1).reshape(-1, 3)
    lo, hi = np.array(center) - np.array(size) / 2, np.array(center) + np.array(size) / 2
    on = np.any(np.isclose(G, lo) | np.isclose(G, hi), axis=1)  # the surface is enough
    return G[on]


def link_points(arm: str, urdf: str = URDF_COLL) -> Dict[str, np.ndarray]:
    """Surface points of each arm link in its own link frame (URDF collision mesh, origin and scale applied)."""
    root = ET.parse(urdf).getroot()
    out = {}
    for l in root.findall("link"):
        name = l.get("name")
        if not any(name == f"openarm_{arm}_link{i}" for i in range(1, 8)):
            continue
        pts = []
        for c in l.findall("collision"):
            m = c.find("geometry/mesh")
            if m is None:
                continue
            f = os.path.join(MESH_DIR, os.path.basename(m.get("filename")))
            if not os.path.exists(f):
                continue
            sc = np.array([float(v) for v in m.get("scale", "1 1 1").split()])
            o = c.find("origin")
            xyz = np.array([float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()])
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            P = surface_points(read_stl(f) * sc)
            pts.append(P @ _rpy(*rpy).T + xyz)
        if pts:
            out[name] = np.vstack(pts)
    return out


class ArmCollision:
    def __init__(self, ik, arm: Optional[str] = None, world_yml: Optional[str] = None, floor_z: Optional[float] = None):
        import pinocchio as pin
        import yaml
        from scipy.spatial import cKDTree
        self.pin, self.ik = pin, ik
        self.arm = arm or os.environ.get("GRASP_ARM", "left")
        self.C = getattr(ik, "C", None)
        lp = link_points(self.arm)
        self.links = [(ik.model.getFrameId(n), P) for n, P in lp.items() if ik.model.existFrame(n)]
        self.l7 = ik.model.getFrameId(f"openarm_{self.arm}_link7")
        self.hand_closed = box_points(GRIPPER_BODY_XYZ, GRIPPER_BODY_SIZE)
        world_yml = world_yml or os.environ.get("GRASP_WORLD_YML")
        w = (yaml.safe_load(open(world_yml)) or {}) if world_yml else {}
        other = "left" if self.arm == "right" else "right"
        cubs = {}
        allcub = dict(w.get("cuboid") or {}); allcub.update(w.get("cuboid_extra") or {})  # cuboid_extra: arm_collision only (not cuRobo)
        for n, c in allcub.items():
            if n == f"{self.arm}_arm_home":
                continue  # this arm itself
            cubs[n] = (np.array(c["dims"], float), np.array(c["pose"], float))
        if f"{other}_arm_home" not in cubs and f"{self.arm}_arm_home" in allcub:
            c = allcub[f"{self.arm}_arm_home"]; p = np.array(c["pose"], float); p[1] = -p[1]
            cubs[f"{other}_arm_home"] = (np.array(c["dims"], float), p)  # the other arm parked at its (mirrored) home
        self.cubs = cubs
        self.meshes = []
        for n, c in (w.get("mesh") or {}).items():
            f = os.path.join(REPO, c["file"])
            if not os.path.exists(f):
                continue
            P = surface_points(read_stl(f) * float(c.get("scale", 1.0)))
            pose = np.array(c.get("pose", [0, 0, 0, 1, 0, 0, 0]), float)
            R = pin.Quaternion(pose[3], pose[4], pose[5], pose[6]).toRotationMatrix()
            self.meshes.append((n, cKDTree(P @ R.T + pose[:3])))
        self.floor_z = floor_z

    def points(self, q_rad, held: Optional[float] = None, open_m: float = 0.05):
        """Arm points in the base frame (N, 3) and their labels."""
        pin, ik = self.pin, self.ik
        pin.framesForwardKinematics(ik.model, ik.data, np.asarray(q_rad, float) + getattr(ik, "q_off", 0.0))  # left: joint offsets
        Cm = np.eye(4)
        if self.C is not None:
            Cm[:3, :3], Cm[:3, 3] = self.C.rotation, self.C.translation
        P, lab = [], []
        for fid, L in self.links:
            M = ik.data.oMf[fid]; T = Cm @ M.homogeneous
            P.append(L @ T[:3, :3].T + T[:3, 3]); lab += [ik.model.frames[fid].name] * len(L)
        T7 = Cm @ ik.data.oMf[self.l7].homogeneous
        g = [self.hand_closed]
        for s in (-1, 1):  # fingers at the given opening (closing axis = link7 y)
            g.append(box_points((0.0, s * (open_m / 2 + FINGER_T / 2), sum(FINGER_Z) / 2), (FINGER_W, FINGER_T, FINGER_Z[1] - FINGER_Z[0])))
        G = np.vstack(g); P.append(G @ T7[:3, :3].T + T7[:3, 3]); lab += ["gripper"] * len(G)
        if held:
            # the bar lies across the jaws: along link7 x (= gripper Yg), centred on the TCP (link7 z 0.2145)
            s = np.linspace(-held / 2, held / 2, max(2, int(held / 0.01)))
            H = np.c_[s, np.zeros_like(s), np.full_like(s, 0.2145)]
            P.append(H @ T7[:3, :3].T + T7[:3, 3]); lab += ["held"] * len(H)
        return np.vstack(P), np.array(lab)

    def clearances(self, q_rad, held: Optional[float] = None, open_m: float = 0.05) -> Dict[str, tuple]:
        """{obstacle: (clearance m, arm part)} for one pose. Held-bar points get a 2 cm radius, the rest PT_R."""
        P, lab = self.points(q_rad, held, open_m)
        rad = np.where(lab == "held", 0.02, PT_R)
        out = {}
        for n, (dims, pose) in self.cubs.items():
            R = self.pin.Quaternion(pose[3], pose[4], pose[5], pose[6]).toRotationMatrix()
            loc = (P - pose[:3]) @ R
            d = np.abs(loc) - dims / 2
            dist = np.linalg.norm(np.maximum(d, 0), axis=1) + np.minimum(d.max(1), 0) - rad
            k = int(np.argmin(dist)); out[n] = (float(dist[k]), str(lab[k]))
        for n, tree in self.meshes:
            ok = ~np.isin(lab, [f"openarm_{self.arm}_link1"])  # link0 is bolted to the body; link1 turns right next to it
            d, _ = tree.query(P[ok]); d = d - rad[ok]
            k = int(np.argmin(d)); out[n] = (float(d[k]), str(lab[ok][k]))
        return out

    def check_path(self, q_from_deg, q_to_deg, held: Optional[float] = None, open_m: float = 0.05, n: Optional[int] = None,
                   margin: float = MARGIN):
        """Worst clearance along a joint-space straight line. Returns (ok, worst {name: (clear, part, s)}).
        An end pose that already sits closer than `margin` to something (e.g. the home pose 1 cm above the cart tray)
        lowers the margin for that obstacle to (that end's clearance - 3 mm): the leg may not get closer than its ends."""
        a, b = np.asarray(q_from_deg, float), np.asarray(q_to_deg, float)
        n = n or max(8, int(np.ceil(np.max(np.abs(b - a)) / 2.0)) + 1)  # every ~2 deg
        worst = {}
        ends = {}
        for s in np.linspace(0, 1, n):
            cl = self.clearances(np.deg2rad(a + (b - a) * s), held, open_m)
            if s in (0.0, 1.0):
                for k, (c, _) in cl.items():
                    ends[k] = min(ends.get(k, 9.0), c)
            for k, (c, part) in cl.items():
                if k not in worst or c < worst[k][0]:
                    worst[k] = (c, part, float(s))
        self.last_margins = {k: min(margin, ends.get(k, 9.0) - 0.003) for k in worst}
        return all(v[0] >= self.last_margins[k] for k, v in worst.items()), worst


def describe(worst: Dict[str, tuple], margin: float = MARGIN) -> str:
    bad = {k: v for k, v in worst.items() if v[0] < margin}
    if not bad:
        bad = dict(sorted(worst.items(), key=lambda kv: kv[1][0])[:1])
    return "・".join(f"{k} に {v[1]} が {v[0]*100:+.1f} cm（s={v[2]:.2f}）" for k, v in sorted(bad.items(), key=lambda kv: kv[1][0]))


def main(argv=None) -> int:
    import argparse
    import sys
    sys.path.insert(0, REPO)
    from ik.openarm_right_ik import arm_ik_class
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--q", default=None)
    ap.add_argument("--traj", default=None)
    ap.add_argument("--held", type=float, default=None)
    a = ap.parse_args(argv)
    ik = arm_ik_class()()
    ac = ArmCollision(ik)
    print(f"[coll] arm {ac.arm}: {sum(len(P) for _, P in ac.links)} link points; obstacles {list(ac.cubs)} + {[n for n, _ in ac.meshes]}")
    if a.q:
        for k, v in ac.clearances(np.deg2rad([float(x) for x in a.q.split(",")]), a.held).items():
            print(f"  {k:16s} {v[0]*100:+6.1f} cm  ({v[1]})")
    if a.traj:
        import csv
        rows = list(csv.DictReader(open(a.traj)))
        jk = [f"joint_{i}" for i in range(1, 8)]
        keys = [k for k in rows[0] if any(k.startswith(j) for j in jk)]
        print("[coll] columns:", rows[0].keys())
        for r in rows[::5]:
            q = [float(r[k]) for k in keys[:7]]
            c = ac.clearances(np.deg2rad(q), a.held)
            m = min(c.items(), key=lambda kv: kv[1][0])
            print(f"  {r.get('phase', r.get('t', ''))[:28]:28s} worst {m[0]} {m[1][0]*100:+.1f} cm ({m[1][1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
