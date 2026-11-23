"""Minimal URDF FK for the left arm + visual mesh surface points (no pinocchio needed; trimesh + numpy only).

Reads joint origins and axes along the chain from urdf/openarm_v10_bimanual.urdf, and returns each link's 4x4 pose in the
URDF root frame (same as ik/openarm_left_ik.py). Mesh paths with the package:// prefix are mapped to the local
openarm_description/meshes (the v10 files have the same coordinates as the URDF's v1.0 assets).
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_URDF = os.path.join(HERE, "..", "urdf", "openarm_v10_bimanual.urdf")
DEFAULT_MESHES = os.path.expanduser("~/projects/openarm-orboh/openarm_description/meshes")


def _rpy(r, p, y):
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _origin(el):
    T = np.eye(4)
    o = el.find("origin") if el is not None else None
    if o is not None:
        T[:3, 3] = [float(v) for v in o.get("xyz", "0 0 0").split()]
        T[:3, :3] = _rpy(*[float(v) for v in o.get("rpy", "0 0 0").split()])
    return T


def _axis_rot(axis, q):
    a = np.asarray(axis, float); a /= np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    R = np.eye(3) + np.sin(q) * K + (1 - np.cos(q)) * K @ K
    T = np.eye(4); T[:3, :3] = R; return T


class LeftArmFK:
    def __init__(self, urdf=DEFAULT_URDF, side="left"):
        root = ET.parse(urdf).getroot()
        self.joints = {j.get("name"): j for j in root.findall("joint")}
        self.links = {l.get("name"): l for l in root.findall("link")}
        child_to_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
        # chain from link7 back to the root
        chain, link = [], f"openarm_{side}_link7"
        while link in child_to_joint:
            j = child_to_joint[link]; chain.append(j); link = j.find("parent").get("link")
        self.root_link = link
        self.chain = list(reversed(chain))
        self.side = side

    def link_poses(self, q_rad):
        """q_rad: 7 joint angles (openarm_<side>_joint1..7) → {link: 4x4}."""
        qmap = {f"openarm_{self.side}_joint{i+1}": float(q_rad[i]) for i in range(7)}
        T = np.eye(4); out = {self.root_link: T.copy()}
        for j in self.chain:
            T = T @ _origin(j)
            if j.get("type") in ("revolute", "continuous"):
                ax = [float(v) for v in j.find("axis").get("xyz").split()]
                T = T @ _axis_rot(ax, qmap.get(j.get("name"), 0.0))
            out[j.find("child").get("link")] = T.copy()
        return out

    def surface_points(self, links, meshes_dir=DEFAULT_MESHES, n_per_link=1500, seed=0):
        """Visual-mesh surface points for each link, in that link's frame → {link: (N,3)} [m]."""
        import trimesh
        rng = np.random.default_rng(seed); out = {}
        for ln in links:
            pts = []
            for vis in self.links[ln].findall("visual"):
                mesh = vis.find("geometry/mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename"); b = os.path.basename(fn)
                path = os.path.join(meshes_dir, "arm/v10/visual", b) if "/arm/" in fn else os.path.join(meshes_dir, "ee/openarm_hand/visual", b)
                if not os.path.exists(path):
                    continue
                sc = np.array([float(v) for v in mesh.get("scale", "1 1 1").split()])
                m = trimesh.load(path, force="mesh")
                p, _ = trimesh.sample.sample_surface(m, n_per_link, seed=int(rng.integers(1 << 30)))
                p = p * sc
                To = _origin(vis); p = p @ To[:3, :3].T + To[:3, 3]
                pts.append(p)
            if pts:
                out[ln] = np.vstack(pts)
        return out
