#!/usr/bin/env python3
"""cuRobo の左腕の球を、関節の軸に合わせて当てはめ直す（grasp/curobo/openarm_left_umi.yml の collision_spheres を差し替える）。

RobotBuilder の当てはめ（README の build_robot_model）は球が少なく、実物から最大 2 cm はみ出し、別の所は覆えていなかった
（2026-11-22 Munenori と測った: link4 の 95% で 16 mm・link5 の覆い残し 20 mm）。ここでは次のように当てはめる。

- 候補の中心: 各リンクの形の内側の点（ボクセルの中心）と、関節の軸の上の点（親の関節の軸・子の関節の軸・2 つの関節を結ぶ線）
- 半径: 中心から実物の表面までの距離 + BULGE。中心が内側なら、球のはみ出しは BULGE 以下になる
- 選び方: 実物の表面の点を一番多く覆う球から順に選ぶ（関節の軸の上の候補は AXIS_BONUS 倍に数え、軸のまわりに対称に置く）。
  覆えていない表面の点が COVER_TOL より外に残らなくなるまで続ける
- 関節の向き: 子の関節の軸の向きは URDF の joint の origin の rpy で回した axis。指（直動）も同じ扱い

    ~/work/curobo/.venv/bin/python grasp/curobo/fit_spheres.py            # 当てはめて yml を書く（前の yml は .<日時>.yml に残す）
    ~/work/curobo/.venv/bin/python grasp/curobo/fit_spheres.py --dry-run  # 測るだけ
"""
from __future__ import annotations

import argparse
import os
import shutil
import time
import xml.etree.ElementTree as ET
from typing import Dict, List, Tuple

import numpy as np
import trimesh
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
URDF = os.path.join(HERE, "openarm_left_umi.urdf")
YML = os.path.join(HERE, "openarm_left_umi.yml")


def tf_of(o) -> np.ndarray:
    """URDF の origin（xyz・rpy）を 4x4 に。"""
    if o is None:
        return np.eye(4)
    xyz = [float(v) for v in o.get("xyz", "0 0 0").split()]
    rpy = [float(v) for v in o.get("rpy", "0 0 0").split()]
    m = trimesh.transformations.euler_matrix(*rpy, "sxyz")
    m[:3, 3] = xyz
    return m


def load_links(names: List[str]) -> Tuple[Dict[str, trimesh.Trimesh], Dict[str, List[Tuple[np.ndarray, np.ndarray]]]]:
    """リンクの形（リンクの座標）と、そのリンクに乗る関節の軸の線 [(点, 向き)] を読む。"""
    root = ET.parse(URDF).getroot()
    meshes: Dict[str, trimesh.Trimesh] = {}
    for link in root.findall("link"):
        n = link.get("name")
        if n not in names:
            continue
        parts = []
        for c in link.findall("collision"):
            g = c.find("geometry")
            m = tf_of(c.find("origin"))
            if g.find("mesh") is not None:
                e = g.find("mesh")
                part = trimesh.load(os.path.join(HERE, e.get("filename")), force="mesh")
                s = [float(v) for v in e.get("scale", "1 1 1").split()]
                part.apply_transform(m @ np.diag(s + [1.0]))
            elif g.find("box") is not None:
                part = trimesh.creation.box([float(v) for v in g.find("box").get("size").split()])
                part.apply_transform(m)
            else:
                raise ValueError(f"{n}: mesh と box のほかの形は扱わない")
            parts.append(part)
        meshes[n] = trimesh.util.concatenate(parts) if len(parts) > 1 else parts[0]

    axes: Dict[str, List[Tuple[np.ndarray, np.ndarray]]] = {n: [] for n in meshes}
    for j in root.findall("joint"):
        if j.get("type") not in ("revolute", "prismatic", "continuous"):
            continue
        parent, child = j.find("parent").get("link"), j.find("child").get("link")
        a = np.array([float(v) for v in (j.find("axis").get("xyz") if j.find("axis") is not None else "1 0 0").split()])
        t = tf_of(j.find("origin"))
        if child in axes and j.get("type") != "prismatic":   # 親の関節の軸: 子のリンクの原点を通る
            axes[child].append((np.zeros(3), a / np.linalg.norm(a)))
        if parent in axes:
            d = t[:3, :3] @ a
            if j.get("type") != "prismatic":                  # 子の関節の軸: 親のリンクの座標で
                axes[parent].append((t[:3, 3].copy(), d / np.linalg.norm(d)))
            if np.linalg.norm(t[:3, 3]) > 1e-6:               # 2 つの関節を結ぶ線（リンクの骨）
                axes[parent].append((np.zeros(3), t[:3, 3] / np.linalg.norm(t[:3, 3]), np.linalg.norm(t[:3, 3])))
    return meshes, axes


class Solid:
    """リンクの形を、中の詰まったボクセルの塊にした物（誤差は pitch/2 程度）。

    元のメッシュは、中が空洞の殻（体積が負の内側の殻・link3・link5）、重なった複数の部品（link4）、閉じていない形（link3・指）が混ざる。
    当たりに効くのは外の形だけなので、ボクセルにして中を埋め（空洞も埋まる）、距離は距離変換で測る。
    """

    def __init__(self, mesh: trimesh.Trimesh, pitch: float = 0.0015, pad: int = 20):
        from scipy import ndimage
        vox = mesh.voxelized(pitch).fill()
        self.pitch = pitch
        self.mat = np.pad(vox.matrix, pad)
        self.origin = vox.transform[:3, 3] - pad * pitch         # mat[0,0,0] の中心
        # 中の点から外の形の表面までの距離（中心どうしの距離から半マス引く）と、外の点から表面までの距離
        self.d_in = np.clip(ndimage.distance_transform_edt(self.mat) * pitch - pitch / 2, 0, None)
        self.d_out = np.clip(ndimage.distance_transform_edt(~self.mat) * pitch - pitch / 2, 0, None)
        # 表面の点: 詰まったマスと空のマスの間の面の中心
        pts = []
        for ax in range(3):
            for s in (1, -1):
                idx = np.argwhere(self.mat & ~np.roll(self.mat, -s, axis=ax))   # s の向きの隣が空のマス
                off = np.zeros(3); off[ax] = 0.5 * s
                pts.append(self.centers(idx) + off * pitch)
        self.surface = np.vstack(pts)

    def centers(self, idx: np.ndarray) -> np.ndarray:
        return self.origin + idx * self.pitch

    def index(self, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """点のマスの番号と、格子の中かどうか。"""
        idx = np.round((p - self.origin) / self.pitch).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(self.mat.shape)), axis=1)
        return idx, ok

    def inside_depth(self, p: np.ndarray) -> np.ndarray:
        """点が中にあればその深さ [m]、外なら -1。"""
        idx, ok = self.index(p)
        out = -np.ones(len(p))
        sel = np.where(ok)[0]
        ins = self.mat[tuple(idx[sel].T)]
        out[sel[ins]] = self.d_in[tuple(idx[sel[ins]].T)]
        return out

    def outside_dist(self, p: np.ndarray) -> np.ndarray:
        """点が外の形からはみ出す量 [m]（中なら 0・格子の外は大きな値）。"""
        idx, ok = self.index(p)
        out = np.full(len(p), 1.0)
        out[ok] = self.d_out[tuple(idx[ok].T)] * (~self.mat[tuple(idx[ok].T)])
        return out


def fit_link(solid: Solid, lines, bulge: float, cover_tol: float, pitch: float,
             axis_bonus: float, max_spheres: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """1 リンクに球を当てはめる。返り値は (中心 [N,3], 半径 [N], 測った値)。

    中心は形の中の点だけ。半径 = その点の深さ + bulge なので、球のはみ出しは bulge 以下（ボクセルの誤差を除く）。
    """
    step = max(1, int(round(pitch / solid.pitch)))
    sub = solid.mat.copy()
    keep = np.zeros_like(sub)
    keep[::step, ::step, ::step] = True
    inner = solid.centers(np.argwhere(sub & keep))
    # 関節の軸の上の点（形の中にあるものだけ）
    span = float(np.linalg.norm(np.array(solid.mat.shape) * solid.pitch))
    ax_pts = []
    for ln in lines:
        p0, d = ln[0], ln[1]
        ts = np.arange(0.0, ln[2] + 1e-9, pitch / 2) if len(ln) == 3 else np.arange(-span, span, pitch / 2)
        ax_pts.append(p0[None] + ts[:, None] * d[None])
    ax_pts = np.vstack(ax_pts) if ax_pts else np.zeros((0, 3))
    cand = np.vstack([inner, ax_pts])
    is_axis = np.r_[np.zeros(len(inner), bool), np.ones(len(ax_pts), bool)]
    depth = solid.inside_depth(cand)
    ok = depth >= 0
    cand, is_axis, rad = cand[ok], is_axis[ok], depth[ok] + bulge

    surf = solid.surface
    if len(surf) > 8000:
        surf = surf[np.random.default_rng(0).choice(len(surf), 8000, replace=False)]
    covered = np.zeros(len(surf), bool)
    cover = np.zeros((len(cand), len(surf)), bool)
    for s in range(0, len(cand), 512):
        dd = np.linalg.norm(cand[s:s + 512, None] - surf[None], axis=2)
        cover[s:s + 512] = dd <= rad[s:s + 512, None] + cover_tol
    weight = np.where(is_axis, axis_bonus, 1.0)
    chosen: List[int] = []
    while not covered.all() and len(chosen) < max_spheres:
        gain = cover[:, ~covered].sum(1) * weight
        k = int(np.argmax(gain))
        if gain[k] <= 0:
            break
        chosen.append(k)
        covered |= cover[k]
    C, R = cand[chosen], rad[chosen]
    gap = (np.linalg.norm(solid.surface[:, None] - C[None], axis=2) - R[None]).min(1)
    stats = dict(n=len(chosen), n_axis=int(is_axis[chosen].sum()), gap_max=float(gap.max()),
                 uncovered=float(np.mean(gap > cover_tol)), r_max=float(R.max()))
    return C, R, stats


def protrusion(solid: Solid, C: np.ndarray, R: np.ndarray) -> Tuple[float, float]:
    """球の外側の表面が実物からはみ出す量の (95%, 最大) [m]。はみ出していない点は数えない。"""
    u = trimesh.sample.sample_surface_sphere(300)
    pts = np.vstack([c + u * r for c, r in zip(C, R)])
    d = np.linalg.norm(pts[:, None] - C[None], axis=2) - R[None]
    outer = pts[(d < -1e-4).sum(1) == 0]
    dist = solid.outside_dist(outer)
    dist = dist[dist > 0]
    if len(dist) == 0:
        return 0.0, 0.0
    return float(np.percentile(dist, 95)), float(dist.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bulge", type=float, default=0.005, help="球が実物からはみ出してよい量 [m]")
    ap.add_argument("--cover-tol", type=float, default=0.004, help="覆い残してよい量 [m]")
    ap.add_argument("--pitch", type=float, default=0.003, help="候補の中心の間隔 [m]")
    ap.add_argument("--axis-bonus", type=float, default=1.3, help="関節の軸の上の候補を何倍に数えるか")
    ap.add_argument("--max-spheres", type=int, default=80, help="1 リンクの球の数の上限")
    ap.add_argument("--dry-run", action="store_true", help="yml を書かずに測るだけ")
    a = ap.parse_args()

    doc = yaml.safe_load(open(YML))
    kin = doc.get("robot_cfg", doc)["kinematics"]
    names = list(kin["collision_link_names"])
    meshes, axes = load_links(names)
    out: Dict[str, List[Dict]] = {}
    total = 0
    for n in names:
        np.random.seed(0)
        solid = Solid(meshes[n])
        C, R, st = fit_link(solid, axes[n], a.bulge, a.cover_tol, a.pitch, a.axis_bonus, a.max_spheres)
        p95, pmax = protrusion(solid, C, R)
        total += st["n"]
        print(f"{n:28s} 球 {st['n']:3d}（軸の上 {st['n_axis']:2d}）  はみ出し 95% {p95*1e3:4.1f} 最大 {pmax*1e3:4.1f} mm  "
              f"覆い残し {st['uncovered']*100:4.1f}%（最大 {st['gap_max']*1e3:4.1f} mm）  半径の最大 {st['r_max']*1e3:4.1f} mm")
        out[n] = [{"center": [round(float(v), 5) for v in c], "radius": round(float(r), 5)} for c, r in zip(C, R)]
    print(f"球の合計 {total}")
    if a.dry_run:
        return
    bak = YML.replace(".yml", time.strftime(".%Y%m%d_%H%M.yml"))
    shutil.copy2(YML, bak)
    kin["collision_spheres"] = out
    with open(YML, "w") as f:
        yaml.safe_dump(doc, f, sort_keys=False, allow_unicode=True)
    print(f"書いた: {YML}（前の yml は {bak}）")


if __name__ == "__main__":
    main()
