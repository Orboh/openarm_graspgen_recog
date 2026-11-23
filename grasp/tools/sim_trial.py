#!/usr/bin/env python3
"""Render one planned trial as a video (kinematic simulation). **Does not move the arm.**

Plays the same sequence as run_grasp.py on the plan in <cap_dir>/plan.json:
rest -> raise -> home -> pregrasp -> grasp -> close -> lift -> hold -> (drop: carry high -> drop point -> open | put back)
-> home -> rest. Joints move on straight lines at run_grasp's speeds (transit / near). The object is a box (bar size
by default) placed from the capture's object points; it follows the TCP after the close and falls straight down when
the jaws open. Drawing reuses the URDF + collision meshes (openarm_description) like the 11/21 URDF picture.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/sim_trial.py grasp/runs/cap_xxx -o sim.mp4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
import pinocchio as pin  # noqa: E402

from grasp.plan import floor_plane_base  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402

DESC = os.path.expanduser("~/work/openarm_description")
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual_umi.urdf")
TCP_OFF = np.array([0.0, 0.0, 0.2145])
MAP = {"assets/robot/openarm_v1.0/mesh/arm/collision/link7_symp.stl": "assets/end_effector/parallel_link/meshes/collision/link7.stl",
       "assets/robot/openarm_v1.0/mesh/arm/collision/": "assets/arm/openarm_v1.0/meshes/collision/",
       "assets/robot/openarm_v1.0/mesh/body/collision/": "assets/body/v1.0/meshes/collision/"}
REST = [20.3, -3.6, -0.9, 92.8, -2.5, -11.0, 22.7]


def resolve(fn):
    if fn.startswith("package://openarm_orboh_grasp/"):
        return os.path.join(REPO, "grasp", fn.split("openarm_orboh_grasp/")[1])
    rel = fn.split("openarm_description/")[1]
    for k, v in MAP.items():
        if rel == k:
            return os.path.join(DESC, v)
        if k.endswith("/") and rel.startswith(k):
            return os.path.join(DESC, rel.replace(k, v))
    return os.path.join(DESC, rel)


def main(argv=None) -> int:
    import trimesh
    import yaml
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    plt.rcParams["font.family"] = "Noto Sans CJK JP"

    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("-o", "--out", required=True)
    ap.add_argument("--home", default=os.path.join(REPO, "grasp", "config", "home_left_cart-sym-v4.yaml"))
    ap.add_argument("--transit-dps", type=float, default=75.0)
    ap.add_argument("--near-dps", type=float, default=20.0)
    ap.add_argument("--box", default="0.22,0.046,0.03", help="object box length,width,height [m] (default: the aluminium bar)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--hold-home", action="store_true", help="end at the home pose (run_grasp --hold-home) instead of going back to rest")
    a = ap.parse_args(argv)

    plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
    T_bc = np.array(json.load(open(os.path.join(REPO, "grasp", "config", "T_base_cam.json")))["T_base_cam"])
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    ik = OpenArmLeftIK()
    home = yaml.safe_load(open(a.home))
    home = [float(home[f"joint_{i}"]) if isinstance(home, dict) and f"joint_{i}" in home else None for i in range(1, 8)] \
        if isinstance(home, dict) else None
    if home is None or None in home:
        home = [53.9, 0.0, -4.0, 118.7, -1.0, 0.2, -33.0]

    def raised(qd, dz=0.08):
        M = ik.fk(np.deg2rad(qd))
        q, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, dz])), np.deg2rad(qd), position_only=False)
        return np.rad2deg(q).tolist() if ok else list(qd)

    pre, grasp, lift = plan["q_pre_deg"], plan["q_grasp_deg"], plan["q_lift_deg"]
    G_OPEN, G_CLOSE = 0, 1
    seq = [(REST, a.transit_dps, G_OPEN, "休みの姿勢"), (raised(REST), a.transit_dps, G_OPEN, "持ち上げる"),
           (home, a.transit_dps, G_OPEN, "初期姿勢"), (pre, a.transit_dps, G_OPEN, "プレグラスプ"),
           (grasp, a.near_dps, G_OPEN, "近づく"), (grasp, None, G_CLOSE, "閉じる"),
           (lift, a.near_dps, G_CLOSE, "持ち上げ"), (lift, None, G_CLOSE, "保持 2 秒")]
    if plan.get("q_drop_deg"):
        seq += [(plan["q_lift_high_deg"], a.near_dps, G_CLOSE, "まっすぐ高く上げる"), (raised(home), a.near_dps, G_CLOSE, "持ったまま初期姿勢の上へ"),
                (plan["q_drop_carry_deg"], a.near_dps, G_CLOSE, "肩の横へ運ぶ"), (plan["q_drop_deg"], a.near_dps, G_CLOSE, "落とす点"),
                (plan["q_drop_deg"], None, G_OPEN, "開いて落とす"), (plan["q_drop_carry_deg"], a.near_dps, G_OPEN, "離れる")]
    else:
        seq += [(grasp, a.near_dps, G_CLOSE, "下ろす"), (grasp, None, G_OPEN, "放す"), (pre, a.near_dps, G_OPEN, "プレグラスプへ")]
    seq += [(home, a.transit_dps, G_OPEN, "初期姿勢へ")]
    if not a.hold_home:
        seq += [(raised(REST), a.transit_dps, G_OPEN, "休みの上"), (REST, 20.0, G_OPEN, "休みの姿勢")]

    # timeline of (t, q_deg, grip, label)
    frames, q, t = [], np.array(REST, float), 0.0
    dt = 1.0 / a.fps
    for qn, sp, g, label in seq:
        qn = np.array(qn, float)
        dur = 2.0 if label.startswith("保持") else (0.8 if sp is None else max(0.5, float(np.max(np.abs(qn - q))) / sp))
        for k in range(max(1, int(round(dur / dt)))):
            s = (k + 1) / max(1, int(round(dur / dt)))
            frames.append((t, q + (qn - q) * s, g, label)); t += dt
        q = qn

    # object box from the capture
    ob = np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
    L, W, H = (float(v) for v in a.box.split(","))
    xy = ob[:, :2] - ob[:, :2].mean(0)
    ax_l = np.linalg.svd(xy, full_matrices=False)[2][0]
    c_xy = np.array([(ob[:, 0].min() + ob[:, 0].max()) / 2, (ob[:, 1].min() + ob[:, 1].max()) / 2])
    floor_z = lambda x, y: -(n_b[0] * x + n_b[1] * y + d_b) / n_b[2]  # noqa: E731
    Tbox = np.eye(4)
    Tbox[:3, 0] = [ax_l[0], ax_l[1], 0]; Tbox[:3, 2] = [0, 0, 1]; Tbox[:3, 1] = np.cross(Tbox[:3, 2], Tbox[:3, 0])
    Tbox[:3, 3] = [c_xy[0], c_xy[1], floor_z(*c_xy) + H / 2]
    corners = np.array([[sx * L / 2, sy * W / 2, sz * H / 2, 1] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
    faces = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]

    def tcp_T(qd):
        M = ik.fk(np.deg2rad(qd)); T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation + M.rotation @ TCP_OFF
        return T

    # object pose per frame: on the floor -> attached after close -> falls when opened
    box_T, rel, falling, vz = [], None, None, 0.0
    held_low = []
    prev_g = G_OPEN
    for _, qd, g, label in frames:
        if g == G_CLOSE and prev_g == G_OPEN and rel is None:
            rel = np.linalg.inv(tcp_T(qd)) @ Tbox
        if rel is not None and falling is None:
            Tbox = tcp_T(qd) @ rel
            if g == G_OPEN and prev_g == G_CLOSE:
                falling = Tbox.copy()
        elif falling is not None:
            Tf = falling.copy()
            vz += 9.8 * dt
            Tf[2, 3] = max(falling[2, 3] - vz * dt, floor_z(*Tf[:2, 3]) + H / 2)
            falling = Tf; Tbox = Tf
        box_T.append(Tbox.copy()); prev_g = g
        if rel is not None and falling is None:  # held: lowest box corner above the floor
            cz = (Tbox @ corners.T).T[:, :3]
            held_low.append((label, min(float(n_b @ c + d_b) for c in cz)))

    # meshes
    model = pin.buildModelFromUrdf(URDF); data = model.createData()
    links = []
    for link in ET.parse(URDF).getroot().iter("link"):
        name = link.get("name")
        if not model.existFrame(name) or name.startswith("openarm_right"):
            continue
        for c in link.findall("collision"):
            m = c.find("geometry/mesh")
            if m is None or not os.path.exists(resolve(m.get("filename"))):
                continue
            mesh = trimesh.load(resolve(m.get("filename")), force="mesh")
            sc = [float(x) for x in m.get("scale", "1 1 1").split()]
            o = c.find("origin"); To = np.eye(4)
            if o is not None:
                To[:3, :3] = pin.rpy.rpyToMatrix(*[float(x) for x in o.get("rpy", "0 0 0").split()])
                To[:3, 3] = [float(x) for x in o.get("xyz", "0 0 0").split()]
            F = mesh.faces[:: max(1, len(mesh.faces) // 700)]
            col = "tab:orange" if "body" in name else ("tab:green" if ("umi" in name or "finger" in name or "hand" in name) else "silver")
            links.append((model.getFrameId(name), np.c_[mesh.vertices * sc, np.ones(len(mesh.vertices))], To, F, col))
    jidx = [model.joints[model.getJointId(f"openarm_left_joint{i}")].idx_q for i in range(1, 8)]

    import subprocess
    fig = plt.figure(figsize=(12, 6), dpi=80)
    wpx, hpx = fig.canvas.get_width_height()
    writer = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{wpx}x{hpx}",
                               "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", a.out], stdin=subprocess.PIPE)
    views = ((22, -55, "斜め前から"), (88, -90, "上から"))
    trail = []
    for fi, ((t, qd, g, label), Tb) in enumerate(zip(frames, box_T)):
        qq = pin.neutral(model)
        for i, j in enumerate(jidx):
            qq[j] = np.deg2rad(qd[i])
        pin.forwardKinematics(model, data, qq); pin.updateFramePlacements(model, data)
        tris = [((data.oMf[fid].homogeneous @ To @ V.T).T[:, :3][F], col) for fid, V, To, F, col in links]
        box = (Tb @ corners.T).T[:, :3]
        Tt = tcp_T(qd); fl = Tt[:3, 3] - Tt[:3, :3] @ TCP_OFF
        half = 0.055 if g == G_OPEN else 0.024  # jaw tips: 110 mm open / around the 46 mm bar closed
        jaws = [(fl + Tt[:3, 0] * sgn * 0.03, Tt[:3, 3] + Tt[:3, 2] * 0.015 + Tt[:3, 0] * sgn * half) for sgn in (-1, 1)]
        trail.append(Tt[:3, 3])
        fig.clf()
        for k, (el, az, title) in enumerate(views):
            ax = fig.add_subplot(1, 2, k + 1, projection="3d")
            for Vt, col in tris:
                ax.add_collection3d(Poly3DCollection(Vt, facecolor=col, edgecolor="none", alpha=0.6))
            ax.add_collection3d(Poly3DCollection([box[f] for f in faces], facecolor="lightsteelblue", edgecolor="k", lw=0.5))
            for p0, p1 in jaws:
                ax.plot(*np.c_[p0, p1], color="darkgreen", lw=3)
            tr = np.array(trail); ax.plot(*tr.T, color="purple", lw=0.8, ls=":")
            X, Y = np.meshgrid(np.linspace(-0.2, 0.65, 2), np.linspace(-0.4, 0.6, 2))
            ax.plot_surface(X, Y, floor_z(X, Y), color="tan", alpha=0.25)
            ax.set_xlim(-0.2, 0.65); ax.set_ylim(-0.4, 0.6); ax.set_zlim(0.2, 1.05); ax.set_box_aspect((0.85, 1.0, 0.85))
            ax.view_init(el, az); ax.set_title(title); ax.set_xlabel("x 前"); ax.set_ylabel("y 左")
        fig.suptitle(f"t = {t:4.1f} s   {label}   （爪: {'閉' if g == G_CLOSE else '開'}）")
        fig.canvas.draw()
        img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
        writer.stdin.write(np.ascontiguousarray(img).tobytes())
    writer.stdin.close(); writer.wait()
    if held_low:
        lows = {}
        for lb, h in held_low:
            lows[lb] = min(h, lows.get(lb, 9))
        print("[sim] lowest corner of the held object above the floor per phase: " + ", ".join(f"{k} {v * 100:.1f} cm" for k, v in lows.items()))
    print(f"[sim] {len(frames)} frames ({frames[-1][0]:.1f} s) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
