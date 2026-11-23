#!/usr/bin/env python3
"""Step 5: grasp candidates (camera frame) → arm base frame → IK. **Does not move the arm.**

1. Object point cloud (object.npy from thor_snap) → GraspGenX (the measured OpenArm gripper)
2. T_base_cam (grasp/config/T_base_cam.json) → base frame
3. GraspGenX gripper frame (origin = flange, +Z approach, +X closing) → link7 target pose.
   Fingers extend along link7 +z and close along y, so in link7: Xg=+y, Yg=-x, Zg=+z, origin = flange z=0.0895
4. Keep only grasps from above (approach within --max-tilt deg of straight down). Try them in score order, and take the first
   one where all three poses solve: pregrasp (10 cm back along the approach) / grasp / lift (10 cm straight up).
5. Write plan.json (joint angles in lerobot degrees) and plan.png (3D view)

    PYTHONPATH=.:~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/plan.py grasp/runs/cap_xxx --q "-17.5,2.1,4,28.6,-8.7,-15.5,-27.7"
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pinocchio as pin  # noqa: E402

from grasp.curobo import client as curobo  # noqa: E402  GRASP_CUROBO があるときだけ使う
from grasp.infer import infer_grasps, infer_graspnet, load_sweep, scene_cloud  # noqa: E402
from ik.openarm_left_ik import OpenArmLeftIK  # noqa: E402
from ik.openarm_right_ik import arm_ik_class, mirror_q  # noqa: E402
ARM = os.environ.get("GRASP_ARM", "left")  # 2026-11-21: right arm for bars the left can't turn its wrist to

HERE = os.path.dirname(os.path.abspath(__file__))
PREFER_LEFT = "-17.7,2.0,4.7,28.1,-8.3,-15.5,-28.9"
CFG = json.load(open(os.environ.get("GRASP_GRIPPER_JSON", os.path.join(HERE, "config", "openarm_left_gripper.json"))))  # 置き場の爪は grasp/config/sites/
SWEEP_JSON = os.environ.get("GRASP_SWEEP_JSON", os.path.join(HERE, "config", "openarm_custom_hand_sweep.json"))
T_L7_G = np.eye(4)
T_L7_G[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # columns: Xg=+y, Yg=-x, Zg=+z
T_L7_G[:3, 3] = [0, 0, CFG["hand_mount_face_z_in_link7_m"]]
if "tcp_in_link7_taught_m" in CFG and ARM == "left":  # taught on the left hand only  # shift the gripper base by the offset between the taught TCP and the model TCP
    _tcp_model = np.array([0, 0, CFG["hand_mount_face_z_in_link7_m"] + CFG["flange_to_tip_m"] - 0.015])
    T_L7_G[:3, 3] += np.array(CFG["tcp_in_link7_taught_m"]) - _tcp_model
FLIP = np.diag([-1.0, -1.0, 1.0, 1.0])  # 180 deg about the approach axis (symmetric gripper)
# 腕が入ってはいけない箱（2026-11-21 北九州）: 45° 傾けてロボット側からコップの縁を挟む計画で、手首と肘のあたりが胸のカメラと
# その取り付け部（本体のフレーム）に当たった。GRASP_KEEPOUT_BOXES="x0,x1,y0,y1,z0,z1;..."（腕の付け根の座標 [m]・腕の太さの分を
# 広げておく）。関節 3〜7・関節の間の中点・指先の中心・爪の先のどれかが箱に入る解は捨てる。無ければ確かめない（東京は無し）。
# cuRobo で本体の形を入れるまでの安全の線（GRASP_CUROBO があれば cuRobo の球でも見る・grasp/curobo/）
KEEPOUT = [np.array([float(v) for v in b.split(",")]) for b in os.environ.get("GRASP_KEEPOUT_BOXES", "").split(";") if b.strip()]


def arm_points(ik, q) -> np.ndarray:
    """関節 3〜7 の原点・その間の中点・指先の中心・爪の先（腕の付け根の座標・(N,3)）。"""
    import pinocchio as pin_
    pin_.forwardKinematics(ik.model, ik.data, np.asarray(q, float))
    js = [ik.data.oMi[ik.model.getJointId(f"openarm_{ARM}_joint{j}")].translation.copy() for j in range(3, 8)]
    pts = js + [(a + b) / 2 for a, b in zip(js[:-1], js[1:])]
    MG = ik.data.oMi[ik.model.getJointId(f"openarm_{ARM}_joint7")].act(pin_.SE3(T_L7_G[:3, :3], T_L7_G[:3, 3]))
    for z in (CFG["flange_to_tip_m"] - 0.015, CFG["flange_to_tip_m"]):
        pts.append(MG.act(np.array([0, 0, z])))
    return np.array(pts)


def in_keepout(ik, q) -> bool:
    """腕の点のどれかが、入ってはいけない箱に入るか。"""
    if not KEEPOUT:
        return False
    P = arm_points(ik, q)
    for b in KEEPOUT:
        if np.any((P[:, 0] >= b[0]) & (P[:, 0] <= b[1]) & (P[:, 1] >= b[2]) & (P[:, 1] <= b[3]) & (P[:, 2] >= b[4]) & (P[:, 2] <= b[5])):
            return True
    return False


FLOOR_FILE = os.environ.get("GRASP_FLOOR_PLANE", os.path.join(HERE, "config", "floor_plane.json"))


def floor_plane_base(cap_dir, T_bc):
    """Floor plane → base frame. Returns (unit normal pointing up, d) with height = n·p + d.
    2026-11-21 23:15 (Sota: 高さもちゃんと一緒に): one fixed floor (config/floor_plane.json, median of 22 captures, 0.8 mm
    spread) instead of a fresh RANSAC per capture, so every capture / plan uses the same floor. GRASP_FLOOR_PLANE=fit
    (or a missing file) fits the capture again, e.g. after moving the cart or recalibrating the camera."""
    # 2026-11-22 16:45: the fixed plane is off by up to +-7 mm across the near-vertical camera's view; a flat bar on
    # the sack sat 6 mm below it and the jaws closed above it. decide.py measures the ground right around the target and
    # passes it here (GRASP_FLOOR_SHIFT, m, clamped there): height = n.p + d - shift
    shift = float(os.environ.get("GRASP_FLOOR_SHIFT", "0") or 0)
    if FLOOR_FILE != "fit" and os.path.exists(FLOOR_FILE):
        f = json.load(open(FLOOR_FILE))
        return np.array(f["n"], float), float(f["d"]) - shift
    import open3d as o3d
    from grasp.capture import depth_to_points
    K = np.array(json.load(open(os.path.join(cap_dir, "intrinsics.json")))["K"])
    pts, _ = depth_to_points(np.load(os.path.join(cap_dir, "depth.npy")), K, zmax=1.0)
    pb = pts @ T_bc[:3, :3].T + T_bc[:3, 3]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pb.astype(float))).voxel_down_sample(0.005)
    o3d.utility.random.seed(0)  # same capture -> same floor (11/21: re-running one capture flipped plan / no plan)
    (a, b, c, d), _ = pc.segment_plane(0.008, 3, 2000)
    n = np.array([a, b, c]); nn = np.linalg.norm(n)
    n, d = n / nn, d / nn
    if n[2] < 0:
        n, d = -n, -d
    return n, d - shift


def se3(T):
    return pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())


def main(argv=None) -> int:
    """Plan one grasp. With object_alt_*.npy from thor_snap --perceive (a pile with several aluminium frames), a target
    with no reachable grasp is swapped for the next one (highest first) and planned again."""
    import glob
    argv = list(sys.argv[1:] if argv is None else argv)
    cap_dir = argv[0] if argv and not argv[0].startswith("-") else None
    if cap_dir:
        for f in glob.glob(os.path.join(cap_dir, "plan_stats.json")) + glob.glob(os.path.join(cap_dir, "plan_funnel_*.jpg")):
            os.remove(f)
    rc = _main(argv)
    if rc == 0 or cap_dir is None:
        return rc
    for alt in sorted(glob.glob(os.path.join(cap_dir, "object_alt_*.npy"))):
        print(f"[plan] no grasp on this frame; trying the next aluminium frame ({os.path.basename(alt)})", flush=True)
        os.replace(os.path.join(cap_dir, "object.npy"), os.path.join(cap_dir, "object_skipped_%s.npy" % os.path.basename(alt)[-5]))
        os.replace(alt, os.path.join(cap_dir, "object.npy"))
        for cache in ("grasps_cam.npz", "grasps_graspnet.npz"):
            if os.path.exists(os.path.join(cap_dir, cache)):
                os.remove(os.path.join(cap_dir, cache))
        rc = _main(argv)
        if rc == 0:
            return rc
    return rc


def _main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True, help="current joint angles (lerobot degrees, 7 values comma-separated)")
    ap.add_argument("--max-tilt", type=float, default=60.0, help="allowed tilt of the approach from straight down [deg] (on the floor, straight-down only reaches up to x≈0.35 m)")
    ap.add_argument("--pregrasp", type=float, default=0.10)
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--floor-clear", type=float, default=0.012, help="minimum height of the lowest jaw-tip corner above the floor [m]")
    ap.add_argument("--center-tol", type=float, default=0.02, help="max horizontal distance from the grasp centre to the object's centre [m]")
    ap.add_argument("--max-branch-deg", type=float, default=45.0, help="max joint change between grasp and pre/lift [deg]")
    ap.add_argument("--wrist-near", default=None,
                    help="j6,j7,tol (lerobot deg): only use grasps whose J6/J7 at the grasp are within tol of these. The taught TCP "
                         "(config tcp_in_link7_taught_m) was measured at J6 +36 / J7 +25; far from it the jaw position error grows (11/20 trial 3)")
    ap.add_argument("--look-back", type=float, default=0.0,
                    help="also require a 'look' pose this far behind the pregrasp along the approach to solve (wrist-camera alignment needs it) [m]")
    ap.add_argument("--obstacle-margin", type=float, default=0.010, help="clearance around the jaw boxes for --avoid-obstacles [m]")
    ap.add_argument("--open-mm", type=float, default=93.0, help="jaw tip gap when open [mm]. Smaller fits between a bar and its "
                    "neighbours in a pile; plan.json then carries gripper_open_deg for run_grasp.py (93 mm = lerobot -60 deg)")
    ap.add_argument("--finger-contact", action="store_true", help="with --avoid-obstacles: let the outside of the fingers touch "
                    "neighbours (they get pushed aside on the way down, 20 deg/s); only things inside the gap reject a grasp")
    ap.add_argument("--gap-soft-below", type=float, default=None, help="with --clear-gap: inside the jaw gap only things lower than "
                    "(bar middle height - this) are soft; e.g. 0.01 (unknown objects in the gap block the grasp)")
    ap.add_argument("--also-graspgenx", action="store_true", help="with --bar-grasps: also ask GraspGenX and pool both kinds of grasps")
    ap.add_argument("--bar-pitch", type=int, default=20, help="--bar-grasps: tilt along the bar up to this [deg] (10 deg steps)")
    ap.add_argument("--bar-grasps", action="store_true", help="instead of GraspGenX: jaws across the bar's long axis, sampled "
                    "along it with roll/pitch (use with --shallow for piles)")
    ap.add_argument("--seen-keepaway", type=float, default=0.12, help="with GRASP_CUROBO and --avoid-obstacles: camera-seen points "
                    "within this xy distance of the target are left to the point-wise jaw checks, not cuRobo boxes [m]")
    ap.add_argument("--shallow", type=float, default=None, help="grip only the top: jaw tips this far [m] below the bar's "
                    "top between the jaws (e.g. 0.015); lower neighbours stay under the fingertips")
    ap.add_argument("--clear-gap", action="store_true", help="with --avoid-obstacles: also reject grasps with other things inside the closing gap")
    ap.add_argument("--soft-below", type=float, default=None,
                    help="with --avoid-obstacles: ignore obstacle points lower than the target top minus this [m] (soft pile under the bar)")
    ap.add_argument("--avoid-obstacles", action="store_true",
                    help="drop grasps whose open jaws (swept from the pregrasp) would hit other above-floor points")
    ap.add_argument("--jaw-clear", type=float, default=None,
                    help="開いた爪の内側と物のすき間が、閉じる向きの両側ともこれ以上ある候補だけ使う [m]（爪の全開が狭いとき。北九州は 0.012）")
    ap.add_argument("--thick-tol", type=float, default=None, help="max object-centre offset along the jaw thickness (Yg) [m]; also bounds the closing-axis offset by --center-tol")
    ap.add_argument("--push-depth", type=float, default=0.0, help="move every grasp this much further along its approach before checks/IK [m]")
    ap.add_argument("--extra-depth", type=float, default=0.0,
                    help="also require the grasp moved this much further along the approach to be IK-solvable [m] "
                         "(the real jaws stop 3-5 cm short of the model; the wrist servo needs the margin to go deeper)")
    ap.add_argument("--min-tilt", type=float, default=0.0, help="only grasps tilted at least this much [deg] (2026-11-22: the "
                    "left TCP survey probes the same bar at several wrist tilts)")
    ap.add_argument("--tip-min", type=float, default=0.012, help="--bar-grasps: lowest jaw-tip height above the floor [m]. "
                    "2026-11-22: a 2 cm bar with tips at 12 mm left ~8 mm of overlap and the jaws closed above it; 0.005 on the soft sack")
    ap.add_argument("--ik-seeds", type=int, default=4, help="random IK start postures per pose (2026-11-22: tilted bar grasps "
                    "needed ~40 for the right arm)")
    ap.add_argument("--raise-to-clear", type=float, default=0.0,
                    help="back grasps off along the approach by up to this much [m] so the jaw tips clear the floor (low objects)")
    ap.add_argument("--min-score", type=float, default=0.0, help="only try candidates with at least this GraspGenX score")
    ap.add_argument("--max-try", type=int, default=120, help="max number of candidates to run IK on")
    ap.add_argument("--rounds", type=int, default=5, help="number of GraspGenX calls to pool")
    ap.add_argument("--per-call", type=int, default=None,
                    help="GraspGenX に 1 回で出させる候補の数（無しなら 100）。--rounds 1 --per-call 1500 は 100×15 回と同じ数で約 1.2 秒速い")
    ap.add_argument("--model", choices=["graspgenx", "graspnet"], default="graspgenx",
                    help="grasp model: GraspGenX (object cloud, port 5556) or GraspNet-baseline (scene cloud, port 5557; step 7 comparison)")
    ap.add_argument("--reinfer", action="store_true", help="re-run GraspGenX even if a cached result exists")
    ap.add_argument("--no-render", action="store_true", help="計画の図（plan.png）を描かない（計算には使わない・2〜3 秒減る）")
    ap.add_argument("--prefer", default=PREFER_LEFT,
                    help="preferred posture (lerobot degrees). Among IK solutions, pick the one closest to this (avoids odd branches like an elbow flung outward)")
    a = ap.parse_args(argv)

    obj = np.load(os.path.join(a.cap_dir, "object.npy"))
    if os.environ.get("GRASP_SITE") and "GRASP_T_BASE_CAM" not in os.environ:  # 東京のハンドアイで計画しない
        raise SystemExit("[plan] GRASP_T_BASE_CAM が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。その置き場のハンドアイを取ってから")
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(HERE, "config", "T_base_cam.json"))))["T_base_cam"])
    cache = os.path.join(a.cap_dir, "grasps_cam.npz" if a.model == "graspgenx" else "grasps_graspnet.npz")
    import hashlib
    obj_key = hashlib.sha1(np.ascontiguousarray(obj).tobytes()).hexdigest()[:16]
    z = np.load(cache) if os.path.exists(cache) and not a.reinfer and not a.bar_grasps else None
    if a.bar_grasps:
        # 2026-11-21 (Sota B: 上だけ浅く掴む): GraspGenX gave only a few orientations per bar, mostly ones the arm can't
        # reach near the robot. For a bar the useful grasps are simple: jaws close across the bar's long axis; the
        # approach may roll about the bar (up to --max-tilt) and pitch along it a little. Sampled every 1 cm along the
        # bar; depth is then set by --shallow. Score = 1 near the bar middle, falling off toward the ends.
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        nfl, dfl = floor_plane_base(a.cap_dir, T_bc)
        mu = ob_b[:, :2].mean(0)
        _, V = np.linalg.eigh(np.cov((ob_b[:, :2] - mu).T)); ax = np.r_[V[:, 1], 0.0]; nx = np.r_[-V[1, 1], V[0, 1], 0.0]
        al = (ob_b[:, :2] - mu) @ ax[:2]
        Gl, Sl = [], []
        tcpz = CFG["flange_to_tip_m"] - 0.015
        for s_ in np.arange(al.min() + 0.015, al.max() - 0.014, 0.01):
            sel = np.abs(al - s_) < 0.008
            if sel.sum() < 15:
                continue
            c = ob_b[sel].mean(0)
            # where the fingertips go (2026-11-22, part of making the grasp, not a later correction): a bar is held
            # from its top down to 25 mm, but the tips stay 12 mm above the floor (a pipe lying on the bare floor)
            h_top = float(np.percentile(ob_b[sel] @ nfl + dfl, 90))
            h_tip = max(a.tip_min, h_top - 0.025)
            tip_pt = c + (h_tip - float(c @ nfl + dfl)) * nfl
            for roll in range(-60, 61, 10):
                for pitch in range(-a.bar_pitch, a.bar_pitch + 1, 10):
                    zr = np.array([0, 0, -1.0])
                    Rr_ = pin.AngleAxis(np.deg2rad(roll), ax).toRotationMatrix() @ pin.AngleAxis(np.deg2rad(pitch), nx).toRotationMatrix()
                    zz = Rr_ @ zr; xx = nx - np.dot(nx, zz) * zz; xx /= np.linalg.norm(xx)
                    Tg = np.eye(4); Tg[:3, :3] = np.c_[xx, np.cross(zz, xx), zz]; Tg[:3, 3] = tip_pt - CFG["flange_to_tip_m"] * zz
                    Gl.append(np.linalg.inv(T_bc) @ Tg)
                    Sl.append(1.0 - 0.3 * abs(s_) / max(1e-6, np.ptp(al) / 2) - 0.002 * (abs(roll) + abs(pitch)))
        G, S = np.array(Gl), np.array(Sl)
        print(f"[plan] bar grasps: {len(G)} (along the bar every 1 cm, roll ±60°, pitch ±20°)")
        SRC = np.array(["bar"] * len(G), dtype=object)
        if a.also_graspgenx:
            # 2026-11-21 (Sota: GraspGen も入れてトライ): pool GraspGenX's own grasps with the bar grasps; every filter,
            # IK and cuRobo check below treats them the same, and plan_stats.json says which kind won
            Gs, Ss = [], []
            for k in range(a.rounds):
                g, sc = infer_grasps(obj, sweep=load_sweep(SWEEP_JSON), host=a.host, num_grasps=400, topk=200, seed=k,
                                     sweep_num=a.per_call)
                Gs.append(g); Ss.append(sc)
            Gx, Sx = np.concatenate(Gs), np.concatenate(Ss)
            G, S = np.concatenate([G, Gx]), np.concatenate([S, Sx])
            SRC = np.concatenate([SRC, np.array(["graspgenx"] * len(Gx), dtype=object)])
            print(f"[plan] + GraspGenX grasps: {len(Gx)}")
    elif z is not None and "obj_key" in z.files and str(z["obj_key"]) == obj_key:  # 11/21: a stale cache from another bar
        G, S = z["G"], z["S"]; print(f"[plan] reusing {cache}")              # (fallback target) planned the wrong bar
    elif a.model == "graspnet":
        scene = scene_cloud(a.cap_dir, obj)
        Gs, Ss = [], []
        for k in range(a.rounds):  # 20k-point sampling differs per seed, so pool a few calls like GraspGenX
            g, sc = infer_graspnet(scene, host=a.host, seed=k)
            Gs.append(g); Ss.append(sc)
        G, S = np.concatenate(Gs), np.concatenate(Ss)
        o = np.argsort(-S); G, S = G[o], S[o]
        np.savez(cache, G=G, S=S, obj_key=obj_key)
    else:
        Gs, Ss = [], []
        for k in range(a.rounds):  # diffusion samples differ each call, so pool several calls
            g, sc = infer_grasps(obj, sweep=load_sweep(SWEEP_JSON),
                                 host=a.host, num_grasps=400, topk=200, seed=k, sweep_num=a.per_call)
            Gs.append(g); Ss.append(sc)
        G, S = np.concatenate(Gs), np.concatenate(Ss)
        np.savez(cache, G=G, S=S, obj_key=obj_key)
    Gb = np.einsum("ij,njk->nik", T_bc, G)  # grasps in base frame
    if not a.bar_grasps:
        SRC = np.array([a.model] * len(G), dtype=object)
    if a.push_depth:  # grasp deeper along the approach (11/20 iter_034: jaws closed near the block's top edge and slipped)
        Gb[:, :3, 3] += a.push_depth * Gb[:, :3, 2]
    down = np.array([0, 0, -1.0])
    tilt = np.degrees(np.arccos(np.clip(Gb[:, :3, 2] @ down, -1, 1)))
    # Floor clearance: in the first real run the lower jaw corner hit the floor (and pushed the robot/cart).
    # Fit the floor plane in the base frame from the capture, and drop grasps where the lowest jaw-tip corner is less than --floor-clear above the floor.
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    if curobo.enabled():  # カメラで見た障害物（机の上の箱など）を cuRobo に入れる（grasp/curobo/seen.py・2026-11-21）
        from grasp.curobo import seen
        P_obs = seen.seen_points(a.cap_dir, T_bc, obj, n_b, d_b)
        if a.avoid_obstacles and len(P_obs):
            # 2026-11-21 Tokyo pile: the seen boxes are coarse bounding boxes of the whole pile (sack + bars + wood), so
            # every grasp into the pile "hits" them. Next to the target the jaws are already checked point by point
            # (--avoid-obstacles / --clear-gap / rigid.png); give cuRobo only what is farther than --seen-keepaway from
            # the target, so it still guards the arm's way in and out (and the body / camera / floor boxes).
            ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
            from scipy.spatial import cKDTree
            dxy, _ = cKDTree(ob_b[:, :2]).query(np.asarray(P_obs)[:, :2])
            P_obs = np.asarray(P_obs)[dxy > a.seen_keepaway]
        np.save(os.path.join(a.cap_dir, seen.POINTS_FILE), P_obs)
        arm_qs = seen.capture_arm_qs()
        fz = float(os.environ.get("GRASP_FLOOR_Z", -d_b / n_b[2]))
        r = curobo.set_obstacle_points(P_obs, arm_qs, T_bc[:3, 3], fz)
        json.dump({"cam_pos": T_bc[:3, 3].tolist(), "floor_z": fz, "arm_qs": arm_qs, "boxes": r["boxes"]},
                  open(os.path.join(a.cap_dir, seen.BOXES_FILE), "w"), indent=1)
        print(f"[plan] cuRobo: カメラで見た障害物の箱 {len(r['boxes'])} 個（点 {r['n_in']} → 腕を消して {r['n_after_arm']}）"
              + "".join(f" {b['name']} 中心 {np.round(b['pose'][:3], 2).tolist()} 大きさ {np.round(b['dims'], 2).tolist()}" for b in r["boxes"]))
    tip_off = CFG["flange_to_tip_m"]
    jaw_out = min(0.058, a.open_mm / 2000.0 + 0.022)  # outer face of a finger at the commanded opening
    def lowest_corner(Tg):
        tip = Tg[:3, 3] + tip_off * Tg[:3, 2]
        cs = [tip + sx * jaw_out * Tg[:3, 0] + sy * 0.024 * Tg[:3, 1] for sx in (-1, 1) for sy in (-1, 1)]
        return min(float(n_b @ c + d_b) for c in cs)
    clear = np.array([lowest_corner(T) for T in Gb])
    if a.raise_to_clear > 0:
        # low objects (aluminium bar ~3 cm): the TCP aims at the object centre, which puts the jaw tips at the floor.
        # Back such grasps off along the approach until the lowest tip corner clears the floor (at most --raise-to-clear).
        cosz = np.clip(-Gb[:, 2, 2], 0.3, 1.0)
        back = np.clip((a.floor_clear - clear) / cosz, 0.0, None)
        fix = (back > 0) & (back <= a.raise_to_clear)
        Gb[fix, :3, 3] -= back[fix, None] * Gb[fix, :3, 2]
        clear = np.array([lowest_corner(T) for T in Gb])
        print(f"[plan] raised {int(fix.sum())} grasps off the floor (<= {a.raise_to_clear*100:.1f} cm)")
    if a.shallow is not None:
        # 2026-11-21 (Sota: かき混ぜ以外で): grip only the top of the bar. The jaw tips go --shallow below the highest
        # target point between the jaws instead of where the model put them, so neighbours lower than that (the
        # sack, a lower bar beside it) stay below the fingertips and no longer block the grasp.
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        hw = a.open_mm / 2000.0
        moved = 0
        for i in range(len(Gb)):
            q = (ob_b - Gb[i][:3, 3]) @ Gb[i][:3, :3]
            m = (np.abs(q[:, 0]) < hw) & (np.abs(q[:, 1]) <= 0.019)
            if m.sum() < 20:
                continue
            top = float(np.percentile(q[m, 2], 3))  # nearest the hand along the approach = top of the bar
            shift = (top + a.shallow) - tip_off      # move the gripper so its tips sit at top + shallow
            Gb[i][:3, 3] += shift * Gb[i][:3, 2]; moved += 1
            # (22:58-23:15 a "raise until the corners clear the floor" correction lived here; removed at Sota's request:
            #  it produced grasps that pinched air. Candidates whose jaws reach the floor are now simply dropped.)
        clear = np.array([lowest_corner(T) for T in Gb])
        print(f"[plan] shallow grip: {moved} grasps set to tips {a.shallow*1000:.0f} mm below the bar top")
    # prefer grasps that go for the object's centre (a corner grasp misses with even a small error)
    ob_c = obj.mean(0) @ T_bc[:3, :3].T + T_bc[:3, 3]
    tcp_all = np.einsum("nij,j->ni", Gb[:, :3, :3], [0, 0, CFG["flange_to_tip_m"] - 0.015]) + Gb[:, :3, 3]
    cdist = np.linalg.norm(tcp_all[:, :2] - ob_c[:2], axis=1)
    # object centre in each grasp's own frame: closing (Xg) and thickness (Yg) offsets from the TCP.
    # The horizontal check above misses thickness offsets of tilted grasps (iter_034/036: misses had the block
    # 5-7 cm off along the jaw thickness, the jaws passed in front of it).
    rel = np.einsum("nji,nj->ni", Gb[:, :3, :3], ob_c[None, :] - tcp_all)
    toff, coff = np.abs(rel[:, 1]), np.abs(rel[:, 0])
    # 開いた爪の内側と物のすき間（2026-11-21 北九州）: 爪の全開が 80 mm だと、直径 5 cm の木片を斜めに挟む候補は
    # すき間が片側 5 mm 以下になり、腕の 1 cm 前後の誤差で爪が物に当たる（probe で木片が 17 mm 動いた）。
    # 爪の厚み（±17.7 mm + 5 mm）と長さ（指先から 80 mm）の範囲にある物の点の、閉じる向きの両端と開いた爪の内側の距離の小さい方
    jclr = np.full(len(Gb), np.inf)
    if a.jaw_clear is not None:
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        half = CFG["max_opening_m"] / 2
        for i in range(len(Gb)):
            q = (ob_b - tcp_all[i]) @ Gb[i][:3, :3]  # 指先の中心から見た（閉じる・厚み・近づく）
            m = (np.abs(q[:, 1]) <= 0.0177 + 0.005) & (q[:, 2] >= 0.015 - 0.080) & (q[:, 2] <= 0.015)
            if m.any():
                jclr[i] = min(half + q[m, 0].min(), half - q[m, 0].max())
        print(f"[plan] jaw clearance: {int(np.sum(jclr >= a.jaw_clear))} grasps keep ≥{a.jaw_clear*1000:.0f} mm between the open jaws and the object")
    hit = np.zeros(len(Gb), bool)
    if a.avoid_obstacles:
        # Other things on the floor next to the target (iter_021: the left jaw came down on a wood block touching
        # the aluminium bar and the arm was pushed off). Obstacles = above-floor scene points not in the target.
        from scipy.spatial import cKDTree
        from grasp.capture import depth_to_points
        Kc = np.array(json.load(open(os.path.join(a.cap_dir, "intrinsics.json")))["K"])
        dep = np.load(os.path.join(a.cap_dir, "depth.npy"))
        rp = os.path.join(a.cap_dir, "rigid.png")  # thor_snap --perceive: other bars / wood (never "soft")
        rig_img = None
        if os.path.exists(rp):
            import cv2
            rig_img = np.repeat((cv2.imread(rp, 0) > 0).astype(np.uint8)[..., None] * 255, 3, -1)
        sp, rg = depth_to_points(dep, Kc, color_bgr=rig_img, zmax=1.0)
        rg = np.zeros(len(sp), bool) if rg is None else rg[:, 0] > 0.5
        sb = sp @ T_bc[:3, :3].T + T_bc[:3, 3]
        hgt = sb @ n_b + d_b
        keep = (hgt > 0.008) & (hgt < 0.25)
        sb, rg = sb[keep], rg[keep]
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        dist, _ = cKDTree(ob_b).query(sb, k=1)
        near = (dist > 0.012) & (np.linalg.norm(sb[:, :2] - ob_b.mean(0)[:2], axis=1) < 0.30)
        obst, rg = sb[near], rg[near]
        if a.soft_below is not None:
            # In a pile the bar lies on soft stuff (sacks, bags, paper). Its surface beside the bar is lower than the bar
            # top and gives way under the jaws, so only things reaching up to (bar top - soft_below) count as obstacles
            # (11/21 mock pile: the sack under the bar blocked every grasp; a wood block next to the bar still counts).
            # Rigid things (rigid.png: other bars, wood) always count.
            top = float(np.percentile(ob_b @ n_b + d_b, 50))
            obst_gap = obst[((obst @ n_b + d_b) > top - a.gap_soft_below) | rg] if a.gap_soft_below is not None else None
            hard = ((obst @ n_b + d_b) > top - a.soft_below) | rg
            obst, rg = obst[hard], rg[hard]
            print(f"[plan] rigid neighbours (other bars / wood): {int(rg.sum())} points")
        if a.soft_below is None or a.gap_soft_below is None:
            obst_gap = None
        half_open = a.open_mm / 2000.0   # jaw inner faces; 93 mm at lerobot -60 deg (wrist camera)
        fx, fy, fl, mg = 0.022, 0.019, 0.050, a.obstacle_margin  # finger thickness (X), half width (Y), length back from tip (Z), margin
        tipz = CFG["flange_to_tip_m"]
        for i in range(len(Gb)):
            if len(obst) == 0:
                break
            q = (obst - Gb[i][:3, 3]) @ Gb[i][:3, :3]  # obstacle points in the grasp frame
            inx = (np.abs(q[:, 0]) >= half_open - mg) & (np.abs(q[:, 0]) <= half_open + fx + mg)
            iny = np.abs(q[:, 1]) <= fy + mg
            inz = (q[:, 2] >= tipz - fl - a.pregrasp - mg) & (q[:, 2] <= tipz + mg)  # swept from the pregrasp
            hit[i] = (not a.finger_contact) and np.sum(inx & iny & inz) > 8
            if a.clear_gap:  # other things inside the closing gap stop the jaws before they reach the target
                # (11/21 iter_049: a wood block and a second bar sat between the jaws; the jaws closed on them)
                # 11/21 iter_051: an L-shaped bar nobody had detected lay between the jaws. "Soft" (sack) was decided by
                # height only, so with --gap-soft-below unknown things inside the gap count unless clearly below the bar.
                qg = q if obst_gap is None else (obst_gap - Gb[i][:3, 3]) @ Gb[i][:3, :3]
                ing = (np.abs(qg[:, 0]) < half_open - 0.004) & (np.abs(qg[:, 1]) <= fy + mg) & (qg[:, 2] >= tipz - fl) & (qg[:, 2] <= tipz + mg)
                hit[i] |= np.sum(ing) > 8
        print(f"[plan] obstacles: {len(obst)} points near the target; {int(hit.sum())} grasps would hit them")
        if os.environ.get("PLAN_DEBUG_HITS"):  # which part of the gripper hits what, for the grasps that pass the tilt check
            rg_o = rg
            ok_t = np.where(tilt <= a.max_tilt)[0]
            hflag = {}
            stats = {"finger_rigid": 0, "finger_soft": 0, "gap_rigid": 0, "gap_soft": 0, "free": 0}
            for i in ok_t:
                q = (obst - Gb[i][:3, 3]) @ Gb[i][:3, :3]
                inx = (np.abs(q[:, 0]) >= half_open - mg) & (np.abs(q[:, 0]) <= half_open + fx + mg)
                iny = np.abs(q[:, 1]) <= fy + mg
                inz = (q[:, 2] >= tipz - fl - a.pregrasp - mg) & (q[:, 2] <= tipz + mg)
                ing = (np.abs(q[:, 0]) < half_open - 0.004) & iny & (q[:, 2] >= tipz - fl) & (q[:, 2] <= tipz + mg)
                f_ = inx & iny & inz
                hits = [("finger_rigid", (f_ & rg_o).sum()), ("finger_soft", (f_ & ~rg_o).sum()),
                        ("gap_rigid", (ing & rg_o).sum()), ("gap_soft", (ing & ~rg_o).sum())]
                big = [k for k, v in hits if v > 8]
                hflag[i] = 2 if "gap_rigid" in big else (1 if big else 0)
                for k in big:
                    stats[k] += 1
                if not big:
                    stats["free"] += 1
            np.save(os.path.join(a.cap_dir, "debug_obst_rigid.npy"), obst[rg_o]); np.save(os.path.join(a.cap_dir, "debug_obst_soft.npy"), obst[~rg_o])
            np.save(os.path.join(a.cap_dir, "debug_tcp_ok.npy"), np.c_[tcp_all[ok_t], [hflag.get(i, 0) for i in ok_t]])
            print(f"[plan] hit breakdown over {len(ok_t)} tilt-ok grasps: {stats}; target median height {top*100:.1f} cm")
        clear = np.where(hit, -1.0, clear)
    print(f"[plan] filters: tilt ok {int(np.sum(tilt <= a.max_tilt))}, floor ok {int(np.sum(clear >= a.floor_clear))}, "
          f"centre ok {int(np.sum(cdist <= a.center_tol))}, score ok {int(np.sum(S >= a.min_score))}")
    order = [i for i in np.argsort(-S) if a.min_tilt <= tilt[i] <= a.max_tilt and clear[i] >= a.floor_clear and cdist[i] <= a.center_tol
             and (a.thick_tol is None or (toff[i] <= a.thick_tol and coff[i] <= a.center_tol))
             and (a.jaw_clear is None or jclr[i] >= a.jaw_clear)
             and S[i] >= a.min_score]
    # 2026-11-21 (Sota: どこがどう動いているのか分かるように): what happened to each candidate, for plan_stats.json /
    # plan_funnel.jpg (the panel shows them). First failing filter wins; IK reasons are filled in below.
    fate = np.array(["ok"] * len(S), dtype=object)
    for name, bad in (("score", S < a.min_score), ("centre", cdist > a.center_tol), ("floor", (clear < a.floor_clear) & ~hit),
                      ("hit", hit), ("tilt", tilt > a.max_tilt)):
        fate[bad] = name
    print(f"[plan] floor normal (base) {np.round(n_b,3).tolist()}; {int(np.sum(clear >= a.floor_clear))} candidates keep ≥{a.floor_clear*100:.0f} cm clearance")
    print(f"[plan] {len(S)} candidates, {len(order)} from above (≤{a.max_tilt:.0f}°)")

    if os.environ.get("PLAN_DEBUG_ORDER"):
        for i in order[:12]:
            xg = Gb[i][:3, 0]
            print(f"[plan]   cand {i}: TCP {np.round(tcp_all[i], 3).tolist()} closing heading {np.degrees(np.arctan2(xg[1], xg[0])) % 180:.0f}° tilt {tilt[i]:.0f}° score {S[i]:.2f}")
    ik = arm_ik_class()()
    if ARM == "right" and a.prefer == PREFER_LEFT:
        a.prefer = ",".join(f"{v:.1f}" for v in mirror_q([float(v) for v in PREFER_LEFT.split(",")]))
    q0 = np.deg2rad([float(v) for v in a.q.split(",")])
    rng = np.random.default_rng(0)
    seeds = [q0, np.deg2rad([0, 0, 0, 60, 0, 0, 0]), np.deg2rad([-30, 0, 0, 90, 0, 0, 0])] + \
            [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(a.ik_seeds)]
    if os.environ.get("PLAN_SEED_Q"):  # 2026-11-22: a posture that worked before (lerobot deg) as an extra IK seed
        seeds.insert(0, np.deg2rad([float(v) for v in os.environ["PLAN_SEED_Q"].split(",")]))

    q_pref = np.deg2rad([float(v) for v in a.prefer.split(",")])
    seeds.insert(0, q_pref)

    def solve(T, first_seed):
        best = None
        for sd in [first_seed] + seeds:
            q, ok, _, _ = ik.solve(se3(T), sd, position_only=False, max_iter=200)
            if ok and ik.within_limits(q):
                d = np.max(np.abs(q - (first_seed if first_seed is not q0 else q_pref)))
                if best is None or d < best[0]:
                    best = (d, q)
        return None if best is None else best[1]
    result = None
    n_keepout = n_curobo = n_coll = 0
    ACOLL = None
    if os.environ.get("GRASP_ARM_COLLISION", "1") == "1":
        from grasp import arm_collision
        ACOLL = arm_collision.ArmCollision(ik)
        _ob = obj @ T_bc[:3, :3].T + T_bc[:3, 3]; _ext = (_ob - _ob.mean(0))[:, :2]
        _ev = np.linalg.eigh(np.cov(_ext.T))[1][:, -1]
        HELD_L = float(min(0.35, np.ptp(_ext @ _ev)))  # the bar we will hold (its length)
        print(f"[plan] 腕の形で当たりを確かめる（{', '.join(list(ACOLL.cubs) + [n for n, _ in ACOLL.meshes])}・余裕 {arm_collision.MARGIN*100:.1f} cm・持つ棒 {HELD_L*100:.0f} cm）")
    for rank, i in enumerate(order[:a.max_try]):
        for flip in (False, True):
            Tg = Gb[i] @ (FLIP if flip else np.eye(4))
            Tpre = Tg.copy(); Tpre[:3, 3] -= a.pregrasp * Tg[:3, 2]
            Tlift = Tg.copy(); Tlift[:3, 3] += [0, 0, a.lift]
            qs, ok_all, seed = [], True, q0
            why = None
            for nm, T in (("ik_grasp", Tg), ("ik_pre", Tpre), ("ik_lift", Tlift)):  # solve the grasp first, then seed the others from it
                q = solve(T @ np.linalg.inv(T_L7_G), seed if not qs else qs[0])
                if q is None:
                    ok_all = False; why = nm; break
                qs.append(q)
            if ok_all:
                # pre/lift must be close to the grasp on the same IK branch (a branch jump means a big swing while near the object)
                if max(np.max(np.abs(np.rad2deg(qs[1] - qs[0]))), np.max(np.abs(np.rad2deg(qs[2] - qs[0])))) > a.max_branch_deg:
                    ok_all = False; why = "branch"
            if ok_all and KEEPOUT and any(in_keepout(ik, q) for q in qs):
                ok_all = False  # 腕が胸のカメラと取り付け部の箱に入る解（本体のフレームに当たる）
                n_keepout += 1
            if ok_all and ACOLL is not None:
                # 2026-11-22 (Sota: 掴みに行くときにまたクランプの方にぶつかっている): the arm's real shape (link meshes,
                # gripper, held bar) along pregrasp→grasp (open jaws) and grasp→lift (holding), vs the clamp / camera / body
                q_deg = [np.rad2deg(q) for q in qs]  # grasp, pre, lift
                ok1, w1 = ACOLL.check_path(q_deg[1], q_deg[0], open_m=a.open_mm / 1000.0)
                ok2, w2 = ACOLL.check_path(q_deg[0], q_deg[2], held=HELD_L, open_m=0.0) if ok1 else (True, {})
                if not (ok1 and ok2):
                    ok_all = False; why = "coll"; n_coll += 1
                    if n_coll <= 3:
                        print(f"[plan] 腕の形: 候補 {rank} を外す（{arm_collision.describe(w1 if not ok1 else w2)}）")
            if ok_all and curobo.enabled():
                # cuRobo の球の腕で、pre → grasp → lift を関節の直線で動かす途中が置き場の箱（本体のフレーム・机）に当たらないか
                # （2026-11-21 北九州: 持ち上げの段で肘が本体のフレームに当たった。腕の点だけを見る keep-out の箱を置き換える）
                bad = curobo.violations([np.rad2deg(qs[1]), np.rad2deg(qs[0]), np.rad2deg(qs[2])],
                                        names=["pregrasp→grasp", "grasp→lift"], n=20)
                if bad:
                    ok_all = False; why = "curobo"
                    n_curobo += 1
                    if n_curobo <= 3:
                        print(f"[plan] cuRobo: 候補 {rank} を外す（{curobo.describe(bad)}）")
            q_look = None
            if ok_all and a.look_back > 0:
                Tlook = Tpre.copy(); Tlook[:3, 3] -= a.look_back * Tg[:3, 2]
                q_look = solve(Tlook @ np.linalg.inv(T_L7_G), qs[1])
                if q_look is None or np.max(np.abs(np.rad2deg(q_look - qs[1]))) > 30:
                    ok_all = False
            if ok_all and a.extra_depth > 0:
                Tdeep = Tg.copy(); Tdeep[:3, 3] += a.extra_depth * Tg[:3, 2]
                q_deep = solve(Tdeep @ np.linalg.inv(T_L7_G), qs[0])
                if q_deep is None or np.max(np.abs(np.rad2deg(q_deep - qs[0]))) > 30:
                    ok_all = False
            if ok_all and a.wrist_near:
                w6, w7, wt = (float(v) for v in a.wrist_near.split(","))
                g6, g7 = np.rad2deg(qs[0][5]), np.rad2deg(qs[0][6])
                if abs(g6 - w6) > wt or abs(g7 - w7) > wt:
                    ok_all = False
            if not ok_all:
                fate[i] = why or "other"
            if ok_all:
                fate[i] = "chosen"
                qs = [qs[1], qs[0], qs[2]]  # → order: pre, grasp, lift
            if ok_all:
                result = dict(rank=int(rank), idx=int(i), score=float(S[i]), tilt_deg=float(tilt[i]), flip=flip,
                              T_grasp_base=Tg.tolist(), q_pre_deg=np.rad2deg(qs[0]).tolist(),
                              q_grasp_deg=np.rad2deg(qs[1]).tolist(), q_lift_deg=np.rad2deg(qs[2]).tolist(),
                              q_start_deg=np.rad2deg(q0).tolist(),
                              q_look_deg=None if q_look is None else np.rad2deg(q_look).tolist(), look_back_m=a.look_back)
                break
        if result:
            break
    if KEEPOUT:
        print(f"[plan] keep-out: {n_keepout} IK solutions dropped (the arm enters a keep-out box: {len(KEEPOUT)} box(es))")
    if ACOLL is not None:
        print(f"[plan] 腕の形: {n_coll} IK solutions dropped (the arm, gripper or held bar comes within {arm_collision.MARGIN*100:.1f} cm of the clamp / camera / body)")
    if curobo.enabled():
        print(f"[plan] cuRobo: {n_curobo} IK solutions dropped (the arm spheres come within {curobo.MARGIN*1000:.0f} mm of a box)")
    for i in order[a.max_try:]:
        fate[i] = "untried"
    if result is not None:
        result["grasp_source"] = str(SRC[result["idx"]])
    srcs = "+".join(sorted(set(SRC.tolist())))
    write_stats(a.cap_dir, obj, T_bc, tcp_all, fate, S, src=srcs, result=result, SRC=SRC)
    if os.environ.get("PLAN_DEBUG_FATES"):  # per-candidate record for offline analysis
        hd = np.degrees(np.arctan2(Gb[:, 1, 0], Gb[:, 0, 0])) % 180
        np.savez(os.path.join(a.cap_dir, "plan_fates.npz"), tcp=tcp_all, heading=hd, tilt=tilt, fate=fate.astype(str), src=SRC.astype(str))
    if not result:
        print("[plan] no candidate where all of pregrasp, grasp and lift solve"); return 1
    # sanity: does the TCP land on the object?
    tcp = (np.array(result["T_grasp_base"]) @ np.r_[0, 0, CFG["flange_to_tip_m"] - 0.015, 1])[:3]
    ob = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
    d = float(np.min(np.linalg.norm(ob - tcp, axis=1)))
    result["tcp_base"] = tcp.tolist(); result["tcp_to_object_m"] = d
    if result is not None and a.open_mm != 93.0:
        # lerobot gripper deg is linear in the tip gap: +6.1 closed, -60 -> 93 mm (wrist camera 11/20)
        result["gripper_open_deg"] = round(6.1 - a.open_mm * 66.1 / 93.0, 1)
        result["open_mm"] = a.open_mm
    json.dump(result, open(os.path.join(a.cap_dir, "plan.json"), "w"), indent=2)
    print(f"[plan] chosen: rank {result['rank']} score {result['score']:.2f} tilt {result['tilt_deg']:.0f}° flip={result['flip']}")
    for k in ("q_start_deg", "q_pre_deg", "q_grasp_deg", "q_lift_deg"):
        print(f"  {k:12s} " + " ".join(f"{v:+6.1f}" for v in result[k]))
    print(f"  TCP {np.round(tcp,3).tolist()}  distance to nearest object point {d*1000:.0f} mm")
    if not a.no_render:  # 図（plan.png）は腕のメッシュの読み込みと描画で 2〜3 秒かかる（2026-11-21 北九州で測った）
        render(a.cap_dir, result, ob)
    return 0


FATE_JA = {"tilt": "傾きすぎ", "hit": "隣の物に当たる", "floor": "床に近すぎ", "centre": "物の中心から遠い", "score": "点数が低い",
           "untried": "試す前に打ち切り", "ik_grasp": "IK: 掴む姿勢が取れない", "ik_pre": "IK: 近づく姿勢が取れない",
           "ik_lift": "IK: 持ち上げる姿勢が取れない", "branch": "IK: 姿勢が大きく飛ぶ", "curobo": "cuRobo: 本体・台に当たる", "coll": "腕がクランプ・カメラ・本体に当たる",
           "other": "その他", "chosen": "選んだ", "ok": "試す前に決まった"}
FATE_BGR = {"tilt": (150, 150, 150), "score": (150, 150, 150), "centre": (150, 150, 150), "untried": (150, 150, 150),
            "floor": (0, 200, 255), "hit": (0, 140, 255), "ik_grasp": (255, 80, 0), "ik_pre": (255, 80, 0), "ik_lift": (255, 80, 0),
            "branch": (255, 0, 160), "curobo": (200, 0, 255), "coll": (200, 0, 255), "other": (120, 120, 120), "chosen": (0, 220, 0), "ok": (150, 150, 150)}


def write_stats(cap_dir, obj, T_bc, tcp_all, fate, S, src, result, SRC=None):
    """plan_stats.json (one entry per target tried; a pile may swap to the next bar) + plan_funnel_<k>.jpg: every
    candidate's TCP on the chest image, coloured by what stopped it. Read by grasp/enpire/panel.py."""
    import cv2
    from collections import Counter
    p = os.path.join(cap_dir, "plan_stats.json")
    allst = json.load(open(p)) if os.path.exists(p) else []
    k = len(allst)
    c = Counter(fate.tolist())
    st = {"target": k, "source": src, "object_points": int(len(obj)), "candidates": int(len(S)),
          "fates": {FATE_JA.get(f, f): int(n) for f, n in c.most_common()},
          "chosen": None if result is None else {kk: result[kk] for kk in ("rank", "score", "tilt_deg", "grasp_source") if kk in result},
          "by_source": {} if SRC is None else {sname: {FATE_JA.get(f, f): int(n) for f, n in Counter(fate[SRC == sname].tolist()).most_common()}
                                               for sname in sorted(set(SRC.tolist()))},
          "image": f"plan_funnel_{k}.jpg"}
    try:
        # advise from the bar grasps when they are pooled: GraspGenX proposes many side grasps that are "too tilted"
        # whatever the placement, which would drown the useful reason (11/21 23:14)
        cb = Counter(fate[SRC == "bar"].tolist()) if SRC is not None and (SRC == "bar").any() else c
        st["advice"], st["advice_image"] = advise(cap_dir, obj, T_bc, cb, k, result)
    except Exception as e:  # noqa: BLE001  (advice is only for people)
        st["advice"], st["advice_image"] = None, None
        print(f"[plan] advice failed: {e}")
    allst.append(st)
    json.dump(allst, open(p, "w"), indent=1, ensure_ascii=False)
    try:
        img = cv2.imread(os.path.join(cap_dir, "color.png"))
        K = np.array(json.load(open(os.path.join(cap_dir, "intrinsics.json")))["K"])
        Ti = np.linalg.inv(T_bc)
        for pt in obj[:: max(1, len(obj) // 1500)]:
            u, v = int(K[0, 0] * pt[0] / pt[2] + K[0, 2]), int(K[1, 1] * pt[1] / pt[2] + K[1, 2])
            cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        order = sorted(range(len(fate)), key=lambda i: (fate[i] == "chosen", fate[i] not in ("tilt", "score", "centre", "untried")))
        for i in order:
            pc = Ti[:3, :3] @ tcp_all[i] + Ti[:3, 3]
            if pc[2] <= 0:
                continue
            u, v = int(K[0, 0] * pc[0] / pc[2] + K[0, 2]), int(K[1, 1] * pc[1] / pc[2] + K[1, 2])
            cv2.circle(img, (u, v), 6 if fate[i] == "chosen" else 2, FATE_BGR.get(fate[i], (255, 255, 255)), -1)
        y = 18
        for f, n in c.most_common():
            cv2.putText(img, f"{f} {n}", (8, y), 0, 0.45, FATE_BGR.get(f, (255, 255, 255)), 2); y += 17
        cv2.imwrite(os.path.join(cap_dir, st["image"]), img)
    except Exception as e:  # noqa: BLE001  (the picture is only for people)
        print(f"[plan] funnel image failed: {e}")


REACH_C = np.array([0.38, 0.05])  # middle of where the left arm reaches the floor well (11/21: x 0.33-0.45, y -0.05..0.15)


def advise(cap_dir, obj, T_bc, counts, k, result):
    """Plain words + a picture with an arrow: what to change so a grasp exists (2026-11-21, Sota: 理由を人に分かるように)."""
    import cv2
    if result is not None:
        return None, None
    groups = {"hit": counts.get("hit", 0), "floor": counts.get("floor", 0), "tilt": counts.get("tilt", 0),
              "reach": sum(counts.get(x, 0) for x in ("ik_grasp", "ik_pre", "ik_lift", "branch")),
              "body": counts.get("curobo", 0)}
    main = max(groups, key=groups.get)
    if groups["reach"] > 0 and groups["reach"] + groups["body"] >= 5:
        # some grasps passed every check on the object itself and only the arm could not take the pose: that is the
        # reason a person can act on (move/turn the bar into reach), even when more candidates failed on hits / tilt
        main = "reach" if groups["reach"] >= groups["body"] else "body"
    img = cv2.imread(os.path.join(cap_dir, "color.png")); H, W = img.shape[:2]
    K = np.array(json.load(open(os.path.join(cap_dir, "intrinsics.json")))["K"]); Ti = np.linalg.inv(T_bc)
    def px(pb):
        pc = Ti[:3, :3] @ pb + Ti[:3, 3]
        return np.array([K[0, 0] * pc[0] / pc[2] + K[0, 2], K[1, 1] * pc[1] / pc[2] + K[1, 2]])
    uv = np.c_[K[0, 0] * obj[:, 0] / obj[:, 2] + K[0, 2], K[1, 1] * obj[:, 1] / obj[:, 2] + K[1, 2]]
    ctr = uv.mean(0)
    ov = img.copy()
    rig = cv2.imread(os.path.join(cap_dir, "rigid.png"), 0)
    if rig is not None:
        ov[rig > 0] = (255, 0, 255)
    for u, v in uv[:: max(1, len(uv) // 3000)].astype(int):
        cv2.circle(ov, (u, v), 2, (0, 0, 255), -1)
    img = cv2.addWeighted(ov, 0.5, img, 0.5, 0)
    ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
    hgt_top = None
    try:
        n_b, d_b = floor_plane_base(cap_dir, T_bc); hgt_top = float(np.percentile(ob_b @ n_b + d_b, 95))
    except Exception:  # noqa: BLE001
        pass
    arrow = None
    if main == "hit":
        txt = ("爪を下ろす所に、隣の物（紫）がくっついている。赤い棒を、矢印の向きへ 5 cm ほど離して置いてください。"
               "（隣の物を棒から離しても同じ）")
        if rig is not None:
            v_, u_ = np.nonzero(rig > 0)
            P = np.c_[u_, v_].astype(float)
            near = P[np.linalg.norm(P - ctr, axis=1) < 120] if len(P) else P
            if len(near):
                d = ctr - near.mean(0); d /= max(1e-6, np.linalg.norm(d)); arrow = (ctr, ctr + 70 * d)
    elif main == "reach":
        far = np.linalg.norm(ob_b[:, :2].mean(0) - REACH_C) > 0.06
        txt = ("腕が届かない（または手首をその向きに回せない）。" +
               ("赤い棒を矢印の先（緑の丸）あたりへ寄せ、" if far else "") +
               "ロボットから見てほぼ横向き（緑の線の向き）に置いてください。")
        tgt = px(np.r_[REACH_C, ob_b[:, 2].mean()])
        arrow = (ctr, tgt) if far else None
        a_ = np.deg2rad(115); u2 = np.array([np.cos(a_), np.sin(a_)])
        p1, p2 = px(np.r_[REACH_C - 0.08 * u2, ob_b[:, 2].mean()]), px(np.r_[REACH_C + 0.08 * u2, ob_b[:, 2].mean()])
        cv2.line(img, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 220, 0), 4)
        cv2.circle(img, tuple(tgt.astype(int)), 10, (0, 220, 0), 3)
    elif main == "floor":
        txt = (f"棒が床にじかに置かれていて低く（上面が床から {hgt_top*100:.0f} cm）、爪の先が床に当たる。"
               "棒を袋の上など少し高い所に乗せてください。") if hgt_top is not None else "爪の先が床に当たる。棒を少し高い所に乗せてください。"
    elif main == "tilt":
        txt = "上から掴める向きがない（爪を寝かせすぎになる）。棒をロボットから見て横向きに置き直してください。"
    else:
        txt = "腕が本体・カメラの台・クランプに当たる。棒を矢印の向き（ロボットから離れる方）へ少し離してください。"
        d = ctr - px(np.array([0.0, 0.0, ob_b[:, 2].mean()])); d /= max(1e-6, np.linalg.norm(d)); arrow = (ctr, ctr + 70 * d)
    if arrow is not None:
        a0, a1 = (tuple(np.clip(x, 5, [W - 5, H - 5]).astype(int)) for x in arrow)
        cv2.arrowedLine(img, a0, a1, (255, 255, 255), 9, tipLength=0.3)
        cv2.arrowedLine(img, a0, a1, (0, 200, 0), 5, tipLength=0.3)
    name = f"advice_{k}.jpg"
    cv2.imwrite(os.path.join(cap_dir, name), img)
    return txt, name


def render(cap_dir, r, ob):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from grasp.urdf_fk import LeftArmFK
    fk = LeftArmFK(); links = [f"openarm_left_link{i}" for i in range(1, 8)]
    S = fk.surface_points(links, n_per_link=300)
    fig = plt.figure(figsize=(13, 6))
    for col, (title, key, c) in enumerate([("current pose", "q_start_deg", "gray"), ("at the grasp pose (IK solution)", "q_grasp_deg", "tab:blue")]):
        ax = fig.add_subplot(1, 2, col + 1, projection="3d")
        P = fk.link_poses(np.deg2rad(r[key]))
        pts = np.vstack([S[l] @ P[l][:3, :3].T + P[l][:3, 3] for l in S])
        ax.scatter(*pts.T, s=1, c=c, alpha=.4)
        ax.scatter(*ob[::3].T, s=2, c="red", label="object (camera → base)")
        Tg = np.array(r["T_grasp_base"]); o = Tg[:3, 3]
        for k, cc in zip(range(3), ["r", "g", "b"]):
            ax.plot(*np.c_[o, o + 0.08 * Tg[:3, k]], c=cc, lw=2)
        tcp = np.array(r["tcp_base"]); ax.scatter(*tcp, s=60, c="lime", marker="*", label="target TCP")
        L7 = P["openarm_left_link7"]; tcp_now = (L7 @ np.r_[0, 0, 0.2145, 1])[:3]
        ax.scatter(*tcp_now, s=40, c="k", marker="x", label="TCP in this pose")
        ax.set_title(title); ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_zlabel("z up")
        allp = np.vstack([pts, ob]); m = allp.mean(0); rr = np.ptp(allp, 0).max() / 2
        ax.set_xlim(m[0] - rr, m[0] + rr); ax.set_ylim(m[1] - rr, m[1] + rr); ax.set_zlim(m[2] - rr, m[2] + rr)
        ax.view_init(elev=20, azim=-60); ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(cap_dir, "plan.png"), dpi=85)


if __name__ == "__main__":
    raise SystemExit(main())
