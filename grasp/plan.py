#!/usr/bin/env python3
"""Step 5: grasp candidates (camera frame) → arm base frame → IK. **Does not move the arm.**

1. Object point cloud (object.npy from thor_snap) → GraspGenX (the measured OpenArm gripper)
2. T_base_cam (grasp/config/T_base_cam.json) → base frame
3. GraspGenX gripper frame (origin = flange, +Z approach, +X closing) → link7 target pose.
   Fingers extend along link7 +z and close along y, so in link7: Xg=+y, Yg=-x, Zg=+z, origin = flange z=0.0895
4. Keep only grasps from above (approach within --max-tilt deg of straight down). Try them in score order, and take the first
   one where all three poses solve: pregrasp (10 cm back along the approach) / grasp / lift (10 cm straight up).
5. Write plan.json (joint angles in lerobot degrees) and plan.png (3D view)

右腕（2026-11-22 Munenori）: --arm left|right|both（既定は GRASP_ARM か left）。both は左右を別のプロセスで同時に計算し、
GraspGenX の点数の高い方（差が --arm-tie 以内なら、今の姿勢から pregrasp までの関節の動きが小さい方）を選ぶ。
plan.json に "arm" を書き、run_grasp.py はその腕を動かす。腕ごとの結果は plan_left.json・plan_right.json にも残す。
右腕の今の角度は --q-right（無ければ GRASP_REST_RIGHT、それも無ければ --q を鏡に写した角度。計算の種と図にだけ使う）。

    PYTHONPATH=.:~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/plan.py grasp/runs/cap_xxx --q "-17.5,2.1,4,28.6,-8.7,-15.5,-27.7"
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import pinocchio as pin  # noqa: E402

from grasp import arm as armcfg  # noqa: E402
from grasp.curobo import client as curobo  # noqa: E402  GRASP_CUROBO があるときだけ使う
from grasp.depth import grasp_depths, jaw_clearance  # noqa: E402
from grasp.infer import infer_grasps, infer_graspnet, load_sweep, scene_cloud  # noqa: E402
from ik.openarm_left_ik import OpenArmIK  # noqa: E402
from grasp import timing  # noqa: E402  段ごとの時間（記録だけ・2026-11-25 Shreyan）

HERE = os.path.dirname(os.path.abspath(__file__))
CFG = json.load(open(os.environ.get("GRASP_GRIPPER_JSON", os.path.join(HERE, "config", "openarm_left_gripper.json"))))  # 置き場の爪は grasp/config/sites/
SWEEP_JSON = os.environ.get("GRASP_SWEEP_JSON", os.path.join(HERE, "config", "openarm_custom_hand_sweep.json"))
T_L7_G = np.eye(4)
T_L7_G[:3, :3] = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # columns: Xg=+y, Yg=-x, Zg=+z
T_L7_G[:3, 3] = [0, 0, CFG["hand_mount_face_z_in_link7_m"]]
if "tcp_in_link7_taught_m" in CFG:  # shift the gripper base by the offset between the taught TCP and the model TCP
    _tcp_model = np.array([0, 0, CFG["hand_mount_face_z_in_link7_m"] + CFG["flange_to_tip_m"] - 0.015])
    T_L7_G[:3, 3] += np.array(CFG["tcp_in_link7_taught_m"]) - _tcp_model
T_L7_G_LEFT = T_L7_G


def t_l7_g(side: str) -> np.ndarray:
    """link7 → グリッパーの基準の座標。右の link7 は左の鏡写しで自分の y が反転している（grasp/arm.py）ので、
    手で教えた TCP のずれの y だけ反転する。回りは同じでよい（爪は対称で、FLIP の向きも両方試す）。"""
    if side == "left":
        return T_L7_G_LEFT
    T = T_L7_G_LEFT.copy()
    T[1, 3] *= -1.0
    return T


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
    js = [ik.data.oMi[ik.model.getJointId(f"openarm_{ik.side}_joint{j}")].translation.copy() for j in range(3, 8)]
    pts = js + [(a + b) / 2 for a, b in zip(js[:-1], js[1:])]
    Tg_ = t_l7_g(ik.side)
    MG = ik.data.oMi[ik.model.getJointId(f"openarm_{ik.side}_joint7")].act(pin_.SE3(Tg_[:3, :3], Tg_[:3, 3]))
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


def floor_plane_base(cap_dir, T_bc):
    """Floor plane from the capture's depth (RANSAC) → base frame. Returns (unit normal pointing up, d) with height = n·p + d."""
    # 2026-11-23 Munenori: 結果を <回>/floor_plane.json に残し、plan_place.py などの 2 回目からは読むだけにする（0.6 秒 × 回数を減らす）。
    # 深度は 2 画素おきに間引く（後で 5 mm の格子に間引くので、当てはめの結果はほとんど変わらない）
    cf = os.path.join(cap_dir, "floor_plane.json")
    if os.path.exists(cf):
        c = json.load(open(cf))
        if np.allclose(c.get("T_base_cam", []), T_bc):
            return np.array(c["n"]), float(c["d"])
    import open3d as o3d
    from grasp.capture import depth_to_points
    K = np.array(json.load(open(os.path.join(cap_dir, "intrinsics.json")))["K"])
    dep = np.load(os.path.join(cap_dir, "depth.npy"))
    K2 = K.copy(); K2[:2] /= 2
    pts, _ = depth_to_points(np.ascontiguousarray(dep[::2, ::2]), K2, zmax=1.0)
    pb = pts @ T_bc[:3, :3].T + T_bc[:3, 3]
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pb.astype(float))).voxel_down_sample(0.005)
    (a, b, c, d), _ = pc.segment_plane(0.008, 3, 2000)
    n = np.array([a, b, c]); nn = np.linalg.norm(n)
    n, d = n / nn, d / nn
    if n[2] < 0:
        n, d = -n, -d
    json.dump({"n": n.tolist(), "d": float(d), "T_base_cam": np.asarray(T_bc).tolist()}, open(cf, "w"))
    return n, d


def se3(T):
    return pin.SE3(T[:3, :3].copy(), T[:3, 3].copy())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--q", required=True, help="current joint angles (lerobot degrees, 7 values comma-separated)")
    ap.add_argument("--arm", choices=["left", "right", "both"], default=None,
                    help="どちらの腕で計画するか（既定 GRASP_ARM か left）。both は左右を同時に計算して良い方を選ぶ")
    ap.add_argument("--q-right", default=None, help="右腕の今の角度（lerobot の度）。無ければ GRASP_REST_RIGHT か --q の鏡写し")
    ap.add_argument("--obstacle-objects", action="store_true",
                    help="カメラで見た点の粗い箱の代わりに、thor_snap.py --yoloe が残したほかの物（object_alt_*.npy）を 1 つずつ箱にして"
                         "障害物に入れる（机から物の上端まで・余白 1 cm）。2026-11-23 Shreyan: --ignore-seen では隣の木の丸太に爪が当たり、"
                         "粗い箱では 37 cm の箱が拾う物ごと覆って計画が立たなかった")
    ap.add_argument("--ignore-seen", action="store_true",
                    help="カメラで見た周りの物（掴む物以外）を障害物に入れない（当たってもよいから目標を拾う）。置き場の物・机・もう片方の腕は見る")
    ap.add_argument("--ik", choices=["pinocchio", "curobo"], default=os.environ.get("GRASP_PLAN_IK", "pinocchio"),
                    help="掴み方の IK: pinocchio（1 つずつ・初期値 7〜11 個）か curobo（cuRobo のサーバーでまとめて解く・周りの物も見る）")
    ap.add_argument("--prefer-right", default=None, help="右腕の好む姿勢（既定は --prefer の鏡写し）")
    ap.add_argument("--arm-tie", type=float, default=0.05,
                    help="both のとき、点数の差がこれ以内なら関節の動きが小さい方の腕を選ぶ")
    ap.add_argument("--max-tilt", type=float, default=60.0, help="allowed tilt of the approach from straight down [deg] (on the floor, straight-down only reaches up to x≈0.35 m)")
    ap.add_argument("--pregrasp", type=float, default=0.10)
    ap.add_argument("--lift", type=float, default=0.10)
    ap.add_argument("--short-fallback", default=None,
                    help="\"pregrasp,lift\" [m]（例 0.06,0.05）。--pregrasp・--lift でどの腕にも候補が無ければ、同じ候補をこの短い距離で解き直す"
                         "（2026-11-25 Shreyan: 右腕の届く端の掴み方は、近づく向きに 10 cm 戻した手前の姿勢が届かず、IK の「3 つとも」が 0 だった。"
                         "6・5 cm なら同じ候補で 3 つ解けた）。使った距離は plan.json の pregrasp_m・lift_m に書き、run_grasp.py がそれで近づく")
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--floor-clear", type=float, default=0.012, help="minimum height of the lowest jaw-tip corner above the floor [m]")
    ap.add_argument("--center-tol", type=float, default=0.02, help="max horizontal distance from the grasp centre to the object's centre [m]")
    ap.add_argument("--check-descent", action="store_true",
                    help="pregrasp → 掴む・掴む → 持ち上げを手先の直線（10 段の IK）でたどれる候補だけ使う（2026-11-25 Shreyan: 左腕で、J3 が可動域の端の"
                         "掴み方は真下へ差し込む直線が解けず、cuRobo の plan_grasp が作れなかった。関節の直線で差し込むと爪が横へ振れて立てた丸太を押した）")
    ap.add_argument("--max-branch-deg", type=float, default=45.0, help="max joint change between grasp and pre/lift [deg]")
    ap.add_argument("--wrist-near", default=None,
                    help="j6,j7,tol (lerobot deg): only use grasps whose J6/J7 at the grasp are within tol of these. The taught TCP "
                         "(config tcp_in_link7_taught_m) was measured at J6 +36 / J7 +25; far from it the jaw position error grows (11/20 trial 3)")
    ap.add_argument("--look-back", type=float, default=0.0,
                    help="also require a 'look' pose this far behind the pregrasp along the approach to solve (wrist-camera alignment needs it) [m]")
    ap.add_argument("--obstacle-margin", type=float, default=0.010, help="clearance around the jaw boxes for --avoid-obstacles [m]")
    ap.add_argument("--avoid-obstacles", action="store_true",
                    help="drop grasps whose open jaws (swept from the pregrasp) would hit other above-floor points")
    ap.add_argument("--jaw-clear", type=float, default=None,
                    help="開いた爪の内側と物のすき間が、閉じる向きの両側ともこれ以上ある候補だけ使う [m]（爪の全開が狭いとき。北九州は 0.012）")
    ap.add_argument("--thick-tol", type=float, default=None, help="max object-centre offset along the jaw thickness (Yg) [m]; also bounds the closing-axis offset by --center-tol")
    ap.add_argument("--min-depth-frac", type=float, default=0.5,
                    help="挟む深さの下限を「--min-depth と、物の机からの高さ × これ の小さい方」にする（2026-11-23: 高さ 3 cm のアルミのバーは"
                         "爪の先を机から浮かせると 30 mm の深さがとれず、全部の候補が落ちた。0 で --min-depth のまま）")
    ap.add_argument("--min-depth", type=float, default=float(os.environ.get("GRASP_MIN_DEPTH", "0")),
                    help="挟む深さ（指先から、閉じた爪が物の一番太い所に当たる所まで）がこれ以上の候補だけ使う [m]。0 で見ない。"
                         "既定は GRASP_MIN_DEPTH。2026-11-22 の木材は 10・17 mm で抜けた（grasp/depth.py）")
    ap.add_argument("--push-depth", type=float, default=0.0, help="move every grasp this much further along its approach before checks/IK [m]")
    ap.add_argument("--extra-depth", type=float, default=0.0,
                    help="also require the grasp moved this much further along the approach to be IK-solvable [m] "
                         "(the real jaws stop 3-5 cm short of the model; the wrist servo needs the margin to go deeper)")
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
    ap.add_argument("--prefer", default="-17.7,2.0,4.7,28.1,-8.3,-15.5,-28.9",
                    help="preferred posture (lerobot degrees). Among IK solutions, pick the one closest to this (avoids odd branches like an elbow flung outward)")
    a = ap.parse_args(argv)
    timing.startup()

    obj = np.load(os.path.join(a.cap_dir, "object.npy"))
    if os.environ.get("GRASP_SITE") and "GRASP_T_BASE_CAM" not in os.environ:  # 東京のハンドアイで計画しない
        raise SystemExit("[plan] GRASP_T_BASE_CAM が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。その置き場のハンドアイを取ってから")
    T_bc = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(HERE, "config", "T_base_cam.json"))))["T_base_cam"])
    cache = os.path.join(a.cap_dir, "grasps_cam.npz" if a.model == "graspgenx" else "grasps_graspnet.npz")
    if os.path.exists(cache) and not a.reinfer:
        z = np.load(cache); G, S = z["G"], z["S"]; print(f"[plan] reusing {cache}")
    elif a.model == "graspnet":
        scene = scene_cloud(a.cap_dir, obj)
        Gs, Ss = [], []
        for k in range(a.rounds):  # 20k-point sampling differs per seed, so pool a few calls like GraspGenX
            g, sc = infer_graspnet(scene, host=a.host, seed=k)
            Gs.append(g); Ss.append(sc)
        G, S = np.concatenate(Gs), np.concatenate(Ss)
        o = np.argsort(-S); G, S = G[o], S[o]
        np.savez(cache, G=G, S=S)
    else:
        Gs, Ss = [], []
        for k in range(a.rounds):  # diffusion samples differ each call, so pool several calls
            _t = time.time()
            g, sc = infer_grasps(obj, sweep=load_sweep(SWEEP_JSON),
                                 host=a.host, num_grasps=400, topk=200, seed=k, sweep_num=a.per_call)
            timing.event("plan/graspgenx infer", _t, time.time(), call=k, per_call=a.per_call, pts=len(obj), grasps=len(sc))
            Gs.append(g); Ss.append(sc)
        G, S = np.concatenate(Gs), np.concatenate(Ss)
        np.savez(cache, G=G, S=S)
    Gb = np.einsum("ij,njk->nik", T_bc, G)  # grasps in base frame
    if a.push_depth:  # grasp deeper along the approach (11/20 iter_034: jaws closed near the block's top edge and slipped)
        Gb[:, :3, 3] += a.push_depth * Gb[:, :3, 2]
    down = np.array([0, 0, -1.0])
    tilt = np.degrees(np.arccos(np.clip(Gb[:, :3, 2] @ down, -1, 1)))
    # Floor clearance: in the first real run the lower jaw corner hit the floor (and pushed the robot/cart).
    # Fit the floor plane in the base frame from the capture, and drop grasps where the lowest jaw-tip corner is less than --floor-clear above the floor.
    _t = time.time()
    n_b, d_b = floor_plane_base(a.cap_dir, T_bc)
    timing.event("plan/floor plane", _t, time.time())
    arm_sel = a.arm or os.environ.get("GRASP_ARM", "left")
    if arm_sel not in ("left", "right", "both"):
        raise SystemExit(f"[plan] 腕は left・right・both: {arm_sel!r}")
    sides = ["left", "right"] if arm_sel == "both" else [arm_sel]
    P_obs = None
    if curobo.enabled() or os.environ.get("GRASP_CUROBO_RIGHT"):
        from grasp.curobo import seen
        # --ignore-seen では障害物にしないので計算しない（0.4 秒・2026-11-23）。run_grasp.py が記録の有無を見るので空で残す
        _t = time.time()
        P_obs = np.zeros((0, 3)) if (a.ignore_seen or a.obstacle_objects) else seen.seen_points(a.cap_dir, T_bc, obj, n_b, d_b)
        timing.event("plan/seen obstacle points", _t, time.time())
        np.save(os.path.join(a.cap_dir, seen.POINTS_FILE), P_obs)

    def prepare_curobo(side):
        """カメラで見た障害物（机の上の箱など）を、その腕の cuRobo のサーバーに入れる（grasp/curobo/seen.py・2026-11-21）。
        撮影時のその腕の姿勢のまわりの点は腕自身として消す。もう片方の腕の点は障害物として残る（当たらない側に倒す）。"""
        curobo.set_arm(side)
        if not curobo.enabled():
            return
        from grasp.curobo import seen
        # 今の角度（--q・--q-right）も腕自身として消す。2026-11-22: 左腕が休みの姿勢（GRASP_REST）から J7 で 9° ずれて置いてあり、
        # 左の爪の先が障害物の箱（2 個）として残った
        arm_qs = seen.capture_arm_qs(extra=[start_q(side)], side=side)
        fz = float(os.environ.get("GRASP_FLOOR_Z", -d_b / n_b[2]))
        # 撮影時の腕自身の点を消す幅。右の爪の先は休みの姿勢で球から 3〜4.2 cm に写った（2026-11-22・右の爪の球をまだ当てはめ直していない）
        margin = float(armcfg.env("GRASP_SEEN_ARM_MARGIN", side, "0.03"))
        obj_boxes = None
        if a.obstacle_objects:
            import glob as _glob
            obj_boxes = []
            # 拾う物の机に沿った外形（主成分の向きの箱・余白 1 cm）。ほかの物の点の 3 割より多くがこの中にあれば、同じ物の別の mask
            # （容器の口・縁など）なので障害物にしない（2026-11-25 Shreyan: 容器の口が 5 × 4 cm の箱になって、爪の入る所をふさいでいた）
            tb_xy = (obj @ T_bc[:3, :3].T + T_bc[:3, 3])[:, :2]
            tc = tb_xy.mean(0); _w, _V = np.linalg.eigh(np.cov((tb_xy - tc).T)); tR = _V[:, ::-1]
            t_lo, t_hi = np.percentile((tb_xy - tc) @ tR, 2, axis=0) - 0.01, np.percentile((tb_xy - tc) @ tR, 98, axis=0) + 0.01
            for k, f in enumerate(sorted(_glob.glob(os.path.join(a.cap_dir, "object_alt_*.npy")))):
                pb = np.load(f) @ T_bc[:3, :3].T + T_bc[:3, 3]
                if len(pb) < 20:
                    continue
                loc_t = (pb[:, :2] - tc) @ tR
                inside = float(np.mean(np.all((loc_t >= t_lo) & (loc_t <= t_hi), axis=1)))
                if inside > 0.3:
                    print(f"[plan] obj_{k}: 点の {inside * 100:.0f} % が拾う物の外形の中 → 同じ物の別の mask として障害物にしない")
                    continue
                # 物の机に沿った長い向き（主成分）に合わせて回した箱（2026-11-25 Shreyan: 斜めに置いた 24 cm のバーは、軸に沿った箱だと
                # 27 × 12 cm になって隣の容器の掴み方の IK を 13 → 2 に減らした）。余白 1 cm・下端は机まで、は今までどおり
                c_xy = pb[:, :2].mean(0)
                w_, V_ = np.linalg.eigh(np.cov((pb[:, :2] - c_xy).T))
                ax_ = V_[:, 1]; yaw = float(np.arctan2(ax_[1], ax_[0]))
                Rz = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
                loc = (pb[:, :2] - c_xy) @ Rz  # 箱の向きの座標（u = 長い向き・v = 横）
                lo_uv, hi_uv = np.percentile(loc, 2, axis=0) - 0.01, np.percentile(loc, 98, axis=0) + 0.01
                z_lo, z_hi = min(float(np.percentile(pb[:, 2], 2)) - 0.01, fz), float(np.percentile(pb[:, 2], 98)) + 0.01
                ctr = c_xy + Rz @ ((lo_uv + hi_uv) / 2)
                obj_boxes.append({"name": f"obj_{k}", "dims": [round(float(v), 4) for v in (*(hi_uv - lo_uv), z_hi - z_lo)],
                                  "pose": [round(float(ctr[0]), 4), round(float(ctr[1]), 4), round((z_lo + z_hi) / 2, 4),
                                           round(float(np.cos(yaw / 2)), 6), 0.0, 0.0, round(float(np.sin(yaw / 2)), 6)]})
            curobo.set_obstacle_boxes(obj_boxes)
            r = {"boxes": obj_boxes, "n_in": 0, "n_after_arm": 0}
        elif a.ignore_seen:  # 2026-11-22: 周りの物に当たってもよいから目標を拾う方針。置き場の物（本体・カメラ・机・もう片方の腕）は見る
            curobo.set_obstacle_boxes([])
            r = {"boxes": [], "n_in": len(P_obs), "n_after_arm": 0}
        else:
            r = curobo.set_obstacle_points(P_obs, arm_qs, T_bc[:3, 3], fz, arm_margin=margin)
        bf = seen.BOXES_FILE if side == "left" else seen.BOXES_FILE.replace(".json", "_right.json")
        json.dump({"cam_pos": T_bc[:3, 3].tolist(), "floor_z": fz, "arm_qs": arm_qs, "boxes": r["boxes"], "arm": side,
                   "ignore_seen": bool(a.ignore_seen), "object_boxes": obj_boxes},
                  open(os.path.join(a.cap_dir, bf), "w"), indent=1)
        print(f"[plan:{side}] cuRobo: カメラで見た障害物の箱 {len(r['boxes'])} 個（点 {r['n_in']} → 腕を消して {r['n_after_arm']}）"
              + "".join(f" {b['name']} 中心 {np.round(b['pose'][:3], 2).tolist()} 大きさ {np.round(b['dims'], 2).tolist()}" for b in r["boxes"]))
    _t_filters = time.time()
    tip_off = CFG["flange_to_tip_m"]
    def lowest_corner(Tg):
        tip = Tg[:3, 3] + tip_off * Tg[:3, 2]
        cs = [tip + sx * 0.058 * Tg[:3, 0] + sy * 0.024 * Tg[:3, 1] for sx in (-1, 1) for sy in (-1, 1)]
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
    # prefer grasps that go for the object's centre (a corner grasp misses with even a small error)
    ob_c = obj.mean(0) @ T_bc[:3, :3].T + T_bc[:3, 3]
    tcp_all = np.einsum("nij,j->ni", Gb[:, :3, :3], [0, 0, CFG["flange_to_tip_m"] - 0.015]) + Gb[:, :3, 3]
    cdist = np.linalg.norm(tcp_all[:, :2] - ob_c[:2], axis=1)
    # 長い物（机に沿った縦横の比が 2 以上・寝かせたバーなど）は、長い向きのどこを掴んでもよい。中心からの距離の代わりに、長い向きの軸
    # からの距離を見て、両端の 3 cm は外す（2026-11-23: 27 cm のアルミのバーで、重心から 2 cm 以内の候補が 600 個中 57 個しか無かった）
    ob_xy = (obj @ T_bc[:3, :3].T + T_bc[:3, 3])[:, :2]
    _, sv, Vt = np.linalg.svd(ob_xy - ob_xy.mean(0), full_matrices=False)
    ax = Vt[0]
    proj = (ob_xy - ob_xy.mean(0)) @ ax
    L = float(np.percentile(proj, 98) - np.percentile(proj, 2))
    W = float(np.percentile((ob_xy - ob_xy.mean(0)) @ Vt[1], 98) - np.percentile((ob_xy - ob_xy.mean(0)) @ Vt[1], 2))
    if W > 0.06:
        # 爪に丸ごと入らない太い物（テープの輪・上の開いた箱）: 縁や壁を掴むので、重心から離れていてよい。掴む所の真下に物の点が
        # あるか（いちばん近い物の点まで水平に 1.5 cm 以内）で見る（2026-11-23: 輪の縁は重心から約 5 cm で、全部の候補が落ちた）
        sub = ob_xy[:: max(1, len(ob_xy) // 600)]
        near = np.full(len(tcp_all), np.inf)
        for k0 in range(0, len(tcp_all), 400):
            dd = np.linalg.norm(tcp_all[k0:k0 + 400, None, :2] - sub[None, :, :2], axis=2)
            near[k0:k0 + 400] = dd.min(1)
        cdist = np.where(near <= 0.015, 0.0, np.inf)
        print(f"[plan] 太い物（{L * 1000:.0f}×{W * 1000:.0f} mm）: 中心の代わりに、掴む所の真下に物があるかで見る"
              f"（{int(np.isfinite(cdist).sum())} 個）")
    elif L > 2 * max(W, 1e-3):
        rel_xy = tcp_all[:, :2] - ob_xy.mean(0)
        along = rel_xy @ ax
        perp = np.abs(rel_xy @ Vt[1])
        cdist = np.where(np.abs(along) <= L / 2 - 0.03, perp, np.inf)
        print(f"[plan] 長い物（{L * 1000:.0f}×{W * 1000:.0f} mm）: 中心の代わりに長い向きの軸からの距離で見る"
              f"（{int(np.sum(cdist <= a.center_tol))} 個が {a.center_tol * 100:.0f} cm 以内）")
    # object centre in each grasp's own frame: closing (Xg) and thickness (Yg) offsets from the TCP.
    # The horizontal check above misses thickness offsets of tilted grasps (iter_034/036: misses had the block
    # 5-7 cm off along the jaw thickness, the jaws passed in front of it).
    rel = np.einsum("nji,nj->ni", Gb[:, :3, :3], ob_c[None, :] - tcp_all)
    toff, coff = np.abs(rel[:, 1]), np.abs(rel[:, 0])
    # 開いた爪の内側と物のすき間（2026-11-21 北九州）: 爪の全開が 80 mm だと、直径 5 cm の木片を斜めに挟む候補は
    # すき間が片側 5 mm 以下になり、腕の 1 cm 前後の誤差で爪が物に当たる（probe で木片が 17 mm 動いた）。
    # 爪の厚み（±17.7 mm + 5 mm）と長さ（指先から 80 mm）の範囲にある物の点の、閉じる向きの両端と開いた爪の内側の距離の小さい方
    # 2026-11-23: 爪のすき間と挟む深さは重い（1500 個で 1.4 秒）ので、安い絞り（傾き・床・中心・点数）を通った候補だけで計算する
    cheap = np.flatnonzero((tilt <= a.max_tilt) & (clear >= a.floor_clear) & (cdist <= a.center_tol) & (S >= a.min_score))
    jclr = np.full(len(Gb), np.inf)
    if a.jaw_clear is not None:
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        # 指令で開いたときの爪の内側の間 [m]（腕ごと・GRASP_GRIPPER_OPEN_WIDTH(_RIGHT)、無ければ全開の max_opening_m）。
        # 2026-11-22: 右は -56 の指令（読み -54.5）で 72 mm しか開かず（指は平行・定規で実測）、80 mm と見て片側 4 mm 多く見積もっていた。
        # 両腕のときは狭い方で確かめる（候補の並びは左右で同じ物を使うので）
        half = min(float(armcfg.env("GRASP_GRIPPER_OPEN_WIDTH", sd, str(CFG["max_opening_m"]))) for sd in sides) / 2
        jclr[:] = -np.inf  # 計算しない候補は通さない（安い絞りで落ちる）
        for i in cheap:
            jclr[i] = np.inf
            q = (ob_b - tcp_all[i]) @ Gb[i][:3, :3]  # 指先の中心から見た（閉じる・厚み・近づく）
            jclr[i] = jaw_clearance(q, half)  # 爪の板の外より遠い点（輪の向こう側の壁など）は数えない（grasp/depth.py）
        print(f"[plan] jaw clearance: {int(np.sum(jclr >= a.jaw_clear))} grasps keep ≥{a.jaw_clear*1000:.0f} mm between the open jaws "
              f"(inner gap {half * 2000:.0f} mm) and the object")
    # 挟む深さ（2026-11-23・作業者 Munenori）: GraspGenX は爪の先端だけで物に触る掴み方も出し、点数の順では浅いのが選ばれる
    depth = np.full(len(Gb), np.inf)
    if a.min_depth > 0 and a.min_depth_frac > 0:
        ob_h = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        h_obj = float(np.percentile(ob_h @ n_b + d_b, 98))
        if h_obj * a.min_depth_frac < a.min_depth:
            print(f"[plan] 物の高さ {h_obj * 1000:.0f} mm → 挟む深さの下限を {a.min_depth * 1000:.0f} から "
                  f"{h_obj * a.min_depth_frac * 1000:.0f} mm に下げる")
            a.min_depth = h_obj * a.min_depth_frac
    if a.min_depth > 0:
        ob_d = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        half_d = min(float(armcfg.env("GRASP_GRIPPER_OPEN_WIDTH", sd, str(CFG["max_opening_m"]))) for sd in sides) / 2
        depth = np.zeros(len(Gb))
        if len(cheap):
            depth[cheap] = grasp_depths(Gb[cheap], ob_d, half_d, CFG["flange_to_tip_m"])
        print(f"[plan] grasp depth: {int(np.sum(depth >= a.min_depth))} of {len(cheap)} grasps hold the object ≥{a.min_depth*1000:.0f} mm "
              f"behind the tips (median {np.median(depth[cheap]) * 1000 if len(cheap) else 0:.0f} mm)")
    hit = np.zeros(len(Gb), bool)
    if a.avoid_obstacles:
        # Other things on the floor next to the target (iter_021: the left jaw came down on a wood block touching
        # the aluminium bar and the arm was pushed off). Obstacles = above-floor scene points not in the target.
        from scipy.spatial import cKDTree
        from grasp.capture import depth_to_points
        Kc = np.array(json.load(open(os.path.join(a.cap_dir, "intrinsics.json")))["K"])
        sp, _ = depth_to_points(np.load(os.path.join(a.cap_dir, "depth.npy")), Kc, zmax=1.0)
        sb = sp @ T_bc[:3, :3].T + T_bc[:3, 3]
        hgt = sb @ n_b + d_b
        sb = sb[(hgt > 0.008) & (hgt < 0.25)]
        ob_b = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
        dist, _ = cKDTree(ob_b).query(sb, k=1)
        obst = sb[dist > 0.012]
        obst = obst[np.linalg.norm(obst[:, :2] - ob_b.mean(0)[:2], axis=1) < 0.30]
        half_open = 0.0465   # jaw inner faces at lerobot -60 deg (wrist camera: tips 93 mm apart)
        fx, fy, fl, mg = 0.022, 0.019, 0.050, a.obstacle_margin  # finger thickness (X), half width (Y), length back from tip (Z), margin
        tipz = CFG["flange_to_tip_m"]
        for i in range(len(Gb)):
            if len(obst) == 0:
                break
            q = (obst - Gb[i][:3, 3]) @ Gb[i][:3, :3]  # obstacle points in the grasp frame
            inx = (np.abs(q[:, 0]) >= half_open - mg) & (np.abs(q[:, 0]) <= half_open + fx + mg)
            iny = np.abs(q[:, 1]) <= fy + mg
            inz = (q[:, 2] >= tipz - fl - a.pregrasp - mg) & (q[:, 2] <= tipz + mg)  # swept from the pregrasp
            hit[i] = np.sum(inx & iny & inz) > 8
        print(f"[plan] obstacles: {len(obst)} points near the target; {int(hit.sum())} grasps would hit them")
        clear = np.where(hit, -1.0, clear)
    print(f"[plan] filters: tilt ok {int(np.sum(tilt <= a.max_tilt))}, floor ok {int(np.sum(clear >= a.floor_clear))}, "
          f"centre ok {int(np.sum(cdist <= a.center_tol))}, score ok {int(np.sum(S >= a.min_score))}")
    order = [i for i in np.argsort(-S) if tilt[i] <= a.max_tilt and clear[i] >= a.floor_clear and cdist[i] <= a.center_tol
             and (a.thick_tol is None or (toff[i] <= a.thick_tol and coff[i] <= a.center_tol))
             and (a.jaw_clear is None or jclr[i] >= a.jaw_clear)
             and depth[i] >= a.min_depth
             and S[i] >= a.min_score]
    timing.event("plan/candidate filters (tilt・floor・centre・jaw clearance・depth)", _t_filters, time.time(),
                 candidates=len(S), kept=len(order))
    print(f"[plan] floor normal (base) {np.round(n_b,3).tolist()}; {int(np.sum(clear >= a.floor_clear))} candidates keep ≥{a.floor_clear*100:.0f} cm clearance")
    print(f"[plan] {len(S)} candidates, {len(order)} from above (≤{a.max_tilt:.0f}°)")

    q_left = [float(v) for v in a.q.split(",")]

    def start_q(side):
        if side == "left":
            return q_left
        if a.q_right:
            return [float(v) for v in a.q_right.split(",")]
        return armcfg.default_rest("right") or armcfg.mirror_q(q_left).tolist()

    def prefer_q(side):
        p = [float(v) for v in a.prefer.split(",")]
        if side == "left":
            return p
        return [float(v) for v in a.prefer_right.split(",")] if a.prefer_right else armcfg.mirror_q(p).tolist()

    def search(side):
        """その腕で、点数の順に pregrasp・grasp・lift が全部解ける最初の候補を探す → result（無ければ None）。"""
        os.environ["GRASP_ARM"] = side
        _t = time.time()
        prepare_curobo(side)
        timing.event(f"plan/{side}/curobo set obstacles", _t, time.time())
        T_L7_G = t_l7_g(side)
        _t = time.time()
        ik = OpenArmIK(side=side)
        timing.event(f"plan/{side}/build pinocchio IK", _t, time.time())
        q0 = np.deg2rad(start_q(side))
        rng = np.random.default_rng(0)
        mid = armcfg.mirror_q if side == "right" else (lambda x: np.asarray(x, float))
        seeds = [q0, np.deg2rad(mid([0, 0, 0, 60, 0, 0, 0])), np.deg2rad(mid([-30, 0, 0, 90, 0, 0, 0]))] + \
                [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(4)]
        # 置き場で確かめた姿勢を IK の初期値に足す（GRASP_IK_SEEDS(_RIGHT)="j1,...,j7;..." [deg]）。2026-11-22: 右腕は休みの姿勢などからだと
        # 真上から掴む解が J1 50° 前後の姿勢に落ち、GRASP_IK_LIMITS_RIGHT の中の解（手で教えた姿勢の仲間）が見つからなかった
        for v in (armcfg.env("GRASP_IK_SEEDS", side) or "").split(";"):
            if v.strip():
                seeds.insert(0, np.deg2rad([float(x) for x in v.split(",")]))
        q_pref = np.deg2rad(prefer_q(side))
        seeds.insert(0, q_pref)
        with timing.stage(f"plan/{side}/search (IK + collision per candidate)"):
            return _search(side, ik, T_L7_G, q0, q_pref, seeds)

    def ik_box(side):
        """計画で使ってよい関節の範囲（GRASP_IK_LIMITS(_RIGHT)="関節番号:下限:上限;..." [deg]）。無ければ可動域だけ。
        2026-11-22 北九州: 右腕で真上から掴むと J1 が 43〜54° の姿勢を選び、指先が左へ 16〜25 mm・3 cm 低くずれて 4 回とも木材に当たった。
        マーカーで確かめた範囲（J1 -36〜+32°）では 3 mm で合っているので、確かめた範囲の中の解だけを使う（作業者 Munenori）。"""
        lo, hi = np.full(7, -np.inf), np.full(7, np.inf)
        for part in (armcfg.env("GRASP_IK_LIMITS", side) or "").split(";"):
            if part.strip():
                j, a_, b_ = part.split(":")
                lo[int(j) - 1], hi[int(j) - 1] = np.radians(float(a_)), np.radians(float(b_))
        return lo, hi

    def use_curobo_ik():
        return a.ik == "curobo" and curobo.enabled()

    def curobo_ik_all(side, ik, T_L7_G, q_pref, box_lo, box_hi):
        """候補の全部（点数の順 × 爪の向き 2 通り）の掴む・手前・持ち上げの姿勢を、cuRobo の IK でまとめて解く（2026-11-22）。
        掴む姿勢は 1 つの目標に 8 つの解をもらい、可動域・関節の範囲の絞り（ik_box）の中で好む姿勢に一番近いものを選ぶ。
        手前と持ち上げは、選んだ掴む姿勢を初期値にして解き、掴む姿勢に一番近い解を選ぶ（同じ枝に乗せる）。
        返り値 {(rank, flip): [q_grasp, q_pre, q_lift]（rad）}。周りの物に当たる解は cuRobo が成功にしない。"""
        import time as _t
        t0 = _t.time()
        keys, Tg_l7, Tpre_l7, Tlift_l7 = [], [], [], []
        inv = np.linalg.inv(T_L7_G)
        for rank, i in enumerate(order[:a.max_try]):
            for flip in (False, True):
                Tg = Gb[i] @ (FLIP if flip else np.eye(4))
                Tpre = Tg.copy(); Tpre[:3, 3] -= a.pregrasp * Tg[:3, 2]
                Tlift = Tg.copy(); Tlift[:3, 3] += [0, 0, a.lift]
                keys.append((rank, flip)); Tg_l7.append(Tg @ inv); Tpre_l7.append(Tpre @ inv); Tlift_l7.append(Tlift @ inv)
        if not keys:
            return {}
        pref = np.rad2deg(q_pref)

        def pick(Q, OK, ref, box=True):
            best = None
            for q, ok in zip(Q, OK):
                r = np.deg2rad(q)
                if ok and ik.within_limits(r) and (not box or (np.all(r >= box_lo) and np.all(r <= box_hi))):
                    d = np.max(np.abs(q - ref))
                    if best is None or d < best[0]:
                        best = (d, r)
            return None if best is None else best[1]
        if os.environ.get("GRASP_IK_DUMP"):  # 調べる用: 解いた掴む姿勢の link7 の目標を残す
            np.save(os.environ["GRASP_IK_DUMP"], np.array(Tg_l7))
        def solve_all(lm_only):
            """lm_only: cuRobo の最初の LM の段だけで終える（2026-11-23 Munenori）。解けない候補が 1 つでも混ざると、全部まわす場合は
            L-BFGS を最後までまわして 1 回 0.7〜2.8 秒かかる。LM の段だけなら 50 個で 0.02 秒で、解ける候補はほぼ同じ（4 回中 1 回、際どい 1 個を落とした）"""
            Qg, OKg = curobo.ik(np.array(Tg_l7), return_seeds=8, lm_only=lm_only)
            g = [pick(Qg[n], OKg[n], pref) for n in range(len(keys))]
            n_raw = int(OKg.any(1).sum())
            n_lim = int(sum(any(ok and ik.within_limits(np.deg2rad(q)) for q, ok in zip(Qg[n], OKg[n])) for n in range(len(keys))))
            idx = [n for n in range(len(keys)) if g[n] is not None]
            out = {}
            if idx:
                seeds_deg = np.rad2deg(np.array([g[n] for n in idx]))
                # 手前と持ち上げは 1 回にまとめて解く（前は 2 回）
                Qpl, OKpl = curobo.ik(np.array([Tpre_l7[n] for n in idx] + [Tlift_l7[n] for n in idx]),
                                      seed_deg=np.concatenate([seeds_deg, seeds_deg]), return_seeds=8, lm_only=lm_only)
                # return_seeds は掴む姿勢と同じ 8 にする（変えるとサーバーが CUDA graph を作り直して 1 回 0.18 秒かかる）
                M = len(idx)
                for m, n in enumerate(idx):
                    # 関節の範囲の絞りは、指先の位置が大事な掴む姿勢だけに掛ける（手前・持ち上げは空中で、掴む姿勢の近くの解ならよい）
                    qp = pick(Qpl[m], OKpl[m], seeds_deg[m], box=False)
                    ql = pick(Qpl[M + m], OKpl[M + m], seeds_deg[m], box=False)
                    if qp is not None and ql is not None:
                        out[keys[n]] = [g[n], qp, ql]
            print(f"[plan:{side}] cuRobo IK{'（LM の段だけ）' if lm_only else '（全部まわす）'}: 候補 {len(keys)} 個"
                  f"（掴む姿勢: 解けた {n_raw}・可動域の中 {n_lim}・絞りの中 {len(idx)}・3 つとも {len(out)}）（{_t.time() - t0:.2f} 秒）")
            return out
        _tik = time.time()
        out = solve_all(True)
        timing.event(f"plan/{side}/curobo batch IK", _tik, time.time(), lm_only=True, targets=len(keys), solved=len(out))
        if not out:  # LM の段で 1 つも残らなければ、L-BFGS まで全部まわして解き直す
            _tik = time.time()
            out = solve_all(False)
            timing.event(f"plan/{side}/curobo batch IK", _tik, time.time(), lm_only=False, targets=len(keys), solved=len(out))
        return out

    def _search(side, ik, T_L7_G, q0, q_pref, seeds):
        box_lo, box_hi = ik_box(side)

        def solve(T, first_seed):
            best = None
            for sd in [first_seed] + seeds:
                q, ok, _, _ = ik.solve(se3(T), sd, position_only=False, max_iter=200)
                if ok and ik.within_limits(q) and np.all(q >= box_lo) and np.all(q <= box_hi):
                    d = np.max(np.abs(q - (first_seed if first_seed is not q0 else q_pref)))
                    if best is None or d < best[0]:
                        best = (d, q)
            return None if best is None else best[1]
        result = None
        n_keepout = n_curobo = n_descent = 0
        pre_ik = curobo_ik_all(side, ik, T_L7_G, q_pref, box_lo, box_hi) if use_curobo_ik() else None
        for rank, i in enumerate(order[:a.max_try]):
            for flip in (False, True):
                Tg = Gb[i] @ (FLIP if flip else np.eye(4))
                Tpre = Tg.copy(); Tpre[:3, 3] -= a.pregrasp * Tg[:3, 2]
                Tlift = Tg.copy(); Tlift[:3, 3] += [0, 0, a.lift]
                qs, ok_all, seed = [], True, q0
                if pre_ik is not None:  # cuRobo の IK でまとめて解いた答え（掴む・手前・持ち上げ）
                    qs = pre_ik.get((rank, flip))
                    ok_all = qs is not None
                    qs = qs or []
                for T in ((Tg, Tpre, Tlift) if pre_ik is None else ()):  # solve the grasp first, then seed the others from it (keeps joint changes small)
                    q = solve(T @ np.linalg.inv(T_L7_G), seed if not qs else qs[0])
                    if q is None:
                        ok_all = False; break
                    qs.append(q)
                if ok_all:
                    # pre/lift must be close to the grasp on the same IK branch (a branch jump means a big swing while near the object)
                    if max(np.max(np.abs(np.rad2deg(qs[1] - qs[0]))), np.max(np.abs(np.rad2deg(qs[2] - qs[0])))) > a.max_branch_deg:
                        ok_all = False
                if ok_all and KEEPOUT and any(in_keepout(ik, q) for q in qs):
                    ok_all = False  # 腕が胸のカメラと取り付け部の箱に入る解（本体のフレームに当たる）
                    n_keepout += 1
                if ok_all and curobo.enabled():
                    # cuRobo の球の腕で、pre → grasp → lift を関節の直線で動かす途中が置き場の箱（本体のフレーム・机）に当たらないか
                    # （2026-11-21 北九州: 持ち上げの段で肘が本体のフレームに当たった。腕の点だけを見る keep-out の箱を置き換える）
                    _t = time.time()
                    bad = curobo.violations([np.rad2deg(qs[1]), np.rad2deg(qs[0]), np.rad2deg(qs[2])],
                                            names=["pregrasp→grasp", "grasp→lift"], n=20)
                    timing.event(f"plan/{side}/curobo path check (one candidate)", _t, time.time(), rank=rank, ok=not bad)
                    if bad:
                        ok_all = False
                        n_curobo += 1
                        if n_curobo <= 3:
                            print(f"[plan:{side}] cuRobo: 候補 {rank} を外す（{curobo.describe(bad)}）")
                if ok_all and a.check_descent:  # qs はまだ [掴む, 手前, 持ち上げ] の順
                    _td = time.time()
                    if not (line_ok(ik, qs[1], qs[0]) and line_ok(ik, qs[0], qs[2])):
                        ok_all = False
                        n_descent += 1
                    timing.event(f"plan/{side}/straight descent check (one candidate)", _td, time.time(), rank=rank, ok=ok_all)
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
                if ok_all:
                    qs = [qs[1], qs[0], qs[2]]  # → order: pre, grasp, lift
                if ok_all:
                    result = dict(rank=int(rank), idx=int(i), score=float(S[i]), tilt_deg=float(tilt[i]), flip=flip,
                                  T_grasp_base=Tg.tolist(), q_pre_deg=np.rad2deg(qs[0]).tolist(),
                                  q_grasp_deg=np.rad2deg(qs[1]).tolist(), q_lift_deg=np.rad2deg(qs[2]).tolist(),
                                  q_start_deg=np.rad2deg(q0).tolist(),
                                  q_look_deg=None if q_look is None else np.rad2deg(q_look).tolist(), look_back_m=a.look_back,
                                  depth_m=None if not np.isfinite(depth[i]) else float(depth[i]),
                                  pregrasp_m=float(a.pregrasp), lift_m=float(a.lift))
                    break
            if result:
                break
        if a.check_descent:
            print(f"[plan:{side}] straight descent: {n_descent} candidates dropped (pregrasp→grasp or grasp→lift is not a straight line the arm can follow)")
        if KEEPOUT:
            print(f"[plan:{side}] keep-out: {n_keepout} IK solutions dropped (the arm enters a keep-out box: {len(KEEPOUT)} box(es))")
        if curobo.enabled():
            print(f"[plan:{side}] cuRobo: {n_curobo} IK solutions dropped (the arm spheres come within {curobo.MARGIN*1000:.0f} mm of a box)")
        if result:
            result["arm"] = side
        return result

    _t = time.time()
    results = run_sides(search, sides)
    timing.event("plan/search all arms", _t, time.time(), arms=",".join(sides))
    if a.short_fallback and not any(results.values()):
        a.pregrasp, a.lift = (float(v) for v in a.short_fallback.split(","))
        print(f"[plan] 手前 10 cm・持ち上げ 10 cm では候補が無い。手前 {a.pregrasp * 100:.0f} cm・持ち上げ {a.lift * 100:.0f} cm で解き直す")
        _t = time.time()
        results = run_sides(search, sides)
        timing.event("plan/search all arms (short pregrasp/lift)", _t, time.time(), arms=",".join(sides))
    for side, r in results.items():
        if r is not None:
            json.dump(r, open(os.path.join(a.cap_dir, f"plan_{side}.json"), "w"), indent=2)
        print(f"[plan:{side}] " + ("候補なし" if r is None else
              f"rank {r['rank']} score {r['score']:.2f} tilt {r['tilt_deg']:.0f}° 関節の動き {travel_deg(r):.0f}°"))
    result = choose_arm(results, a.arm_tie)
    if not result:
        print("[plan] no candidate where all of pregrasp, grasp and lift solve"); return 1
    if len(sides) > 1:
        print(f"[plan] 腕を選んだ: {result['arm']}（" + "・".join(
            f"{sd} {'なし' if r is None else format(r['score'], '.2f')}" for sd, r in results.items()) + "）")
    # sanity: does the TCP land on the object?
    tcp = (np.array(result["T_grasp_base"]) @ np.r_[0, 0, CFG["flange_to_tip_m"] - 0.015, 1])[:3]
    ob = obj @ T_bc[:3, :3].T + T_bc[:3, 3]
    d = float(np.min(np.linalg.norm(ob - tcp, axis=1)))
    result["tcp_base"] = tcp.tolist(); result["tcp_to_object_m"] = d
    json.dump(result, open(os.path.join(a.cap_dir, "plan.json"), "w"), indent=2)
    print(f"[plan] chosen: {result['arm']} arm, rank {result['rank']} score {result['score']:.2f} tilt {result['tilt_deg']:.0f}° flip={result['flip']}"
          + (f" depth {result['depth_m']*1000:.0f} mm" if result.get('depth_m') is not None else ""))
    for k in ("q_start_deg", "q_pre_deg", "q_grasp_deg", "q_lift_deg"):
        print(f"  {k:12s} " + " ".join(f"{v:+6.1f}" for v in result[k]))
    print(f"  TCP {np.round(tcp,3).tolist()}  distance to nearest object point {d*1000:.0f} mm")
    if not a.no_render:  # 図（plan.png）は腕のメッシュの読み込みと描画で 2〜3 秒かかる（2026-11-21 北九州で測った）
        _t = time.time()
        render(a.cap_dir, result, ob)
        timing.event("plan/render plan.png", _t, time.time())
    return 0


def line_ok(ik, q_a, q_b, steps: int = 10, max_jump_deg: float = 10.0) -> bool:
    """q_a の手先から q_b の手先へ、手先の姿勢を空間の直線（向きは測地線）で steps 等分して、前の段から IK で解いて
    たどれるか（どの段も可動域の中・1 段で max_jump_deg を超えて跳ばない）。run_grasp.py の cart_line と同じ考え方。
    終わりの関節角は q_b と比べない（2026-11-25 Shreyan: 7 関節の腕は同じ手先の姿勢を肘の違う角度でも取れる。cuRobo が別に解いた持ち上げの
    姿勢と 6.6° 違うだけで、たどれる直線の候補を全部落としていた）"""
    M0, M1 = ik.fk(q_a), ik.fk(q_b)
    dv = pin.log6(M0.inverse() * M1)
    prev = np.asarray(q_a, float)
    for f in np.linspace(0.0, 1.0, steps + 1)[1:]:
        q, ok, _, _ = ik.solve(M0 * pin.exp6(dv * f), prev, position_only=False, max_iter=200)
        if not ok or not ik.within_limits(q) or np.max(np.abs(np.rad2deg(q - prev))) > max_jump_deg:
            return False
        prev = q
    return True


def travel_deg(r) -> float:
    """今の姿勢から pregrasp までの関節の動きの一番大きいもの [deg]。"""
    return float(np.max(np.abs(np.asarray(r["q_pre_deg"]) - np.asarray(r["q_start_deg"]))))


def choose_arm(results: dict, tie: float = 0.05):
    """腕ごとの結果から 1 つ選ぶ。点数の高い方。差が tie 以内なら関節の動きが小さい方。どれも無ければ None。"""
    ok = [r for r in results.values() if r is not None]
    if not ok:
        return None
    best = max(r["score"] for r in ok)
    near = [r for r in ok if r["score"] >= best - tie]
    return min(near, key=travel_deg)


def run_sides(fn, sides) -> dict:
    """腕ごとに fn(side) を回す。2 本なら別のプロセスで同時に（cuRobo のクライアントと IK を腕ごとに分けるため）。"""
    if len(sides) == 1:
        return {sides[0]: fn(sides[0])}
    if os.environ.get("GRASP_PLAN_SEQUENTIAL"):
        # 常駐のプロセス（resident_plan.py の run・カメラのスレッドを持つ）の中では fork しないで順に回す（2026-11-25 Shreyan）。
        # 結果は同じ（腕ごとに search が GRASP_ARM と cuRobo のサーバーを切り替える）。片方の誤りで両方を落とさないのも同じ
        out = {}
        for sd in sides:
            try:
                out[sd] = fn(sd)
            except Exception as e:  # noqa: BLE001
                print(f"[plan:{sd}] 計算が止まった: {type(e).__name__}: {e}")
                out[sd] = None
        return out
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    q = ctx.Queue()

    def work(side):
        try:
            q.put((side, fn(side), None))
        except BaseException as e:  # noqa: BLE001  片方の誤りで両方を落とさない
            q.put((side, None, f"{type(e).__name__}: {e}"))

    ps = [ctx.Process(target=work, args=(sd,)) for sd in sides]
    for p in ps:
        p.start()
    out = {}
    for _ in ps:
        side, r, err = q.get()
        if err:
            print(f"[plan:{side}] 計算が止まった: {err}")
        out[side] = r
    for p in ps:
        p.join()
    return {sd: out.get(sd) for sd in sides}


def render(cap_dir, r, ob):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from grasp.urdf_fk import LeftArmFK
    side = r.get("arm", "left")
    fk = LeftArmFK(side=side); links = [f"openarm_{side}_link{i}" for i in range(1, 8)]
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
        L7 = P[f"openarm_{side}_link7"]; tcp_now = (L7 @ np.r_[0, 0, 0.2145, 1])[:3]
        ax.scatter(*tcp_now, s=40, c="k", marker="x", label="TCP in this pose")
        ax.set_title(title); ax.set_xlabel("x fwd"); ax.set_ylabel("y left"); ax.set_zlabel("z up")
        allp = np.vstack([pts, ob]); m = allp.mean(0); rr = np.ptp(allp, 0).max() / 2
        ax.set_xlim(m[0] - rr, m[0] + rr); ax.set_ylim(m[1] - rr, m[1] + rr); ax.set_zlim(m[2] - rr, m[2] + rr)
        ax.view_init(elev=20, azim=-60); ax.legend(loc="upper left", fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(cap_dir, "plan.png"), dpi=85)


if __name__ == "__main__":
    raise SystemExit(main())
