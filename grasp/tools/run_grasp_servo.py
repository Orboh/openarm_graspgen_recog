#!/usr/bin/env python3
"""Step 6 with wrist-camera alignment (runs on Thor, lerobot venv). **The arm moves.** Dry run by default.

Why: 3 real trials (11/20) missed by 2-8 cm although every joint reached its target within 1-2 deg.
The arm model (wrist joint zeros / jaw position in link7) is off by a pose-dependent few cm. The left
wrist D405 is rigid with the jaws, so "where is the block relative to the jaw centre" can be measured
directly at the pregrasp and corrected, without trusting the arm model.

Sequence:
  start -> raise 8 cm -> open -> pregrasp
  (v2, after iter_010/014: at the pregrasp the far side of the block is always cut off by the image top)
  -> look pose = pregrasp + 8 cm further back along the approach, whole block in view
  -> [jog +1.5 cm along the grasp closing axis Xg and the thickness axis Yg, measure, back]
     => block shift per metre of gripper motion = the camera's view of Xg, Yg (no extrinsic needed)
  -> lateral block-vs-jaw-centre offset, damped x0.8, up to 3x (<= 6 cm total)
  -> depth along the approach from the same camera, jaw tips kept >= 8 mm above the camera-measured floor
  -> corrected pregrasp
  -> approach (6 deg/s) -> close -> lift 10 cm, hold 2 s -> put back -> release -> pregrasp -> raise -> start

The motion helpers (soft move with gravity feed-forward, residual-error correction, floor checks) are an
inline copy of grasp/run_grasp.py: there they are closures inside main(), so they can't be imported.

    ~/projects/lerobot/.venv/bin/python grasp/run_grasp_servo.py <cap_dir>          # dry: floor checks only
    ~/projects/lerobot/.venv/bin/python grasp/run_grasp_servo.py <cap_dir> --go
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.expanduser("~/projects/openarm-orboh")
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
G_OPEN = float(os.environ.get("GRASP_GRIPPER_OPEN", "-60.0")); G_CLOSE = float(os.environ.get("GRASP_GRIPPER_CLOSE", "2.0"))
FLOOR_Z = float(os.environ.get("GRASP_FLOOR_Z", "0.27"))
if os.environ.get("GRASP_SITE") and "GRASP_FLOOR_Z" not in os.environ:  # 東京の台車以外で床の高さを決め忘れたら動かさない
    sys.exit("[grasp] GRASP_FLOOR_Z が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の値のまま床の確認をしない。grasp/config/sites/ で決める")
HERE = os.path.dirname(os.path.abspath(__file__))
_JP = [os.path.join(HERE, "config", "wrist_jaw_model.json"), os.path.join(HERE, "..", "config", "wrist_jaw_model.json")]
JAW = json.load(open(next(p for p in _JP if os.path.exists(p))))  # deployed next to grasp/ on Thor, or in grasp/tools locally
JOG_M = 0.015
MAX_CORR_M = 0.05
TOL_M = 0.005
LOOK_BACK = 0.08   # (v3/v4, unused by v5)
NEAR_BACK = 0.04   # v5: measure/correct at the grasp moved this much back along the approach
EXPECT_ALONG_NEAR_M = NEAR_BACK - 0.045  # block top vs jaw tips along the approach at the near pose
TIP_MAX_M = 0.035  # jaw tips must get at least this low (above the floor) or we don't close (block ~5-6 cm tall)
GAIN = 0.8         # damped correction (cut-off views made full steps oscillate, 11/20 iter_014)
FLOOR_MIN_M = 0.008
# at the look pose (pregrasp + 8 cm back) the block top should sit this far beyond the jaw tips along the approach:
# TCP (= tips - 15 mm) lands on the block centre (= top - 30 mm) -> tips 45 mm past the top at the grasp
EXPECT_ALONG_LOOK_M = LOOK_BACK + 0.10 - 0.045


# ------------------------------------------------------------------ wrist camera
class Wrist:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.p = rs.pipeline(); c = rs.config(); c.enable_device(JAW["serial"])
        c.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        c.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        pr = self.p.start(c)
        self.scale = pr.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        for _ in range(20):
            self.p.wait_for_frames()

    def grab(self, n=5):
        ds = []
        for _ in range(n):
            fs = self.align.process(self.p.wait_for_frames())
            ds.append(np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.scale)
        col = np.asanyarray(fs.get_color_frame().get_data()).copy()
        d = np.median(np.stack(ds), 0)
        return col, d

    def stop(self):
        self.p.stop()


def wood_mask(col):
    """Wood-coloured pixels (brown hue, saturated), dilated. Same thresholds as thor_snap --exclude-wood."""
    import cv2
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    w = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 30)).astype(np.uint8)
    w = cv2.morphologyEx(w, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(w, np.ones((9, 9), np.uint8)).astype(bool)


def floor_fit(d):
    """Floor plane (n, d) in the wrist camera frame, normal toward the camera: RANSAC on the far 60% of points."""
    I = JAW["intrinsics"]
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    P = np.dstack([(u - I["ppx"]) / I["fx"] * d, (v - I["ppy"]) / I["fy"] * d, d])
    valid = (d > 0.04) & (d < 0.6)
    far = P[valid & (d > np.percentile(d[valid], 40))][::7]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(300):
        s_ = far[rng.choice(len(far), 3, replace=False)]
        n = np.cross(s_[1] - s_[0], s_[2] - s_[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; dd = -n @ s_[0]
        k = int(np.sum(np.abs(far @ n + dd) < 0.006))
        if k > best[0]:
            best = (k, (n, dd))
    n, dd = best[1]
    if dd < 0:
        n, dd = -n, -dd
    return n, float(dd)


def bright_offset(d, col, A_, C_):
    """Aluminium (bright, unsaturated, often no depth) between the jaws: closing-axis offset of the bar centre
    from the jaw-tip midpoint, from the bar pixels in the rows around the jaw tips. Depth only for scale."""
    import cv2
    I = JAW["intrinsics"]; mid = np.array(JAW["mid_tips_cam_m"])
    H, W = d.shape; v, u = np.mgrid[0:H, 0:W]
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    jaw = (v > 120) & ((u < JAW["left_tip_px"][0] + 25) | (u > JAW["right_tip_px"][0] - 25))
    tv = (JAW["left_tip_px"][1] + JAW["right_tip_px"][1]) // 2
    band = (hsv[..., 2] >= 200) & (hsv[..., 1] <= 45) & ~jaw & ~wood_mask(col) & (v >= tv - 80) & (v <= tv + 40)
    if band.sum() < 200:
        return None
    uc, vc = float(np.median(u[band])), float(np.median(v[band]))
    dz = d[band]; ok = (dz > 0.04) & (dz < 0.4)
    Z = float(np.median(dz[ok])) if ok.sum() > 50 else float(mid[2] + 0.02)
    p = np.array([(uc - I["ppx"]) / I["fx"] * Z, (vc - I["ppy"]) / I["fy"] * Z, Z])
    n, dd = floor_fit(d)
    tips = [np.array(JAW["left_tip_cam_m"]), np.array(JAW["right_tip_cam_m"])]
    return {"e_c": float(p @ C_ - mid @ C_), "e_t": 0.0, "e_a": float("nan"), "cut_top": False,
            "tip_floor_m": min(float(q @ n + dd) for q in tips), "drop_per_m": float(-(A_ @ n)),
            "px": int(band.sum()), "u": uc}


def bright_mask(col):
    """Aluminium-like pixels (bright, unsaturated), dilated. Same thresholds as thor_snap --exclude-bright."""
    import cv2
    hsv = cv2.cvtColor(col, cv2.COLOR_BGR2HSV)
    b = ((hsv[..., 2] >= 200) & (hsv[..., 1] <= 45)).astype(np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.dilate(b, np.ones((7, 7), np.uint8)).astype(bool)


def block_in_wrist(d, col=None, exclude_wood=False, exclude_bright=False):
    """Block top-surface centroid in the wrist camera frame (m), or None. Floor = RANSAC plane of the far
    points; block = the largest above-floor connected component outside the jaw bands."""
    import cv2
    I = JAW["intrinsics"]
    H, W = d.shape
    v, u = np.mgrid[0:H, 0:W]
    P = np.dstack([(u - I["ppx"]) / I["fx"] * d, (v - I["ppy"]) / I["fy"] * d, d])
    valid = (d > 0.04) & (d < 0.6)
    far = P[valid & (d > np.percentile(d[valid], 40))][::7]
    rng = np.random.default_rng(0)
    best = (0, None)
    for _ in range(300):
        s = far[rng.choice(len(far), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0]); nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn; dd = -n @ s[0]
        k = int(np.sum(np.abs(far @ n + dd) < 0.006))
        if k > best[0]:
            best = (k, (n, dd))
    n, dd = best[1]
    if dd < 0:
        n, dd = -n, -dd
    h = P @ n + dd
    above = valid & (h > 0.015)
    if exclude_wood and col is not None:  # target is not wood (aluminium bar): ignore a wood block next to it
        above &= ~wood_mask(col)
    if exclude_bright and col is not None:  # target is the wood block next to an aluminium bar: wood pixels only
        above &= wood_mask(col)
    jaw = (v > 120) & ((u < JAW["left_tip_px"][0] + 25) | (u > JAW["right_tip_px"][0] - 25))
    num, lab, st, cen = cv2.connectedComponentsWithStats((above & ~jaw).astype(np.uint8), 8)
    if num <= 1:
        return None, None
    k = 1 + int(np.argmax(st[1:, 4]))
    if st[k, 4] < 800:
        return None, None
    m = lab == k
    top = m & (h > np.percentile(h[m], 60))  # upper part = top face
    x, y, bw, bh = st[k, :4]
    cut = bool(x <= 1 or y <= 1 or x + bw >= W - 1 or y + bh >= H - 1)  # block runs off the image
    # nearest-to-jaws corner: lowest image rows of the block = a fixed physical point even when the block is cut off
    vv = v[m]
    low = m & (v >= np.percentile(vv, 97))
    corner = P[low].mean(0)
    return P[top].mean(0), {"px": int(st[k, 4]), "bbox": st[k, :4].tolist(), "height_m": float(np.median(h[m])),
                            "cut": cut, "corner": corner.round(4).tolist(),
                            "floor_n": n.round(5).tolist(), "floor_d": round(float(dd), 5)}


def jaw_frame_offset(d, w_t, A_, C_, T_, col=None, exclude_wood=False, exclude_bright=False):
    """Offset of the block from the jaw-tip midpoint in the jaw frame (m). The far side of the block is usually
    cut off by the image top, so the thickness centre comes from the visible near edge minus half the block
    width along Yg (from the chest capture), and the closing centre from the extent when the sides are in view."""
    b, info = block_in_wrist(d, col, exclude_wood, exclude_bright)
    if b is None:
        return None
    I = JAW["intrinsics"]; mid = np.array(JAW["mid_tips_cam_m"])
    H, W = d.shape; v, u = np.mgrid[0:H, 0:W]
    P = np.dstack([(u - I["ppx"]) / I["fx"] * d, (v - I["ppy"]) / I["fy"] * d, d])
    n = np.array(info["floor_n"]); dd = info["floor_d"]; h = P @ n + dd
    x, y, bw, bh = info["bbox"]
    m = (u >= x) & (u < x + bw) & (v >= y) & (v < y + bh) & (h > 0.015) & (d > 0.04)
    jaw = (v > 120) & ((u < JAW["left_tip_px"][0] + 25) | (u > JAW["right_tip_px"][0] - 25))
    m &= ~jaw
    if exclude_wood and col is not None:
        m &= ~wood_mask(col)
    if exclude_bright and col is not None:
        m &= wood_mask(col)
    if m.sum() < 300:
        return None
    top = m & (h > np.percentile(h[m], 60)); Q = P[top]
    pc, pt, pa = Q @ C_, Q @ T_, Q @ A_
    cut_top = y <= 1
    sides_in = x > 1 and x + bw < W - 1
    e_c = ((np.percentile(pc, 2) + np.percentile(pc, 98)) / 2 if sides_in else pc.mean()) - mid @ C_
    e_t = ((np.percentile(pt, 98) - w_t / 2) if cut_top else (np.percentile(pt, 2) + np.percentile(pt, 98)) / 2) - mid @ T_
    e_a = float(np.median(pa) - mid @ A_)
    tips = [np.array(JAW["left_tip_cam_m"]), np.array(JAW["right_tip_cam_m"])]
    tip_floor = min(float(p @ n + dd) for p in tips)
    return {"e_c": float(e_c), "e_t": float(e_t), "e_a": e_a, "cut_top": bool(cut_top),
            "tip_floor_m": tip_floor, "drop_per_m": float(-(A_ @ n))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--start", default=None)
    ap.add_argument("--no-grasp", action="store_true", help="align and approach, take a wrist frame, then go back (no close)")
    a = ap.parse_args()
    sys.path.insert(0, REPO)
    import pinocchio as pin
    from ik.openarm_left_ik import OpenArmLeftIK
    plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
    ik = OpenArmLeftIK(URDF)
    Tg = np.array(plan["T_grasp_base"]); Xg, Yg, Zg = Tg[:3, 0], Tg[:3, 1], Tg[:3, 2]

    def tcp_z(qd):
        M = ik.fk(np.deg2rad(qd)); return float((M.translation + M.rotation @ [0, 0, 0.2145])[2])

    def check(q_from, q_to, name, allow_low=False):
        tmin = min(FLOOR_Z + 0.02, tcp_z(q_from) - 0.01, tcp_z(q_to) - 0.01)
        for s in np.linspace(0, 1, 20):
            q = np.deg2rad(np.asarray(q_from) + (np.asarray(q_to) - np.asarray(q_from)) * s)
            M = ik.fk(q); tcp = M.translation + M.rotation @ [0, 0, 0.2145]
            if M.translation[2] < FLOOR_Z + 0.05 or (not allow_low and tcp[2] < tmin):
                raise SystemExit(f"[servo] {name}: s={s:.2f} link7 z={M.translation[2]:.3f} TCP z={tcp[2]:.3f} too low. Stopping")

    def shifted(qd, delta, seed=None):
        """Same EE orientation, translated by delta (base frame). Returns lerobot deg, or raises."""
        M = ik.fk(np.deg2rad(qd)); T = pin.SE3(M.rotation, M.translation + np.asarray(delta))
        q0 = np.deg2rad(seed if seed is not None else qd)
        rng = np.random.default_rng(0)
        best = None
        for sd in [q0] + [q0 + rng.normal(0, 0.3, 7) for _ in range(8)]:  # several seeds, keep the closest solution
            q, ok, _, _ = ik.solve(T, np.clip(sd, ik.lo, ik.hi), position_only=False, max_iter=300)
            if ok and ik.within_limits(q):
                dmax = float(np.max(np.abs(np.rad2deg(q) - np.asarray(qd))))
                if best is None or dmax < best[0]:
                    best = (dmax, list(np.rad2deg(q)))
        if best is None:
            raise SystemExit(f"[servo] IK failed for a shift of {np.round(delta, 3).tolist()} m. Stopping")
        if best[0] > 30:
            raise SystemExit("[servo] shifted pose jumped IK branch (>30 deg). Stopping")
        return best[1]

    def raised(qd, dz=0.08):
        return shifted(qd, [0, 0, dz])

    def lift_from(qd):
        """Lift straight up as high as reachable (iter_024: 10 cm was out of reach next to the joint limits)."""
        for dz in (0.10, 0.08, 0.06, 0.05):
            try:
                return shifted(qd, [0, 0, dz], seed=qd)
            except SystemExit:
                continue
        raise SystemExit("[servo] cannot lift >= 5 cm from the grasp pose. Stopping")

    pre, grasp, lift = plan["q_pre_deg"], plan["q_grasp_deg"], plan["q_lift_deg"]
    if not a.go:
        start = [float(v) for v in a.start.split(",")] if a.start else plan["q_start_deg"]
        up = raised(start)
        check(start, up, "start->raise"); check(up, pre, "raise->pre"); check(pre, grasp, "pre->grasp", True)
        check(grasp, lift, "grasp->lift", True)
        near = shifted(grasp, -NEAR_BACK * Zg, seed=grasp); check(pre, near, "pre->near", True)
        print("[servo] near pose " + " ".join(f"{v:+.1f}" for v in near))
        try:
            deep = shifted(near, (NEAR_BACK + 0.03) * Zg, seed=grasp); lift_from(deep)
            print("[servo] lift ok from 3 cm deeper than the model grasp")
        except SystemExit as e:
            print(f"[servo] note: no lift from 3 cm deeper than the model grasp ({e})")
        jq = shifted(near, 0.01 * Xg); check(near, jq, "jog Xg", True)
        for s_ in (MAX_CORR_M * Xg, -MAX_CORR_M * Xg, MAX_CORR_M * Yg, -MAX_CORR_M * Yg):
            try:
                shifted(near, s_)
            except SystemExit as e:
                print(f"[servo] note: lateral correction {np.round(s_, 3).tolist()} not solvable at the near pose ({e})")
        off = np.array(list(plan.get("place_offset_m", [0.0, 0.0])) + [0.0])
        if np.linalg.norm(off) > 0:
            lift_d = lift_from(grasp)
            carry = shifted(lift_d, off, seed=lift_d); place = shifted(grasp, off + np.array([0, 0, 0.01]), seed=grasp)
            check(lift_d, carry, "carry", True); check(carry, place, "place", True)
            print("[servo] place aside ok: " + " ".join(f"{v:+.1f}" for v in place))
        print("[servo] DRY RUN ok (add --go to move)"); return 0

    import cv2
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    grav = OpenArmGravityModel(URDF, "left", 1.0).torque
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from grasp.soft_release import release_and_disconnect  # 2026-11-22 Sota: 力を抜くときはゆっくり（ダンプのように）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_l"), side="left", id=os.environ.get("GRASP_ROBOT_ID", "follower_left"),
                                                  align_on_connect=False, cameras={}))
    from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell onto the cart
    connect_holding(robot, grav)
    obs = robot.get_observation(); start = [float(obs[f"{j}.pos"]) for j in J]
    log = open(os.path.join(a.cap_dir, "traj.csv"), "w", newline=""); w = csv.writer(log)
    w.writerow(["t", "phase"] + [f"{j}.pos" for j in J] + ["gripper.pos"])
    servo_log = []

    def rec(phase):
        o = robot.get_observation()
        w.writerow([f"{time.time():.3f}", phase] + [f"{o[f'{j}.pos']:.2f}" for j in J] + [f"{o['gripper.pos']:.2f}"])

    def go(q, speed, g, phase, correct=True):
        goal = {j: float(q[i]) for i, j in enumerate(J)}; goal["gripper"] = float(g)
        soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=speed)
        cmd = dict(goal)
        for _ in range(3 if correct else 0):
            o = robot.get_observation()
            err = {j: goal[j] - float(o[f"{j}.pos"]) for j in J}
            if max(abs(v) for v in err.values()) < 1.0:
                break
            for j in J:
                cmd[j] = goal[j] + float(np.clip(cmd[j] - goal[j] + err[j], -15, 15))
            soft_move_to_position(robot.bus, cmd, duration_s=1.0, torque_ff_fn=grav)
        rec(phase)
        o = robot.get_observation(); e = max(abs(goal[j] - float(o[f"{j}.pos"])) for j in J)
        print(f"[servo] {phase} done (max joint error {e:.1f} deg)", flush=True)

    def look(tag):
        """Returns (centroid, corner, cut) of the block in the wrist camera frame, or (None, None, None)."""
        time.sleep(0.6)
        col, d = cam.grab()
        cv2.imwrite(os.path.join(a.cap_dir, f"wrist_{tag}.jpg"), col)
        np.save(os.path.join(a.cap_dir, f"wrist_{tag}_depth.npy"), d.astype(np.float16))
        b, info = block_in_wrist(d, col, bool(plan.get("exclude_wood")), bool(plan.get("exclude_bright")))
        servo_log.append({"tag": tag, "block_cam": None if b is None else b.round(4).tolist(), "info": info})
        print(f"[servo] look {tag}: block {None if b is None else np.round(b, 3).tolist()} {info}", flush=True)
        if b is None:
            return None, None, None
        return b, np.array(info["corner"]), info["cut"]

    def measure(tag, w_t, A_, C_, T_):
        """Block offset from the jaw centre in the jaw frame: e_c (closing), e_t (thickness), e_a (approach)."""
        time.sleep(0.6)
        col, d = cam.grab()
        cv2.imwrite(os.path.join(a.cap_dir, f"wrist_{tag}.jpg"), col)
        np.save(os.path.join(a.cap_dir, f"wrist_{tag}_depth.npy"), d.astype(np.float16))
        r = (bright_offset(d, col, A_, C_) if ALU else
             jaw_frame_offset(d, w_t, A_, C_, T_, col, False, bool(plan.get("exclude_bright"))))
        servo_log.append({"tag": tag, **({} if r is None else {k: (round(float(v), 4) if not isinstance(v, bool) else v) for k, v in r.items()})})
        print(f"[servo] measure {tag}: " + ("block not found" if r is None else
              f"closing {r['e_c']*1000:+.0f} thickness {r['e_t']*1000:+.0f} along {r['e_a']*1000:+.0f} mm cut_top={r['cut_top']}"), flush=True)
        return r

    ALU = bool(plan.get("exclude_wood"))  # target = aluminium bar (bright): measure by colour, depth from the floor
    g_now = float(obs["gripper.pos"])
    cam = None
    try:
        cam = Wrist()
        up = raised(start)
        check(start, up, "start->raise"); check(up, pre, "raise->pre")
        go(up, 10, g_now, "raise")
        go(up, 15, G_OPEN, "open gripper")
        # --- v5: measure and correct 4 cm before the grasp (kinematically close to it: the model error moved
        # 36 mm between the pregrasp and the grasp in iter_016), in the jaw frame from the jaw geometry
        A_, C_, T_ = (np.array(JAW[k]) for k in ("approach_cam", "closing_cam", "thickness_cam"))
        w_t = float(plan.get("block_extent_Yg_m", 0.065))
        go(pre, 15, G_OPEN, "pregrasp")
        near = shifted(grasp, -NEAR_BACK * Zg, seed=grasp); check(pre, near, "pre->near", True)
        go(near, 6, G_OPEN, "near")
        m0 = measure("near0", w_t, A_, C_, T_)
        if m0 is None:
            raise KeyboardInterrupt("block not seen by the wrist camera near the grasp")
        # sanity jog: +1 cm along Xg must move the block about -1 cm along the closing axis
        jq = shifted(near, 0.01 * Xg); check(near, jq, "jog Xg", True)
        go(jq, 4, G_OPEN, "jog Xg")
        mj = measure("jog_Xg", w_t, A_, C_, T_)
        go(near, 4, G_OPEN, "jog back")
        dj = None if mj is None else mj["e_c"] - m0["e_c"]
        if dj is None or dj > 0.004:  # wrong sign = camera/arm mapping flipped -> corrections would diverge
            raise KeyboardInterrupt(f"jog check failed: closing offset change {None if dj is None else round(dj*1000)} mm for +10 mm (expect about -10)")
        if dj > -0.004:  # too small to confirm (iter_023: +1 mm); mapping was confirmed on iter_017/018, continue
            print(f"[servo] warning: jog response small ({dj*1000:+.0f} mm for +10 mm); continuing with the confirmed mapping", flush=True)
        # closing-axis response per metre from the jog (iter_024: 1.7x -> damped steps still overshot): divide it out
        kx = float(np.clip(-dj / 0.01, 0.5, 2.5)) if dj <= -0.004 else 1.0
        print(f"[servo] closing response {kx:.2f} (1.0 = as modelled)", flush=True)
        total = np.zeros(3); m = m0; cur = near
        for it in range(3):
            servo_log.append({"iter": it, **{k: round(float(v), 4) for k, v in m.items() if k.startswith("e_")}})
            print(f"[servo] iter {it}: closing {m['e_c']*1000:+.0f} mm, thickness {m['e_t']*1000:+.0f} mm, "
                  f"along {m['e_a']*1000:+.0f} mm (expect {EXPECT_ALONG_NEAR_M*1000:+.0f})", flush=True)
            # thickness axis = along the object's length for long objects (aluminium bar ~20 cm): anywhere at least
            # 4 cm inside its ends is fine, so only correct it when the jaws would come within 4 cm of an end
            tol_t = max(TOL_M, w_t / 2 - 0.04)
            # thickness only from a full view: the cut-off estimate (near edge - half width) disagreed by 4-7 cm (iter_022)
            e_t_corr = 0.0 if (m["cut_top"] or abs(m["e_t"]) < tol_t) else m["e_t"] - np.sign(m["e_t"]) * (tol_t - TOL_M)
            if abs(m["e_c"]) < TOL_M and e_t_corr == 0.0:
                break
            step = GAIN * (m["e_c"] / kx * Xg + e_t_corr * Yg)
            if np.linalg.norm(total + step) > MAX_CORR_M:
                raise KeyboardInterrupt(f"correction would exceed {MAX_CORR_M} m")
            total = total + step
            nq = shifted(near, total, seed=cur); check(cur, nq, f"align {it}", True)
            go(nq, 4, G_OPEN, f"align {it}"); cur = nq
            m = measure(f"align{it}", w_t, A_, C_, T_)
            if m is None:
                raise KeyboardInterrupt("block lost while aligning")
        # depth: jaw tips kept >= FLOOR_MIN_M above the camera floor, and at least 1.5 cm below the object's top.
        # Search from the wanted depth upward in 5 mm steps for an IK-reachable one (iter_017: reach edge).
        tip_h, drop = m["tip_floor_m"], m["drop_per_m"]
        # the jaw corners (±19 mm across the thickness) sit lower than the tip centre when tilted: count them
        # (iter_027: 30 deg tilt, centre 8 mm up -> corners ~1 mm into the floor, 6.4 deg joint error on contact)
        corner_drop = 0.019 * float(np.sqrt(max(0.0, 1.0 - drop ** 2)))
        tip_h = tip_h - corner_drop
        tip_max = float(np.clip(plan.get("block_height_m", TIP_MAX_M + 0.015) - 0.015, 0.012, TIP_MAX_M))
        if ALU:  # no usable depth on the bar: go by the floor only, tips at ~1.2 cm
            want = (tip_h - min(tip_max, 0.012)) / max(drop, 1e-3)
        else:
            want = NEAR_BACK + float(np.clip(m["e_a"] - EXPECT_ALONG_NEAR_M, -0.03, 0.04))
        want = min(want, (tip_h - FLOOR_MIN_M) / max(drop, 1e-3))
        min_travel = (tip_h - tip_max) / max(drop, 1e-3)
        travel = want; ok_depth = False
        while travel >= min_travel - 1e-9:
            try:
                shifted(near, total + travel * Zg, seed=cur); ok_depth = True; break
            except SystemExit:
                travel -= 0.005
        if not ok_depth:
            raise KeyboardInterrupt(f"no reachable depth: tips would stay above {tip_max*100:.1f} cm (tips now {tip_h*100:.1f} cm, want travel {want*100:.1f} cm)")
        lift_from(shifted(near, total + travel * Zg, seed=cur))  # fail before descending, not after
        h_final = tip_h - travel * drop
        print(f"[servo] descend {travel*100:.1f} cm along the approach in 2 cm steps; tips end {h_final*1000:.0f} mm above the floor", flush=True)
        # stepwise descent with re-measurement: in iter_021 the jaws drifted 35 mm sideways over a single 6 cm approach
        steps = max(1, int(np.ceil(travel / 0.02)))
        for k in range(1, steps + 1):
            t_k = travel * k / steps
            qk = shifted(near, total + t_k * Zg, seed=cur); check(cur, qk, f"descend {k}", True)
            go(qk, 4, G_OPEN, f"descend {k}/{steps}"); cur = qk
            mk = measure(f"descend{k}", w_t, A_, C_, T_)
            if mk is None:
                continue
            tol_t = max(TOL_M, w_t / 2 - 0.04)
            e_t_corr = 0.0 if (ALU or mk["cut_top"] or abs(mk["e_t"]) < tol_t) else mk["e_t"] - np.sign(mk["e_t"]) * (tol_t - TOL_M)
            if abs(mk["e_c"]) < TOL_M and e_t_corr == 0.0:
                continue
            cap = 0.015 if k == steps else 0.03  # jaws already around the object at the last step: small moves only
            step = np.clip(GAIN * mk["e_c"] / kx, -cap, cap) * Xg + np.clip(GAIN * e_t_corr, -cap, cap) * Yg
            if np.linalg.norm(total + step) > MAX_CORR_M:
                raise KeyboardInterrupt(f"correction would exceed {MAX_CORR_M} m")
            total = total + step
            qc = shifted(near, total + t_k * Zg, seed=cur); check(cur, qc, f"descend {k} fix", True)
            go(qc, 4, G_OPEN, f"descend {k} fix"); cur = qc
        grasp_c = cur
        lift_c = lift_from(grasp_c)
        pre_c = shifted(pre, total, seed=pre)
        check(grasp_c, lift_c, "grasp->lift (corrected)", True)
        servo_log.append({"total_lateral_m": total.round(4).tolist(), "travel_m": round(travel, 4),
                          "tip_floor_final_m": round(h_final, 4)})
        print(f"[servo] lateral correction {np.round(total*1000).tolist()} mm, travel {travel*1000:.0f} mm", flush=True)
        measure("at_grasp", w_t, A_, C_, T_)  # evidence only
        if a.no_grasp:
            go(cur, 6, G_OPEN, "back to near"); go(pre_c, 10, G_OPEN, "back to pregrasp"); go(up, 12, G_OPEN, "back to raise")
            raise KeyboardInterrupt("no-grasp probe done")
        go(grasp_c, 15, G_CLOSE, "close gripper", correct=False); time.sleep(0.5)
        go(lift_c, 8, G_CLOSE, "lift"); time.sleep(2.0); rec("hold")
        look("hold")  # evidence only
        off = np.array(list(plan.get("place_offset_m", [0.0, 0.0])) + [0.0])
        if np.linalg.norm(off) > 0:  # carry it aside (clearing an obstacle) instead of putting it back
            carry = shifted(lift_c, off, seed=lift_c); check(lift_c, carry, "carry", True)
            place = shifted(grasp_c, off + np.array([0, 0, 0.01]), seed=grasp_c); check(carry, place, "place", True)
            go(carry, 8, G_CLOSE, "carry aside")
            go(place, 6, G_CLOSE, "put down aside")
            go(place, 15, G_OPEN, "release", correct=False)
            go(carry, 8, G_OPEN, "up from the placed object")
            go(lift_c, 8, G_OPEN, "back over the pick spot")
            raise KeyboardInterrupt("placed aside; done")
        go(grasp_c, 6, G_CLOSE, "put back down")
        go(grasp_c, 15, G_OPEN, "release", correct=False)
        go(cur, 6, G_OPEN, "back to near")
        go(pre_c, 10, G_OPEN, "back to pregrasp")
        go(up, 12, G_OPEN, "back to raise")
    except (KeyboardInterrupt, SystemExit) as e:  # SystemExit = IK/floor check refused a pose: retreat the same way
        print(f"[servo] stopping ({e}): returning to the start pose", flush=True)
        try:
            o = robot.get_observation(); cur = [float(o[f"{j}.pos"]) for j in J]
            up_now = raised(cur, 0.03)
            go(up_now, 8, G_OPEN, "lift clear")
            go(raised(start), 12, G_OPEN, "back to raise")
        except BaseException as e2:  # noqa: BLE001
            print(f"[servo] retreat via raise failed ({e2}); going straight to start", flush=True)
    finally:
        json.dump(servo_log, open(os.path.join(a.cap_dir, "servo.json"), "w"), indent=1)
        if cam is not None:
            cam.stop()
        try:
            go(start, 12, g_now if g_now > -10 else G_OPEN, "back to start")
        finally:
            release_and_disconnect(robot, grav); log.close()
            print("[servo] back at start, torque off", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
