#!/usr/bin/env python3
"""机に置いた ArUco マーカーの真上へ、指先（爪の間の中心）を持っていく確かめ（左右どちらも）。**--go を付けると腕が動く。**

2026-11-22・作業者: Munenori。右腕の J1・J2 のゼロ点を直した後に、胸のカメラ（ハンドアイ）・腕の運動学・指先の位置が
合っているかを、机の上の 1 点で目で確かめる。マーカーの中心と爪の間の中心の水平のずれを定規で測る。

段:
  1. 胸の D435i で、机に平らに置いたマーカー（DICT_4X4_50・--marker）の中心を撮る（深度は 15 枚の中央値）→ 腕の付け根の座標へ
  2. 指先を中心の --height（既定 5 cm）上に、爪を真下に向けた姿勢を IK で解く（手首のひねりは 30° おきに試し、休みの姿勢から近い解）
  3. 休みの姿勢 → 真上に 8 cm（torque_test.py と同じ）→ マーカーの上 → --hold 秒止める → 真上 8 cm → 休みの姿勢 → トルクを切る
     どの区間も 20 点で、手首と爪の先が机（GRASP_FLOOR_Z）より上か・関節が可動域の中かを確かめる（駄目なら動かさない）
止める: Ctrl-C（休みの姿勢に戻ってからトルクを切る）。危ないときは非常停止。

    set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
    ~/projects/lerobot/.venv/bin/python grasp/tools/point_test.py --arm right --marker 24          # 撮って計算だけ
    ~/projects/lerobot/.venv/bin/python grasp/tools/point_test.py --arm right --marker 24 --go     # 動かす
    # 手首の D405 で爪の向きを測る（マーカーの上 6 cm で爪を開き、±2 cm ずつ 6 回動かす・2026-11-23）
    ~/projects/lerobot/.venv/bin/python grasp/tools/point_test.py --arm left --marker 24 --height 0.06 --tilts 0 --jog 0.02        # 計算だけ
    ~/projects/lerobot/.venv/bin/python grasp/tools/point_test.py --arm left --marker 24 --height 0.06 --tilts 0 --jog 0.02 --go   # 動かす
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
TIP_LEN = 0.2295  # link7 から爪の先（torque_test.py と同じ）
REST_FIX_MAX_DEG = 30.0  # 始める前に休みの姿勢へ自動で合わせる、ずれの上限 [deg]
AX_L7_G = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)  # 列: 爪の座標の Xg・Yg・Zg を link7 の座標で（plan.py の T_L7_G と同じ）


def run_jogs(a, side, jogs, tgt, qh, go, ik, tcp_l7, g_open) -> None:
    """マーカーの上で爪を少しずつ動かし、手首の D405 でマーカーを測って、カメラの座標での爪の向きを当てはめて書く。"""
    from grasp.wrist import WristMarker, fit_axes
    import cv2
    cam = WristMarker(a.wrist_serial)

    def tcp_g(qd):  # 読んだ角度での指先の位置（腕の付け根の座標）
        M = ik.fk(np.deg2rad(qd))
        return M.translation + M.rotation @ tcp_l7

    try:
        R0 = ik.fk(np.deg2rad(qh)).rotation @ AX_L7_G  # 爪の座標 → 腕の付け根の座標（基準の姿勢で）
        p0 = cam.measure(a.marker)
        if p0 is None:
            sys.exit("[point] 手首のカメラにマーカーが見えない（爪の間から見える所に置く）")
        out_dir = os.path.join(REPO, "grasp", "runs", "wristaxes_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(out_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_dir, "base.jpg"), cam.last_color)
        print(f"[point] 手首のカメラのマーカー（基準）: {np.round(p0 * 1000, 1).tolist()} mm", flush=True)
        rows, D, P = [], [], []
        for k, sg, qj in jogs:
            name = f"{'XYZ'[k]}g {sg * a.jog * 100:+.0f} cm"
            qa = go(qj, g_open, f"jog {name}", speed=5.0)
            time.sleep(1.0)
            pj = cam.measure(a.marker)
            qb = go(tgt, g_open, "jog 戻り", speed=5.0)
            time.sleep(1.0)
            pb = cam.measure(a.marker)
            if pj is None or pb is None:
                print(f"[point] {name}: マーカーが見えなかった。飛ばす", flush=True)
                continue
            d_g = R0.T @ (tcp_g(qa) - tcp_g(qb))  # 読んだ角度で動いた量（爪の座標）
            sh = pj - (pb + p0) / 2
            p0 = pb
            D.append(d_g); P.append(sh)
            rows.append(dict(jog=name, delta_g_m=d_g.tolist(), shift_cam_m=sh.tolist()))
            print(f"[point] {name}: 動いた（FK）{np.round(d_g * 1000, 1).tolist()} mm → マーカー {np.round(sh * 1000, 1).tolist()} mm", flush=True)
        if len(D) < 4:
            sys.exit("[point] 測れた jog が 4 個より少ない。当てはめない")
        R, ratio, resid = fit_axes(np.array(D), np.array(P))
        res = dict(_note=f"手首の D405 から見た爪の向き（grasp/tools/point_test.py --jog・{time.strftime('%Y-%m-%d %H:%M')}・作業者 Munenori）。"
                         "列が Xg（閉じる）・Yg（厚み）・Zg（近づく）のカメラの座標の向き。記録: " + os.path.relpath(out_dir, REPO),
                   arm=side, serial=a.wrist_serial, jog_m=a.jog, q_hold_deg=list(map(float, qh)),
                   R_cam_g=R.tolist(), closing_cam=R[:, 0].tolist(), thickness_cam=R[:, 1].tolist(), approach_cam=R[:, 2].tolist(),
                   ratio=ratio.tolist(), resid_m=resid.tolist(), jogs=rows)
        out = a.jog_out or os.path.join(REPO, "grasp", "config", "sites",
                                        f"{os.environ.get('GRASP_SITE', 'site')}.wrist_axes_{side}.json")
        json.dump(res, open(out, "w"), indent=2, ensure_ascii=False)
        json.dump(res, open(os.path.join(out_dir, "axes.json"), "w"), indent=2, ensure_ascii=False)
        print(f"[point] 爪の向き（カメラの座標）: 閉じる {np.round(R[:, 0], 3).tolist()}  厚み {np.round(R[:, 1], 3).tolist()}  "
              f"近づく {np.round(R[:, 2], 3).tolist()}", flush=True)
        print(f"[point] 動きの比 {np.round(ratio, 3).tolist()}（1 に近いほど良い）・当てはめの残り 最大 {resid.max() * 1000:.1f} mm → {out}", flush=True)
    finally:
        cam.close()


def capture_marker(serial: str, marker: int, T_bc: np.ndarray, n: int = 15) -> np.ndarray:
    """マーカーの中心（腕の付け根の座標 [m]）。見えなければ止める。"""
    import cv2
    import pyrealsense2 as rs
    from grasp.wrist import start_pipeline
    cfg = rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    pipe, prof = start_pipeline(cfg, serial)  # フレームが来なければ hardware_reset して 1 回やり直す
    try:
        scale = prof.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        for _ in range(30):
            pipe.wait_for_frames()
        det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        pts = []
        for _ in range(n):
            fs = align.process(pipe.wait_for_frames())
            col = np.asanyarray(fs.get_color_frame().get_data())
            dep = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * scale
            i = fs.get_color_frame().profile.as_video_stream_profile().intrinsics
            corners, ids, _ = det.detectMarkers(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
            if ids is None or marker not in ids.ravel():
                continue
            c = corners[list(ids.ravel()).index(marker)].reshape(4, 2)
            u, v = c.mean(0)
            x0, x1, y0, y1 = int(c[:, 0].min()), int(c[:, 0].max()) + 1, int(c[:, 1].min()), int(c[:, 1].max()) + 1
            zs = dep[y0:y1, x0:x1]
            zs = zs[zs > 0]
            if len(zs) < 10:
                continue
            z = float(np.median(zs))
            pts.append([(u - i.ppx) * z / i.fx, (v - i.ppy) * z / i.fy, z])
            cv2.imwrite("/tmp/point_test_marker.jpg", cv2.aruco.drawDetectedMarkers(col.copy(), corners, ids))
    finally:
        pipe.stop()
    if len(pts) < n // 2:
        sys.exit(f"[point] マーカー {marker} が見えない（{len(pts)}/{n} 枚）。机に平らに置き、腕や手で隠さない")
    p = np.median(np.array(pts), axis=0)
    return T_bc[:3, :3] @ p + T_bc[:3, 3]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--marker", type=int, default=24)
    ap.add_argument("--height", type=float, default=0.05, help="マーカーの中心から指先（爪の間の中心）までの高さ [m]")
    ap.add_argument("--hold", type=float, default=15.0, help="マーカーの上で止める秒数（定規で測る間）")
    ap.add_argument("--dps", type=float, default=10.0)
    ap.add_argument("--tilts", type=float, nargs="+", default=[0, 20, 40], help="試す爪の傾き（真下から）[deg]")
    ap.add_argument("--jog", type=float, default=0.0,
                    help="マーカーの上で爪を開き、爪の向き（閉じる・厚み・近づく）に ±この量 [m] 動かして手首の D405 でマーカーを測り、"
                         "カメラの座標での爪の向きを出す（grasp/wrist.py・2026-11-23）。0 なら測らない（--hold だけ）")
    ap.add_argument("--wrist-serial", default=os.environ.get("GRASP_WRIST_SERIAL"))
    ap.add_argument("--jog-out", default=None, help="--jog の結果の JSON（既定 grasp/config/sites/<置き場>.wrist_axes_<腕>.json）")
    ap.add_argument("--go", action="store_true")
    a = ap.parse_args()
    side = a.arm
    suf = "_RIGHT" if side == "right" else ""
    need = {"GRASP_CAN_PORT" + suf: None, "GRASP_ROBOT_ID" + suf: None, "GRASP_REST" + suf: None, "GRASP_FLOOR_Z": None,
            "GRASP_T_BASE_CAM": None, "GRASP_CHEST_SERIAL": None, "GRASP_GRIPPER_JSON": None}
    missing = [k for k in need if k not in os.environ]
    if missing:
        sys.exit(f"[point] 置き場の env に無い: {', '.join(missing)}（set -a; source grasp/config/sites/<置き場>.env; set +a）")
    rest = np.array([float(v) for v in os.environ["GRASP_REST" + suf].split(",")])
    fz = float(os.environ["GRASP_FLOOR_Z"])
    T_bc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
    tcp_l7 = np.array(json.load(open(os.environ["GRASP_GRIPPER_JSON"]))["tcp_offset_from_link7_m"])
    if side == "right":
        tcp_l7 = tcp_l7 * [1, -1, 1]  # 右の link7 は自分の y が反転（grasp/arm.py）

    import pinocchio as pin
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(URDF, side=side)  # 先端は hand_tcp（link7 と同じ所・同じ向き）

    def heights(qd):
        M = ik.fk(np.deg2rad(qd))
        return float(M.translation[2]), float((M.translation + M.rotation[:, 2] * TIP_LEN)[2]), M

    def path_bad(q0, q1):
        """関節の直線の途中で、手首と爪の先が端の低い方 − 1 cm より下がるか・爪の先が机 + 1 cm より下がるか・可動域の外なら、その説明。"""
        l7a, tipa, _ = heights(q0)
        l7b, tipb, _ = heights(q1)
        l7min, tipmin = min(l7a, l7b) - 0.01, min(tipa, tipb) - 0.01
        for s in np.linspace(0, 1, 20):
            l7, tip, _ = heights(q0 + (q1 - q0) * s)
            if l7 < l7min or tip < tipmin or tip < fz + 0.01 or not ik.within_limits(np.deg2rad(q0 + (q1 - q0) * s)):
                return f"s={s:.2f} で手首 {l7 - fz:+.3f}・爪の先 {tip - fz:+.3f} m（机から）か可動域の外"
        return None

    def path_ok(q0, q1, name):
        bad = path_bad(q0, q1)
        if bad:
            sys.exit(f"[point] {name} の途中（{bad}）。動かさない")

    def raised(qd, dz=0.08):
        _, _, M = heights(qd)
        q, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + [0, 0, dz]), np.deg2rad(qd), position_only=False, max_iter=300)
        if not ok or not ik.within_limits(q):
            sys.exit(f"[point] 休みの姿勢から {dz * 100:.0f} cm 上げる姿勢が解けない。動かさない")
        return np.rad2deg(q)

    p = capture_marker(os.environ["GRASP_CHEST_SERIAL"], a.marker, T_bc)
    print(f"[point] マーカー {a.marker} の中心（付け根の座標）: x {p[0]:+.3f}  y {p[1]:+.3f}  z {p[2]:+.3f} m（机 {p[2] - fz:+.3f} m）"
          "  画像: /tmp/point_test_marker.jpg", flush=True)
    if abs(p[2] - fz) > 0.03:
        sys.exit("[point] マーカーが机の面から 3 cm 以上離れている。平らに置いたか・ハンドアイが合っているかを見る。動かさない")
    up = raised(rest)
    goal = p + [0, 0, a.height]
    rng = np.random.default_rng(0)  # 上げた姿勢・休みの姿勢からは解けないことがあるので、乱数の初期値も試す
    seeds = [up, rest] + [np.rad2deg(rng.uniform(ik.lo, ik.hi)) for _ in range(12)]
    cands = []  # (上げた姿勢からの最大の関節の差, 角度, 傾き)
    for tilt in a.tilts:  # 爪を真下から傾けてもよい（爪の間の中心の位置を見る確かめなので、向きは問わない）
        for az in np.radians(np.arange(0, 360, 45)):
            z = np.array([np.sin(np.radians(tilt)) * np.cos(az), np.sin(np.radians(tilt)) * np.sin(az), -np.cos(np.radians(tilt))])
            x0 = np.cross(z, [0, 0, 1.0]) if tilt > 0 else np.array([np.cos(az), np.sin(az), 0.0])
            x0 /= np.linalg.norm(x0)
            for roll in (np.radians(np.arange(0, 180, 45)) if tilt > 0 else [0.0]):
                x = np.cos(roll) * x0 + np.sin(roll) * np.cross(z, x0)
                R = np.column_stack([x, np.cross(z, x), z])
                T = pin.SE3(R, goal - R @ tcp_l7)
                for seed in seeds:
                    q, ok, _, _ = ik.solve(T, np.deg2rad(seed), position_only=False, max_iter=300)
                    if ok and ik.within_limits(q):
                        qd = np.rad2deg(q)
                        if not any(np.max(np.abs(qd - c[1])) < 5 for c in cands):
                            cands.append((float(np.max(np.abs(qd - up))), qd, tilt))
    if not cands:
        sys.exit("[point] マーカーの上の姿勢が解けない（届く範囲の外）。マーカーを腕の付け根に近づける")
    cands.sort(key=lambda c: c[0])
    path_ok(rest, up, "上げる")
    use_curobo = bool(os.environ.get("GRASP_CUROBO" + suf))
    route = None  # 上げた姿勢の後に通る角度（終点がマーカーの上）
    if use_curobo:  # 本体・カメラ・机・もう片方の腕との当たりを cuRobo で見る（大きく回る経路を床の高さだけで決めない）
        from grasp.curobo import client as curobo
        curobo.set_arm(side)
        bad = curobo.violations([rest, up], ["上げる"], table_margin=None)
        if bad:
            sys.exit(f"[point] 上げる区間が当たる: {curobo.describe(bad)}。動かさない")
        for cost, qd, tilt in cands[:12]:
            if cost > 150.0:
                break
            if not curobo.violations([up, qd], ["マーカーの上へ"]) and not path_bad(up, qd):
                route, tgt, tsel = [qd], qd, tilt
                break
            l7g, tipg, _ = heights(qd)
            vias = curobo.detour(up, qd, floor={"tcp_min_z": min(tipg, heights(up)[1]) - 0.005, "l7_min_z": min(l7g, heights(up)[0]) - 0.01,
                                               "tcp_len": TIP_LEN})
            if vias:
                r = [np.array(v) for v in vias]
                if not any(path_bad(a_, b_) for a_, b_ in zip([up] + r[:-1], r)):
                    route, tgt, tsel = r, qd, tilt
                    break
        if route is None:
            sys.exit("[point] マーカーの上へ当たらずに行く経路が作れない（cuRobo）。マーカーを近づける。動かさない")
    else:
        cost, tgt, tsel = cands[0]
        if cost > 60.0:
            sys.exit(f"[point] マーカーの上の解が上げた姿勢から {cost:.0f}° 離れている（60° まで）。cuRobo（GRASP_CUROBO{suf}）"
                     "を立てて当たりを見るか、マーカーを近づける。動かさない")
        route = [tgt]
    prev = up
    for i, q in enumerate(route):
        path_ok(prev, q, f"マーカーの上へ（{i + 1}/{len(route)}）")
        prev = q
    print(f"[point] 爪の傾き {tsel:.0f}°・経由 {len(route) - 1} 点・当たりの確かめ {'cuRobo' if use_curobo else '床の高さだけ'}", flush=True)
    M = ik.fk(np.deg2rad(tgt))
    tcp = M.translation + M.rotation @ tcp_l7
    print(f"[point] {side} 腕・{a.dps:.0f} deg/s・指先の目標 x {goal[0]:+.3f} y {goal[1]:+.3f} z {goal[2]:+.3f}（FK で {np.linalg.norm(tcp - goal) * 1000:.1f} mm）")
    for name, q in [("休み", rest), ("上げる", up)] + [(f"経由 {i + 1}", v) for i, v in enumerate(route[:-1])] + [("マーカーの上", tgt)]:
        l7, tip, _ = heights(q)
        print(f"[point] {name:6s} " + " ".join(f"{v:+6.1f}" for v in q) + f"   手首 {l7 - fz:+.3f}・爪の先 {tip - fz:+.3f} m（机から）")
    jogs = []  # (爪の座標の向きの番号, 符号, 角度)
    if a.jog > 0:
        if not a.wrist_serial:
            sys.exit("[point] --jog には手首の D405 のシリアル（GRASP_WRIST_SERIAL か --wrist-serial）が要る")
        Mt = ik.fk(np.deg2rad(tgt))
        for k in range(3):
            ax = Mt.rotation @ AX_L7_G[:, k]  # 爪の座標の向き（腕の付け根の座標）
            for sg in (1, -1):
                q, ok, _, _ = ik.solve(pin.SE3(Mt.rotation, Mt.translation + sg * a.jog * ax), np.deg2rad(tgt),
                                       position_only=False, max_iter=300)
                qd = np.rad2deg(q)
                if not ok or not ik.within_limits(q) or np.max(np.abs(qd - tgt)) > 15.0:
                    sys.exit(f"[point] jog {'XYZ'[k]}g {sg * a.jog * 100:+.0f} cm が解けない（か関節が 15° より動く）。動かさない")
                path_ok(tgt, qd, f"jog {'XYZ'[k]}g {sg:+d}")
                jogs.append((k, sg, qd))
        print(f"[point] jog: 6 個とも解けた（関節の差 最大 {max(np.max(np.abs(j[2] - tgt)) for j in jogs):.1f}°）", flush=True)
    if not a.go:
        print("[point] DRY RUN（--go で動かす）")
        return 0

    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # grasp/gravity.py を読む
    from grasp.gravity import gravity_torque_fn
    grav = gravity_torque_fn(side)  # 手と爪の質量入り（2026-11-23）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ["GRASP_CAN_PORT" + suf], side=side,
                                                  id=os.environ["GRASP_ROBOT_ID" + suf], align_on_connect=False, cameras={}))
    robot.connect()

    def now():
        o = robot.get_observation()
        return np.array([float(o[f"{j}.pos"]) for j in J]), float(o["gripper.pos"])

    def go(q, g, phase, speed=a.dps):
        goal_d = {j: float(q[i]) for i, j in enumerate(J)}
        goal_d["gripper"] = float(g)
        soft_move_to_position(robot.bus, goal_d, torque_ff_fn=grav, max_speed_dps=speed)
        age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
        if age > 0.3:
            raise SystemExit(f"[point] {phase}: モーターの返事が {age:.1f} 秒来ない（非常停止・電源）。指令をやめる")
        qn, _ = now()
        print(f"[point] {phase}: 関節の誤差 最大 {np.max(np.abs(qn - q)):.1f}°", flush=True)
        return qn

    q0, g0 = now()
    d0 = float(np.max(np.abs(q0 - rest)))
    if d0 > 5.0:
        # 2026-11-23（作業者 Munenori の依頼）: ずれていたら止めずに、休みの姿勢へゆっくり合わせてから始める（torque_test.py --to-rest と同じ考え）。
        # 関節の直線で戻すので、離れすぎ（REST_FIX_MAX_DEG より上）・床に近づく・cuRobo で当たるときは動かさない
        why = None
        if d0 > REST_FIX_MAX_DEG:
            why = f"{REST_FIX_MAX_DEG:.0f}° より離れている（関節の直線で戻すと大きく回る）。手で近づけてから打つ"
        else:
            why = path_bad(q0, rest)
            if why is None and use_curobo:
                bad = curobo.violations([q0, rest], ["休みの姿勢へ"], table_margin=None)
                why = f"cuRobo で当たる: {curobo.describe(bad)}" if bad else None
        if why:
            robot.disconnect()
            sys.exit(f"[point] 今の角度が休みの姿勢から {d0:.1f}° ずれていて、合わせられない（{why}）。動かしていない")
        print(f"[point] 今の角度が休みの姿勢から {d0:.1f}° ずれている。5 deg/s で休みの姿勢に合わせる", flush=True)
        q0 = go(rest, g0, "休みの姿勢に合わせた", speed=5.0)
    reached = []
    g_open = float(os.environ.get("GRASP_GRIPPER_OPEN" + suf, g0)) if jogs else g0  # 開いた爪の間からマーカーを見る
    try:
        go(up, g_open, "上げた")
        for i, q in enumerate(route[:-1]):
            reached.append(q)
            go(q, g_open, f"経由 {i + 1}")
        reached.append(tgt)
        qh = go(tgt, g_open, "マーカーの上")
        M = ik.fk(np.deg2rad(qh))
        tcp = M.translation + M.rotation @ tcp_l7
        d = tcp - goal
        print(f"[point] 読んだ角度での指先: 目標から x {d[0] * 1000:+.1f} y {d[1] * 1000:+.1f} z {d[2] * 1000:+.1f} mm（関節の追従の分）", flush=True)
        if jogs:
            try:  # 測れなくても、来た経路を逆にたどって戻す（休みの姿勢へ関節の直線で戻さない）
                run_jogs(a, side, jogs, tgt, qh, go, ik, tcp_l7, g_open)
            except (Exception, SystemExit) as e:  # noqa: BLE001
                print(f"[point] jog を止めた: {e}。マーカーの上に戻してから、来た経路で戻す", flush=True)
                go(tgt, g_open, "マーカーの上（戻す前）", speed=5.0)
        else:
            print(f"[point] {a.hold:.0f} 秒止める。マーカーの中心と爪の間の中心の水平のずれ（前後・左右）と、爪の間の中心の高さを測る", flush=True)
            time.sleep(a.hold)
        reached.pop()
        while reached:
            go(reached.pop(), g_open, "経由（戻り）")
        go(up, g_open, "上げた")
    except KeyboardInterrupt:
        print("[point] Ctrl-C: 来た経由点を逆にたどって上げ、休みの姿勢に戻す", flush=True)
        try:
            reached and reached.pop()  # 向かっていた点は飛ばし、その前の点から逆にたどる
            while reached:
                go(reached.pop(), g_open, "経由（戻り）")
            go(up, g_open, "上げた")
        except Exception as e:  # noqa: BLE001  戻すのを優先する
            print(f"[point] 上げられなかった: {e}", flush=True)
    finally:
        try:
            go(rest, g0, "休みの姿勢に戻した", speed=min(a.dps, 10.0))
        finally:
            robot.disconnect()
            print("[point] トルクを切った", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
