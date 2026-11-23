#!/usr/bin/env python3
"""Step 6: execute plan.json on the real arm (runs on Thor, lerobot venv). **The arm moves.** Dry run by default.

Sequence (all joint-space interpolation, with gravity feed-forward):
  start (the current resting pose) → [open gripper] → pregrasp (15 deg/s) → grasp (6 deg/s, slow)
  → [close gripper] → lift (8 deg/s) → hold 2 s → grasp (put it back down) → [open gripper] → pregrasp → start → torque off
Before moving, FK-checks every segment at 20 points to make sure the TCP and link7 stay above the floor (+2 cm) (aborts if not).
GRASP_CUROBO があれば cuRobo の腕の球と置き場の周りの物（本体・カメラ・机）でも確かめ、移動の区間（物の近く以外）が当たるときは
cuRobo で回り込む経由点を作って順に動かす（grasp/curobo/）。物の近くの区間は直線のままで、当たれば止まる。
GRASP_ARM=right drives the right follower. --push runs push.json (plan_push.py) instead; a plan with q_drop_deg (plan_drop.py)
drops beside the shoulder after a held check (--held-check camera|angle, --retries).
Writes the joint-angle trajectory to traj.csv in the run folder.

    ~/projects/lerobot/.venv/bin/python run_grasp.py <cap_dir>          # check only
    ~/projects/lerobot/.venv/bin/python run_grasp.py <cap_dir> --go     # run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

ARM = os.environ.get("GRASP_ARM", "left")  # 2026-11-21: "right" drives the right follower (can_fol_r, follower_right)
REPO = os.path.expanduser(os.environ.get("OPENARM_REPO", "~/projects/openarm-orboh"))  # PC dry checks: OPENARM_REPO=~/work/openarm-orboh
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
# 腕が入ってはいけない箱（plan.py の KEEPOUT と同じ・2026-11-21 北九州で胸のカメラの取り付け部に当たった）。無ければ確かめない
KEEPOUT = [np.array([float(v) for v in b.split(",")]) for b in os.environ.get("GRASP_KEEPOUT_BOXES", "").split(";") if b.strip()]
TCP_LEN, TIP_LEN = 0.2145, 0.2295  # link7 から指先の中心・爪の先（plan.py の CFG と同じ値。東京の爪の実測）
G_OPEN = float(os.environ.get("GRASP_GRIPPER_OPEN", "-60.0")); G_CLOSE = float(os.environ.get("GRASP_GRIPPER_CLOSE", "2.0"))   # gripper.pos (lerobot deg). fully open -66 / fully closed +6.1
FLOOR_Z = float(os.environ.get("GRASP_FLOOR_Z", "0.27"))                # floor height in the base frame (from the T_base_cam fit)
if os.environ.get("GRASP_SITE") and "GRASP_FLOOR_Z" not in os.environ:  # 東京の台車以外で床の高さを決め忘れたら動かさない
    sys.exit("[grasp] GRASP_FLOOR_Z が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の値のまま床の確認をしない。grasp/config/sites/ で決める")


class PushDone(Exception):
    """--push finished: skip the grasp phases and go home."""


class TrackingLost(RuntimeError):
    pass


class MotorLost(SystemExit):
    """モーターの返事が途切れた（非常停止・電源断）。これ以上は腕に指令を送らず、記録も先へ進めない。"""


class Blocked(SystemExit):
    """区間が床・keep-out の箱・置き場の周りの物（cuRobo）に当たる。移動の区間なら cuRobo で回り込める（route）。
    SystemExit の仲間なので、cuRobo が無いときは今までどおり止まる。"""


CUROBO = None  # GRASP_CUROBO（cuRobo のサーバー grasp/curobo/server.py）があれば、腕の球と置き場の箱でも区間を確かめる
if os.environ.get("GRASP_CUROBO"):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from grasp.curobo import client as CUROBO
    except ImportError as e:
        sys.exit(f"[grasp] GRASP_CUROBO があるのに grasp/curobo/client.py が読めない（{e}）。当たりを確かめずに動かさない")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--probe", action="store_true", help="don't grasp: stop at the grasp pose, take one chest frame, and go back (to check the position error)")
    ap.add_argument("--transit-dps", type=float, default=15.0, help="joint speed for free-space moves [deg/s] (OpenArm exception, Sota 11/20)")
    ap.add_argument("--grip-kp", type=float, default=10.0,
                    help="gripper stiffness in soft moves [Nm/rad]; lerobot's soft default is 10 (~1.6 Nm when the jaws stop 9 deg short "
                         "of the +2 deg target). Collection and GR00T rollouts use 12 (GRIPPER_KP): at 25 the DM4310 gripper hit its 100 C over-temperature latch "
                         "(2026-09-21). 2026-11-21: 30 lifted the bare aluminium bar once, but stay at the data value 12 unless told otherwise")
    ap.add_argument("--grip-close", type=float, default=None,
                    help="gripper close target [lerobot deg] (default G_CLOSE=+2). The GR00T aluminium data (Orboh/bi_openarm_ti_alframe_t2-mix) "
                         "commands ~0 while holding, with follower gripper kp 12 (collection/rollout GRIPPER_KP=12, set for motor heat)")
    ap.add_argument("--retries", type=int, default=1, help="after the lift, if the gripper closed on nothing (see --held-below), "
                    "open, go back down and grasp again this many times (same plan); 0 = no retry")
    ap.add_argument("--held-below", type=float, default=-4.0, help="gripper.pos (lerobot deg) after the lift at or below this = "
                    "something between the jaws (closed command 0..+2; empty closes to about -1 .. -3)")
    ap.add_argument("--held-check", choices=["camera", "angle"], default=None,
                    help="how to tell after the lift whether something is held. camera (2026-11-21 Sota): the wrist D405 sees "
                         "the object between the jaws (new near pixels vs. empty-jaw references taken at the raise pose); "
                         "falls back to the gripper angle (--held-below) when the camera fails. angle: gripper angle only")
    ap.add_argument("--held-near", type=float, default=0.16, help="camera check: wrist-camera depth nearer than this [m] = "
                    "something at the jaws (tips ~0.125 m, a held bar 0.09-0.16 m; after a 10 cm lift the floor and the pile top are >= ~0.22 m)")
    ap.add_argument("--held-px", type=int, default=1500, help="camera check: at least this many new near pixels = held")
    ap.add_argument("--max-track-err", type=float, default=20.0,
                    help="stop the sequence when a phase ends more than this far from its target [deg] (collision / lost motor replies)")
    ap.add_argument("--near-dps", type=float, default=6.0, help="joint speed near the object/floor: approach, close, put down, release [deg/s]")
    ap.add_argument("--blend", action="store_true",
                    help="move through consecutive free-space legs and detour points without stopping (one trajectory, "
                         "ramp only at both ends, corners rounded like lerobot align_through_poses). The rounded trajectory "
                         "is collision-checked before the arm moves; a chain that fails the check is moved leg by leg as before "
                         "(2026-11-23 Sota: 区間をつなげて止まらずに動く)")
    ap.add_argument("--push", action="store_true", help="run push.json (grasp/tools/plan_push.py) instead of plan.json: closed jaws push the bar sideways to open a gap, then home")
    ap.add_argument("--hold-home", action="store_true",
                    help="end every trial at --home and keep torque ON (disconnect without disabling torque); "
                         "release later with grasp/tools/release_arm.py (Sota 11/20)")
    ap.add_argument("--close-wait", type=float, default=0.5, help="seconds to keep squeezing at the grasp pose before the lift "
                    "(2026-11-22: thin low bars; give the jaws time to settle)")
    ap.add_argument("--wrist-snap", action="store_true", help="take one wrist-camera frame right before closing (evidence only)")
    ap.add_argument("--wrist-center", action="store_true", help="left arm: right before closing, measure where the bar is "
                    "between the open jaws with the wrist camera (grasp/wrist_center.py) and shift the hand along the closing "
                    "axis by that much (up 15 mm, across, down) before closing. Implies --wrist-snap (2026-11-22)")
    ap.add_argument("--close-press", type=float, default=0.0, help="close while commanding the hand this much deeper along "
                    "the approach [m] (presses the jaws into a soft surface so a low bar ends up between them; 2026-11-22)")
    ap.add_argument("--standby", default=None, help="standby pose YAML (2026-11-22 Sota): home -> standby (hands raised and "
                    "forward, clear of the cart) -> the task -> standby -> home, e.g. config/standby_left_cart-v1.yaml")
    ap.add_argument("--home", default=None, help="start/end pose YAML (joint_1..7, lerobot deg), e.g. config/home_left_cart-sym-v4.yaml: "
                    "go there first (via an 8 cm raise), use it as the start, come back to it at the end")
    ap.add_argument("--avoid", action="store_true",
                    help="運ぶ区間（lift→置き先の上）と戻る区間（置き先の上→raise）を、胸のカメラで割り込む物を見ながら動く"
                         "（grasp/reactive.py・止まって経路を作り直す）。GRASP_CUROBO と plan_place.py の置き先が要る")
    ap.add_argument("--avoid-dps", type=float, default=10.0, help="--avoid で見ながら動くときの関節の速さ [deg/s]")
    ap.add_argument("--start", default=None, help="start pose (lerobot deg, comma-separated). Default = read from the robot (only with --go)")
    a = ap.parse_args(argv)
    if a.held_check is None:  # 2026-11-22: the left hand has the wrist D405; the right arm has no wrist camera
        a.held_check = "camera" if os.environ.get("GRASP_ARM", "left") == "left" else "angle"
    global G_OPEN, G_CLOSE
    sys.path.insert(0, REPO)
    from ik.openarm_right_ik import arm_ik_class
    push = None
    if a.push:  # grasp/tools/plan_push.py: open a gap beside the bar (closed jaws, push sideways), no grasp
        push = json.load(open(os.path.join(a.cap_dir, "push.json")))
        plan = {"q_pre_deg": push["q_above_deg"], "q_grasp_deg": push["q_down_deg"], "q_lift_deg": push["q_rise_deg"],
                "q_start_deg": push["q_start_deg"]}
    else:
        plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
    ik = arm_ik_class()(URDF)
    # 2026-11-22 (Sota, right arm): the arm's real shape (link meshes, gripper, the held bar) against the clamp / chest
    # camera / body / the other arm, instead of 12 centre-line points in a box (grasp/arm_collision.py)
    ACOLL = None
    if os.environ.get("GRASP_ARM_COLLISION", "1") == "1":  # 11/22: both arms (the cart tray is only in arm_collision)
        from grasp import arm_collision
        ACOLL = arm_collision.ArmCollision(ik)
    HELD_L = None
    if os.path.exists(os.path.join(a.cap_dir, "object.npy")):
        _T = np.array(json.load(open(os.environ.get("GRASP_T_BASE_CAM", os.path.join(REPO, "grasp", "config", "T_base_cam.json"))))["T_base_cam"])
        _ob = np.load(os.path.join(a.cap_dir, "object.npy")) @ _T[:3, :3].T + _T[:3, 3]; _e = (_ob - _ob.mean(0))[:, :2]
        HELD_L = float(min(0.35, np.ptp(_e @ np.linalg.eigh(np.cov(_e.T))[1][:, -1])))
    HELD_LEGS = ("grasp→lift", "lift→lift high", "lift high→", "raise→drop carry", "drop carry→drop", "lift→carry", "carry→place", "now→lift")
    # transit legs (route): the TCP stays this high above the floor unless an end is lower (Sota 11/22: 左右に振るときに台車に
    # ぶつかっている → もうちょっと上にして移動)
    TRANSIT_CLEAR = float(os.environ.get("GRASP_TRANSIT_CLEAR", "0.20" if ARM == "right" else "0.02"))

    def tcp_z(qd):
        M = ik.fk(np.deg2rad(qd)); return float((M.translation + M.rotation @ [0, 0, 0.2145])[2])

    def l7_z(qd):
        return float(ik.fk(np.deg2rad(qd)).translation[2])

    def check(q_from, q_to, name, allow_low=False, transit=False, quiet=False):
        # the resting pose already sits close to the floor, so allow down to "the lower of the two ends' TCP heights, minus 1 cm"
        tmin = min(FLOOR_Z + (TRANSIT_CLEAR if transit else 0.02), tcp_z(q_from) - 0.01, tcp_z(q_to) - 0.01)
        # link7 も同じ考え方（2026-11-21 北九州）: 机に寝かせた休みの姿勢は手首が机から 4 cm ほどで、5 cm の線を始めから下回る。
        # 区間の端が低いときだけ「低い方の端の高さ − 1 cm」まで許す。両端が高い区間の線（FLOOR_Z + 5 cm）は変わらない
        l7min = min(FLOOR_Z + 0.05, l7_z(q_from) - 0.01, l7_z(q_to) - 0.01)
        for s in np.linspace(0, 1, 20):
            q = np.deg2rad(np.asarray(q_from) + (np.asarray(q_to) - np.asarray(q_from)) * s)
            M = ik.fk(q); tcp = M.translation + M.rotation @ [0, 0, 0.2145]
            if KEEPOUT:  # 腕が胸のカメラと取り付け部の箱に入る動き（本体のフレームに当たる）を止める
                js = [ik.data.oMi[ik.model.getJointId(f"openarm_{ARM}_joint{j}")].translation.copy() for j in range(3, 8)]
                P = np.array(js + [(p0 + p1) / 2 for p0, p1 in zip(js[:-1], js[1:])]
                             + [M.translation + M.rotation @ [0, 0, TCP_LEN], M.translation + M.rotation @ [0, 0, TIP_LEN]])
                for b in KEEPOUT:
                    if np.any((P[:, 0] >= b[0]) & (P[:, 0] <= b[1]) & (P[:, 1] >= b[2]) & (P[:, 1] <= b[3]) & (P[:, 2] >= b[4]) & (P[:, 2] <= b[5])):
                        raise Blocked(f"[run] {name}: at s={s:.2f} the arm enters the keep-out box {b.round(3).tolist()}: "
                                         "it would hit the body frame / chest camera. Stopping")
            if M.translation[2] < l7min or (not allow_low and tcp[2] < tmin):
                raise Blocked(f"[run] {name}: at s={s:.2f} link7 z={M.translation[2]:.3f} TCP z={tcp[2]:.3f}, too close to the floor. Stopping")
        if ACOLL is not None:
            held = HELD_L if any(name.startswith(h) for h in HELD_LEGS) else None
            ok, worst = ACOLL.check_path(q_from, q_to, held=held, open_m=0.0 if held else OPEN_M)
            if not ok:
                raise Blocked(f"[run] {name}: 腕の形が当たる（{arm_collision.describe(worst)}）。Stopping")
        if CUROBO is not None:
            # 机は、端の姿勢がもともと机に近い区間（机に寝かせた休みの姿勢）だけ「低い方の端 − 1 cm」まで許す（上の link7 と同じ考え方）
            d_end = min(CUROBO.check_path([q], n=1)[0]["table/arm"][0] for q in (q_from, q_to))
            bad = CUROBO.violations([q_from, q_to], names=[name], table_margin=min(CUROBO.TABLE_MARGIN, d_end - 0.01))
            if bad:
                raise Blocked(f"[run] cuRobo: {CUROBO.describe(bad)}。腕が置き場の周りの物（本体・カメラ・机）に当たる。Stopping")
        if not quiet:
            print(f"[run] check ok: {name}")

    BLEND_S, RAMP_S, STEP_S = 0.3, 0.4, 0.01

    def blend_traj(q0, g0, legs, speed):
        """One trajectory through every pose of `legs` ([(name, path, gripper at the leg's end, transit)], path = poses
        after the leg's start) from q0. Same profile as lerobot align_through_poses (damiao_alignment.py, Orboh/lerobot
        a720ec19): L-inf joint arc length, constant `speed` with a RAMP_S ramp at both ends only, corners rounded by a
        BLEND_S moving average (<= about speed * BLEND_S / 4 deg off the polyline). Inline copy, not an import: that
        function computes and sends in one call, and here the same samples are collision-checked before they are sent.
        The gripper is not part of the arc length; it moves linearly within each leg.
        Returns (joints [n, 7], gripper [n], leg index [n], off-polyline [n] deg)."""
        P, G, L = [list(map(float, q0))], [float(g0)], [0]
        for li, (_, path, g, _) in enumerate(legs):
            gs = G[-1]
            for k, q in enumerate(path):
                if max(abs(x - y) for x, y in zip(q, P[-1])) < 1e-6:
                    continue
                P.append(list(map(float, q))); G.append(gs + (g - gs) * (k + 1) / len(path)); L.append(li)
        if len(P) < 2:
            return None, None, None, None
        P, G = np.array(P), np.array(G)
        cum = np.concatenate([[0.0], np.cumsum(np.abs(np.diff(P, axis=0)).max(axis=1))]); total = float(cum[-1])
        cruise = total / speed; ramp = min(RAMP_S, max(cruise, STEP_S)); t_total = max(cruise + ramp, 0.3)
        n = max(1, round(t_total / STEP_S)); t = np.arange(1, n + 1) * (t_total / n); vmax = total / max(t_total - ramp, 1e-9)
        sa = np.where(t < ramp, 0.5 * vmax * t ** 2 / ramp,
                      np.where(t > t_total - ramp, total - 0.5 * vmax * (t_total - t) ** 2 / ramp, vmax * (t - ramp / 2)))
        sa = np.clip(sa, 0.0, total)
        X = np.stack([np.interp(sa, cum, P[:, j]) for j in range(P.shape[1])] + [np.interp(sa, cum, G)], axis=1)
        w = int(round(BLEND_S / STEP_S))
        if w >= 3 and len(X) > w:
            pad = np.concatenate([np.repeat(X[:1], w, axis=0), X, np.repeat(X[-1:], w, axis=0)])
            X = np.stack([np.convolve(pad[:, j], np.ones(w) / w, mode="same") for j in range(X.shape[1])], axis=1)[w:-w]
            u = np.linspace(0.0, 1.0, w)
            X[-w:] += (3 * u ** 2 - 2 * u ** 3)[:, None] * (np.append(P[-1], G[-1]) - X[-1])[None, :]
        leg = np.array(L)[np.clip(np.searchsorted(cum, sa, side="left"), 1, len(cum) - 1)]
        # how far each sample is off the polyline (nearest point on any leg segment, Euclidean in degrees)
        A, B = P[:-1], P[1:]; D = B - A; dd = np.maximum((D * D).sum(1), 1e-12)
        u = np.clip(((X[:, None, :7] - A[None]) * D[None]).sum(2) / dd[None], 0.0, 1.0)
        off = np.sqrt(((X[:, None, :7] - (A[None] + u[..., None] * D[None])) ** 2).sum(2)).min(axis=1)
        return X[:, :7], X[:, 7], leg, off

    def chain_plan(q0, g0, legs, speed, label):
        """Check the rounded trajectory (every 0.1 s, the same floor / keep-out / arm-shape / cuRobo checks as a leg).
        Returns the chain to pass to go_chain, or None (then the legs are moved one by one with stops, as before)."""
        legs = [l for l in legs if l[1]]
        if not a.blend or len(legs) == 0 or sum(len(l[1]) for l in legs) < 2:
            return None
        X, _, leg, off = blend_traj(q0, g0, legs, speed)
        if X is None:
            return None
        # Where the rounded trajectory stays on the polyline (off <= 0.05 deg) it is the legs already checked; check the
        # rounded parts (around the corners and the end) every 0.1 s, from the last on-polyline sample to the next one.
        runs, i = [], 0
        while i < len(X):
            if off[i] > 0.05:
                j = i
                while j + 1 < len(X) and off[j + 1] > 0.05:
                    j += 1
                runs.append((max(i - 1, 0), min(j + 1, len(X) - 1))); i = j + 1
            else:
                i += 1
        t_chk = time.time(); n_chk = 0
        try:
            for i0, i1 in runs:
                idx = list(range(i0, i1, 10)) + [i1]
                prev = list(q0) if i0 == 0 else list(X[i0])
                for i in idx[1:] if i0 > 0 else idx:
                    name, _, _, transit = legs[leg[i]]
                    check(prev, list(X[i]), f"{name} (つなぎ)", transit=transit, quiet=True); prev = list(X[i]); n_chk += 1
        except Blocked as e:
            print(f"[run] {label}: つないだ経路は当たる（{str(e)[:140]}）→ 区間ごとに止まって動く", flush=True)
            return None
        print(f"[run] {label}: {len(legs)} 区間・{sum(len(l[1]) for l in legs)} 姿勢を止まらずに動く（{len(X) * STEP_S:.1f} 秒・"
              f"{speed:.0f}°/秒）check ok（丸めた所 {len(runs)} か所・{n_chk} 回・{time.time() - t_chk:.1f} 秒）", flush=True)
        return {"q0": list(q0), "g0": float(g0), "legs": legs, "speed": float(speed), "label": label, "X": X, "leg": leg}

    def load_obstacles(q_now):
        """記録のカメラの障害物（plan.py が残した点）を、今の腕の角度 q_now で腕の点を消し直してサーバーに入れる。
        記録に無ければ障害物を空にする（前の記録の障害物を残さない）。"""
        if CUROBO is None:
            return
        pf, jf = os.path.join(a.cap_dir, "obstacle_points.npy"), os.path.join(a.cap_dir, "obstacles.json")
        if not (os.path.exists(pf) and os.path.exists(jf)):
            CUROBO.set_obstacle_boxes([])
            print("[run] cuRobo: カメラの障害物の記録が無い（置き場の物だけで見る）", flush=True)
            return
        meta = json.load(open(jf))
        # arm_margin 5 cm (was 3): the cuRobo server's arm model has no left-arm correction (ik/openarm_left_ik.py, up to
        # 2.4 cm at home v5), so the real jaws at home stayed in the cloud as a 'seen' box and blocked home->standby (16:44)
        r = CUROBO.set_obstacle_points(np.load(pf), [list(q_now)] + meta["arm_qs"], meta["cam_pos"], meta["floor_z"], arm_margin=0.05)
        print(f"[run] cuRobo: カメラで見た障害物の箱 {len(r['boxes'])} 個" + "".join(
            f"・{b['name']} 中心 {np.round(b['pose'][:3], 2).tolist()} 大きさ {np.round(b['dims'], 2).tolist()}" for b in r["boxes"]), flush=True)

    def floor_line(q_from, q_to):
        """check と同じ床の線（回り込みの確かめに渡す）。"""
        return {"tcp_min_z": min(FLOOR_Z + 0.02, tcp_z(q_from) - 0.01, tcp_z(q_to) - 0.01),
                "l7_min_z": min(FLOOR_Z + 0.05, l7_z(q_from) - 0.01, l7_z(q_to) - 0.01), "tcp_len": TCP_LEN}

    def route(q_from, q_to, name):
        """移動の区間を確かめ、当たるなら cuRobo で回り込む。返り値は通る姿勢の列（始点を除き終点を含む）。

        物の近くの区間（pregrasp↔grasp↔lift・置く）には使わない（そこは直線で動かし、当たれば check が止める）。
        回り込みの経由点の間も、床・keep-out の箱・cuRobo の確認（check）をもう一度通す。"""
        try:
            check(q_from, q_to, name, transit=True)
            return [list(q_to)]
        except Blocked as e:
            if CUROBO is not None:
                try:
                    d_end = min(CUROBO.check_path([q], n=1)[0]["table/arm"][0] for q in (q_from, q_to))
                    vias = CUROBO.detour(q_from, q_to, table_margin=min(CUROBO.TABLE_MARGIN, d_end - 0.01), floor=floor_line(q_from, q_to))
                    if not vias:
                        raise Blocked(f"{e}（cuRobo の回り込みも作れない）")
                    prev = q_from
                    for k, v in enumerate(vias):
                        check(prev, v, f"{name} 回り込み {k + 1}/{len(vias)}")
                        prev = v
                    print(f"[run] {name}: 直線は当たるので回り込む（経由点 {len(vias)} 個）", flush=True)
                    return [list(v) for v in vias]
                except (Blocked, SystemExit) as e2:
                    # 2026-11-22: cuRobo does not know the cart tray (arm_collision only): try the simple detours below
                    print(f"[run] {name}: cuRobo の回り込みは使えない（{str(e2)[:120]}）。手先をずらす回り込みを試す", flush=True)
            if True:
                # no cuRobo (the right arm: its cuRobo model does not exist yet, 2026-11-22): try simple detours that
                # keep the hand orientation: over the target raised 8-25 cm, or the start raised first, then down.
                # Every leg goes through check() again (floor, keep-out boxes), else stop as before.
                # 2026-11-22: the clamp sits high in front of the chest, so going higher does not get round it; also try
                # moving the hand forward / outward (away from the body, -y for the right arm) first or last.
                out = -1.0 if ARM == "right" else 1.0
                offs = [(0, 0, dz) for dz in (0.08, 0.12, 0.18, 0.25)] + \
                       [(dx, out * dy, dz) for dz in (0.0, 0.08) for dx in (0.10, 0.15, 0.20) for dy in (0.0, 0.10)] + \
                       [(0.0, out * 0.12, 0.05)]
                for off in offs:
                    dz = off[2]
                    for path in (("to_up",), ("from_up",), ("from_up", "to_up")):
                        try:
                            vias = []
                            if "from_up" in path:
                                vias.append(shifted(q_from, off))
                            if "to_up" in path:
                                vias.append(shifted(q_to, off))
                            vias.append(list(q_to))
                            prev = q_from
                            for k, v in enumerate(vias):
                                check(prev, v, f"{name} 回り込み {k + 1}/{len(vias)}", transit=True); prev = v
                            print(f"[run] {name}: 直線は当たる（または低い）ので回り込む（手先を {np.round(np.array(off)*100).astype(int).tolist()} cm ずらす・経由点 {len(vias) - 1} 個）", flush=True)
                            return [list(v) for v in vias]
                        except (Blocked, SystemExit):
                            continue
                raise e

    pre, grasp, lift = plan["q_pre_deg"], plan["q_grasp_deg"], plan["q_lift_deg"]
    OPEN_M = float(plan.get("open_mm") or 93.0) / 1000.0  # the jaw gap while moving open (arm_collision fingers)
    if plan.get("gripper_open_deg") is not None:  # plan.py --open-mm: open only as wide as the gap in the pile allows
        G_OPEN = float(plan["gripper_open_deg"]); print(f"[run] gripper opens to {G_OPEN} deg ({plan.get('open_mm')} mm, from the plan)")
    carry, place = plan.get("q_carry_deg"), plan.get("q_place_deg")
    dcarry, drop, lhigh = plan.get("q_drop_carry_deg"), plan.get("q_drop_deg"), plan.get("q_lift_high_deg")  # grasp/tools/plan_drop.py: drop beside the left shoulder
    import pinocchio as pin

    def carry_routes(lhigh_, up_, dcarry_):
        """Holding the bar: lift high → drop carry directly when that is clear, so the held bar never swings past the
        clamp / chest camera next to the raise pose (Sota 11/22: アルミフレームを持った状態で左にアームが当たっている).
        Otherwise the old way via the raise pose. Returns (lift high→raise, raise→drop carry) paths."""
        try:
            d = route(lhigh_, dcarry_, "lift high→drop carry")
            print("[run] 持ったまま lift high から落とす所へ直接運ぶ（raise を通らない）", flush=True)
            return [], d
        except (Blocked, SystemExit) as e:
            print(f"[run] 直接は運べない（{e}）: raise を通る", flush=True)
            return route(lhigh_, up_, "lift high→raise"), route(up_, dcarry_, "raise→drop carry")

    def raised_or_same(qd):
        """Raise for floor clearance when possible; the home pose is already high (and 8 cm more is out of reach)."""
        for dz in (0.08, 0.05, 0.03):
            try:
                return raised(qd, dz)
            except SystemExit:
                pass
        return list(qd)

    def raised_some(qd):
        """休みの姿勢から持ち上げる: 8 cm で解けなければ 5 cm・3 cm（電源を入れ直した後など、休みの姿勢が少し違うとき）。"""
        for dz in (0.08, 0.05, 0.03):
            try:
                return raised(qd, dz)
            except SystemExit:
                pass
        raise SystemExit("[run] can't solve the lift (8/5/3 cm) from the start pose. Stopping")

    def shifted(qd, off):
        """Pose with the TCP moved by off (x, y, z in the base frame) keeping its orientation."""
        M = ik.fk(np.deg2rad(qd)); T = pin.SE3(M.rotation, M.translation + np.asarray(off, float))
        q, ok, _, _ = ik.solve(T, np.deg2rad(qd), position_only=False, max_iter=300)
        if not ok or not ik.within_limits(q):
            raise SystemExit(f"[run] can't solve the hand moved by {off}")
        return list(np.rad2deg(q))

    def raised(qd, dz=0.08):
        """Pose with the TCP raised dz straight up while keeping its orientation (lift off the floor first when leaving the resting pose)."""
        M = ik.fk(np.deg2rad(qd)); T = pin.SE3(M.rotation, M.translation + np.array([0, 0, dz]))
        q, ok, _, _ = ik.solve(T, np.deg2rad(qd), position_only=False, max_iter=300)
        if not ok or not ik.within_limits(q):
            raise SystemExit("[run] can't solve the lift from the start pose. Stopping")
        return list(np.rad2deg(q))

    home = None
    if a.home:
        import yaml
        hy = yaml.safe_load(open(a.home)); home = [float(hy[j]) for j in J]
    standby = None
    if a.standby:
        import yaml
        sy = yaml.safe_load(open(a.standby)); standby = [float(sy[j]) for j in J]
    if not a.go:
        start = [float(v) for v in a.start.split(",")] if a.start else plan["q_start_deg"]
        load_obstacles(start)
        if home is not None:  # rest -> raise -> home, then home is the start of the grasp sequence
            if max(abs(x - y) for x, y in zip(start, home)) >= 10.0:  # already held at home: nothing to check
                up0 = raised(start); route(start, up0, "rest→raise"); route(up0, home, "raise→home")
                route(home, up0, "home→raise over rest")
            start = home
            if standby is not None:
                d_hsb = route(home, standby, "home→standby"); d_sbh = route(standby, home, "standby→home"); start = standby
        up = raised_or_same(start) if home is not None else raised_some(start)
        if push is not None:
            route(start, up, "start→raise"); route(up, pre, "raise→above the push"); check(pre, grasp, "above→down", allow_low=True)
            prev = grasp
            for k, q in enumerate(push["q_path_deg"]):
                check(prev, q, f"push step {k + 1}", allow_low=True); prev = q
            check(prev, lift, "push end→rise", allow_low=True); route(lift, up, "rise→raise"); route(up, start, "raise→start")
            print(f"[run] push {push['push_dist_m']*100:.0f} cm toward {push['push_dir_base']} from {push['contact_base_m']}")
            print("[run] DRY RUN (add --go to move)"); return 0
        d_su = route(start, up, "start→raise"); d_up = route(up, pre, "raise→pregrasp"); check(pre, grasp, "pregrasp→grasp", allow_low=True)
        check(grasp, lift, "grasp→lift", allow_low=True); route(lift, pre, "lift→pregrasp"); d_pu = route(pre, up, "pregrasp→raise"); d_us = route(up, start, "raise→start")
        if carry is not None:
            route(lift, carry, "lift→carry"); check(carry, place, "carry→place", allow_low=True)
            check(place, carry, "place→carry", allow_low=True); route(carry, up, "carry→raise")
            print("[run] place at the new spot " + " ".join(f"{v:+.1f}" for v in place))
        if drop is not None:
            check(lift, lhigh, "lift→lift high"); d_lu, d_udc = carry_routes(lhigh, up, dcarry)
            check(dcarry, drop, "drop carry→drop")
            check(drop, dcarry, "drop→drop carry"); d_dcu = route(dcarry, up, "drop carry→raise")
            print("[run] drop beside the shoulder " + " ".join(f"{v:+.1f}" for v in drop))
        if a.blend:  # the same chains as with --go (what would move without stopping, and that they pass the checks)
            g0 = 0.0
            sb = standby is not None and home is not None
            tail = [("raise→start", d_us, g0, True)] + ([("standby→home", d_sbh, g0, True)] if sb else [])
            chains = []
            chains.append(chain_plan(home if sb else start, g0, ([("home→standby", d_hsb, g0, True)] if sb else []) + [("start→raise", d_su, g0, True)]
                       + ([("raise→pregrasp", d_up, G_OPEN, True)] if a.held_check != "camera" else []), a.transit_dps, "行き（→ pregrasp）"))
            chains.append(chain_plan(pre, G_OPEN, [("pregrasp→raise", d_pu, G_OPEN, True)] + tail, a.transit_dps, "戻り（空のとき）"))
            if drop is not None:
                chains.append(chain_plan(lift, G_CLOSE, [("lift→lift high", [lhigh], G_CLOSE, False), ("lift high→raise", d_lu, G_CLOSE, True),
                                           ("raise→drop carry" if d_lu else "lift high→drop carry", d_udc, G_CLOSE, True)],
                           a.near_dps, "持って運ぶ（→ 落とす所の上）"))
                chains.append(chain_plan(dcarry, G_OPEN, [("drop carry→raise", d_dcu, G_OPEN, True)] + tail, a.transit_dps, "戻り（落とした後）"))
            chains = [c for c in chains if c is not None]
            if chains and os.environ.get("GRASP_BLEND_PREVIEW") == "1":  # (slow: ~5 s of clearances) MuJoCo preview: MUJOCO_GL=egl ~/dimos/.venv/bin/python grasp/tools/mujoco_preview.py <cap>/blend_traj.json <tag>
                other = "left" if ARM == "right" else "right"; ho = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), f"grasp/config/home_{other}_cart-sym-v5.yaml")
                import yaml
                q_other = [float(yaml.safe_load(open(ho))[j]) for j in J] if os.path.exists(ho) else [0.0] * 7
                frames = []
                for ci, c in enumerate(chains):
                    for i in list(range(0, len(c["X"]), 3)) + [len(c["X"]) - 1]:
                        nm = c["legs"][c["leg"][i]][0]; held = HELD_L if any(nm.startswith(h) for h in HELD_LEGS) else None
                        cl = ACOLL.clearances(np.deg2rad(c["X"][i]), held=held, open_m=0.0 if held else OPEN_M) if ACOLL is not None else {}
                        frames.append({"arm": ARM, "q": {ARM: [float(v) for v in c["X"][i]], other: q_other},
                                       "label": f"{ARM}: chain {ci + 1}/{len(chains)} {c['speed']:.0f} deg/s / {nm.replace('→', ' -> ')}", "clear_cm": {k: round(v[0] * 100, 1) for k, v in cl.items()} or {"none": 99.0}})
                    frames += [dict(frames[-1])] * 15
                json.dump({"frames": frames}, open(os.path.join(a.cap_dir, "blend_traj.json"), "w"))
                print(f"[run] つないだ経路 {len(chains)} 本を {os.path.join(a.cap_dir, 'blend_traj.json')} に書いた（MuJoCo で見る用）")
        print("[run] raise pose " + " ".join(f"{v:+.1f}" for v in up))
        print("[run] DRY RUN (add --go to move)"); return 0

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import OPENARM_ALIGN_KP, soft_move_to_position
    KP = dict(OPENARM_ALIGN_KP, gripper=a.grip_kp)
    if a.grip_close is not None:
        G_CLOSE = a.grip_close
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    grav = OpenArmGravityModel(URDF, ARM, 1.0).torque
    from grasp.soft_release import release_and_disconnect  # 2026-11-22 Sota: 力を抜くときはゆっくり（ダンプのように）
    robot = OpenArmFollower(OpenArmFollowerConfig(port=os.environ.get("GRASP_CAN_PORT", "can_fol_r" if ARM == "right" else "can_fol_l"), side=ARM, id=os.environ.get("GRASP_ROBOT_ID", f"follower_{ARM}"), align_on_connect=False, cameras={},
                                                  disable_torque_on_disconnect=not (a.hold_home and a.home)))
    from grasp.held_connect import connect_holding  # 11/22: robot.connect() turned torque off -> the held arm fell onto the cart
    connect_holding(robot, grav)
    obs = robot.get_observation(); start = [float(obs[f"{j}.pos"]) for j in J]
    print("[run] 今の腕の角度 " + ",".join(f"{v:.1f}" for v in start), flush=True)
    if home is not None:  # 11/22: did the held arm stay at home between trials?
        dh = max(abs(x - y) for x, y in zip(start, home))
        print(f"[run] start vs home max diff {dh:.1f} deg" + ("  (WARNING: the arm moved since the last trial)" if dh > 2.0 else ""), flush=True)
    log = open(os.path.join(a.cap_dir, "traj.csv"), "w", newline=""); w = csv.writer(log)
    w.writerow(["t", "phase"] + [f"{j}.pos" for j in J] + ["gripper.pos"])

    def rec(phase):
        o = robot.get_observation(); w.writerow([f"{time.time():.3f}", phase] + [f"{o[f'{j}.pos']:.2f}" for j in J] + [f"{o['gripper.pos']:.2f}"])

    def go(q, speed, g, phase, correct=True):
        goal = {j: float(q[i]) for i, j in enumerate(J)}; goal["gripper"] = float(g)
        soft_move_to_position(robot.bus, goal, kp=KP, torque_ff_fn=grav, max_speed_dps=speed)
        # The soft gains don't fully reach the target (first real run: J7 14.5° short, TCP 5 cm short).
        # Add the remaining error to the command and send it again (up to 3 times, at most 15° of added offset per joint).
        cmd = dict(goal); err = None
        for _ in range(3 if correct else 0):
            o = robot.get_observation()
            err = {j: goal[j] - float(o[f"{j}.pos"]) for j in J}
            if max(abs(v) for v in err.values()) < 1.0:
                break
            for j in J:
                cmd[j] = goal[j] + float(np.clip(cmd[j] - goal[j] + err[j], -15, 15))
            soft_move_to_position(robot.bus, cmd, kp=KP, duration_s=0.6, torque_ff_fn=grav)
        age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
        if age > 0.3:  # 返事が無いまま「done」と書かない（2026-11-21 18:29: 非常停止の後に置く・離すが done になった）
            raise MotorLost(f"[run] {phase}: モーターの返事が {age:.1f} 秒来ない（非常停止・電源・通信）。腕に指令を送るのをやめる")
        rec(phase)
        o = robot.get_observation(); e = max(abs(goal[j] - float(o[f"{j}.pos"])) for j in J)
        print(f"[run] {phase} done (max joint error {e:.1f}°)", flush=True)
        # Stop when the arm did not follow (hit something, or the motors stopped answering and the reading is stale).
        # 2026-11-21 Kitakyushu: after the arm hit the camera mount, run_grasp kept sending the remaining phases
        # ("max joint error 104°/129°"). Raise TrackingLost: the sequence stops; the finally block returns slowly only
        # if fresh readings are close to where they should be, otherwise it just disconnects (torque stays as configured).
        if e > a.max_track_err:
            raise TrackingLost(f"{phase}: max joint error {e:.1f}° > {a.max_track_err}°")

    def wrist_frame(tag):
        """Wrist D405 color + aligned depth [m] (saved as evidence). None if the camera fails."""
        try:
            import cv2, pyrealsense2 as rs
            pw = rs.pipeline(); rw = rs.config(); rw.enable_device(os.environ.get("GRASP_WRIST_SERIAL", "409122272149"))
            rw.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rw.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            pr = pw.start(rw); scw = pr.get_device().first_depth_sensor().get_depth_scale(); alw = rs.align(rs.stream.color)
            try:
                for _ in range(10):
                    pw.wait_for_frames()
                fw = alw.process(pw.wait_for_frames())
                col = np.asanyarray(fw.get_color_frame().get_data()).copy()
                dep = np.asanyarray(fw.get_depth_frame().get_data()).astype(np.float32) * scw
            finally:
                pw.stop()
            cv2.imwrite(os.path.join(a.cap_dir, f"wrist_{tag}.jpg"), col)
            np.save(os.path.join(a.cap_dir, f"wrist_{tag}_depth.npy"), dep.astype(np.float16))
            return dep
        except Exception as e:  # noqa: BLE001
            print(f"[run] wrist camera failed ({tag}): {e}", flush=True)
            return None

    jaw_mask = None  # pixels the jaws themselves cover anywhere between open and closed (from the references)

    def held_by_camera(tag):
        """True / False from the wrist camera, None when it can't tell (camera failed or no references)."""
        if jaw_mask is None:
            return None
        dep = wrist_frame(tag)
        if dep is None:
            return None
        import cv2
        near = (dep > 0.03) & (dep < a.held_near)
        new = near & ~jaw_mask
        n = int(new.sum())
        cv2.imwrite(os.path.join(a.cap_dir, f"wrist_{tag}_new.png"), new.astype(np.uint8) * 255)
        print(f"[run] wrist camera: {n} new near pixels (< {a.held_near:.2f} m, outside the jaws) vs {a.held_px} needed", flush=True)
        return n >= a.held_px

    def go_route(path, speed, g, phase):
        """route の姿勢の列を順に動かす。経由点でも位置の補正をかけ、遅れたまま次の直線に入らない。"""
        for k, q in enumerate(path):
            go(q, speed, g, phase if k == len(path) - 1 else f"{phase} (回り込み {k + 1}/{len(path) - 1})")

    def go_chain(ch, fallback_phases):
        """--blend: send the checked trajectory of chain_plan without stopping, then correct once at the end (go).
        Falls back to go_route per leg when the arm is not where the chain was planned from (> 3 deg)."""
        from lerobot.motors.damiao.damiao_alignment import OPENARM_ALIGN_KD
        o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
        if max(abs(x - y) for x, y in zip(q_now, ch["q0"])) > 3.0:
            print(f"[run] {ch['label']}: 腕が計画の始点から 3° 以上ずれている → 区間ごとに動く", flush=True)
            for (name, path, g, _), ph in zip(ch["legs"], fallback_phases):
                go_route(path, ch["speed"], g, ph)
            return
        X, Gr, leg, _ = blend_traj(q_now, float(o["gripper.pos"]), ch["legs"], ch["speed"])
        st, lim = robot.bus._last_known_states, a.max_track_err + 15.0
        t0 = t_next = time.time(); max_lag = 0.0; cur = -1
        for i in range(len(X)):
            target = {j: float(X[i][k]) for k, j in enumerate(J)}; target["gripper"] = float(Gr[i])
            ff = grav(target)
            robot.bus._mit_control_batch({m: (KP[m], OPENARM_ALIGN_KD[m], target[m], 0.0, float(ff.get(m, 0.0))) for m in target})
            age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
            if age > 0.3:
                raise MotorLost(f"[run] {ch['label']}: モーターの返事が {age:.1f} 秒来ない（非常停止・電源・通信）。腕に指令を送るのをやめる")
            lag = max(abs(target[j] - float(st[j]["position"])) for j in J); max_lag = max(max_lag, lag)
            if lag > lim:
                raise TrackingLost(f"{ch['label']}: 動いている途中で目標から {lag:.1f}° 遅れた（> {lim:.0f}°・当たった？）")
            if leg[i] != cur:  # traj.csv: when each leg starts (from the cached replies, no extra bus read)
                cur = int(leg[i])
                w.writerow([f"{time.time():.3f}", f"{ch['legs'][cur][0]} (つなぎ)"] + [f"{float(st[j]['position']):.2f}" for j in J]
                           + [f"{float(st['gripper']['position']):.2f}"])
            t_next += STEP_S
            time.sleep(max(0.0, t_next - time.time()))
        print(f"[run] {ch['label']}: 止まらずに {time.time() - t0:.1f} 秒（一番の遅れ {max_lag:.1f}°）", flush=True)
        name, path, g, _ = ch["legs"][-1]
        go(path[-1], ch["speed"], g, fallback_phases[-1])  # one correction at the end + the usual checks and traj.csv row

    watcher = None
    try:  # 動く前の準備（経路を作る・カメラの背景）。ここで止まったら、腕を動かさずにモーターとの接続を切る
        g_now = float(obs["gripper.pos"])
        TR, NR = a.transit_dps, a.near_dps
        rest = list(start)  # where the arm hung limp before the trial: go back here before torque off
        load_obstacles(start)
        at_home = home is not None and max(abs(x - y) for x, y in zip(start, home)) < 10.0
        if home is not None and not at_home:
            try:
                up0 = raised_some(start)
                r_rest_up = route(start, up0, "rest→raise"); r_up_home = route(up0, home, "raise→home")
                r_home_up0 = route(home, up0, "home→raise over rest")
            except SystemExit as e:  # 11/22: a held arm that sagged near home (raise IK at a joint limit): go straight if clear
                print(f"[run] 持ち上げの姿勢が作れない（{e}）: 今の姿勢から home へ直接", flush=True)
                up0 = list(start); r_rest_up = []; r_up_home = route(start, home, "now→home"); r_home_up0 = route(home, start, "home→now")
        # 動く前に移動の区間の経路を全部作る（回り込みを作れない区間があれば、腕を動かす前に止まる）
        s0 = standby if standby is not None else (home if home is not None else start)
        up = raised_or_same(s0) if home is not None else raised_some(s0)
        if standby is not None:
            r_home_sb = route(home, standby, "home→standby"); r_sb_home = route(standby, home, "standby→home")
        r_start_up = route(s0, up, "start→raise"); r_up_pre = route(up, pre, "raise→pregrasp")
        r_pre_up = route(pre, up, "pregrasp→raise"); r_up_start = route(up, s0, "raise→start")
        if push is not None:  # the push ends at its rise pose: route back to raise from there
            r_lift_up = route(lift, up, "rise→raise")
        if carry is not None:
            r_lift_carry = route(lift, carry, "lift→carry"); r_carry_up = route(carry, up, "carry→raise")
        if drop is not None:  # grasp/tools/plan_drop.py: lift high (straight) → raise → drop carry (routes) → drop (straight)
            r_lhigh_up, r_up_dcarry = carry_routes(lhigh, up, dcarry)
            r_dcarry_up = route(dcarry, up, "drop carry→raise")
        # --blend: chains of legs moved without stopping (checked here, before the arm moves)
        g_end0 = g_now if g_now > -10 else G_OPEN
        out_to_pre = push is None and a.held_check != "camera"  # the camera check takes its references at raise: stop there
        CH_OUT = chain_plan(s0 if standby is None else home, g_now,
                            ([("home→standby", r_home_sb, g_now, True)] if standby is not None else [])
                            + [("start→raise", r_start_up, g_now, True)]
                            + ([("raise→pregrasp", r_up_pre, G_OPEN, True)] if out_to_pre else []), TR, "行き（→ pregrasp）")
        if CH_OUT is None:
            out_to_pre = False
        ret_tail = [("raise→start", r_up_start, g_end0, True)] + ([("standby→home", r_sb_home, g_end0, True)] if standby is not None else [])
        CH_RET_EMPTY = chain_plan(pre, G_OPEN, [("pregrasp→raise", r_pre_up, G_OPEN, True)] + ret_tail, TR, "戻り（空のとき）")
        CH_HOLD = CH_RET_DROP = None
        if drop is not None:
            CH_HOLD = chain_plan(lift, G_CLOSE, [("lift→lift high", [lhigh], G_CLOSE, False),
                                                 ("lift high→raise", r_lhigh_up, G_CLOSE, True),
                                                 ("raise→drop carry" if r_lhigh_up else "lift high→drop carry", r_up_dcarry, G_CLOSE, True)],
                                 NR, "持って運ぶ（→ 落とす所の上）")
            CH_RET_DROP = chain_plan(dcarry, G_OPEN, [("drop carry→raise", r_dcarry_up, G_OPEN, True)] + ret_tail, TR, "戻り（落とした後）")
        ret_chain = None
        watcher = None
        if a.avoid:
            if CUROBO is None or carry is None:
                raise SystemExit("[run] --avoid には GRASP_CUROBO と置き先（grasp/tools/plan_place.py）が要る")
            from grasp import reactive
            T_bc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
            watcher = reactive.Watcher(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"), T_bc, FLOOR_Z)
            watcher.start()  # 動き出す前に背景（机の上の様子）を覚える。この間は何も割り込ませない
            print("[run] 胸のカメラで背景を覚える（約 2 秒・割り込ませない）", flush=True)
            t0 = time.time()
            while watcher.stamp == 0.0 and time.time() - t0 < 8.0 and watcher.error is None:
                time.sleep(0.05)
            if watcher.stamp == 0.0:
                raise SystemExit(f"[run] 胸のカメラの映像が来ない（{watcher.error}）。動かさない")

            # 持っているアルミの点（掴んだ姿勢の link7 の座標）。運ぶ間は手と一緒に動かし、そのまわりの点を割り込んだ物と見ない
            held_l7 = reactive.held_in_link7(np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3], grasp)

            def carry_watch(path, g, phase, holding_now=False):
                reactive.follow(robot.bus, grav, path, a.avoid_dps, g, watcher, T_bc[:3, 3], FLOOR_Z, floor_line, phase,
                                held_l7=held_l7 if holding_now else None)
    except BaseException:
        print("[run] 動く前に止まった（腕は動かしていない）。接続を切る", flush=True)
        if watcher is not None:
            watcher.stop()
        release_and_disconnect(robot, grav); log.close()
        raise
    done = holding = motor_lost = no_return = False
    video = video_start(a.cap_dir, a.avoid)  # 11/23 Sota: 試行は動画も残す（胸カメラ → video_chest.mp4、S3 へ）
    try:
        if home is not None:  # yesterday's start pose for every trial (Sota 11/20)
            if not at_home:
                go_route(r_rest_up, TR, g_now, "raise from rest")
            go_route(r_up_home, TR, g_now, "home") if not at_home else go(home, TR, g_now, "home")
            start = home
            if standby is not None and CH_OUT is None:  # hands up and forward first, clear of the cart (Sota 11/22)
                go_route(r_home_sb, TR, g_now, "standby")
            if standby is not None:
                start = standby
        if CH_OUT is not None:
            go_chain(CH_OUT, (["standby"] if standby is not None else []) + ["raise"] + (["pregrasp"] if out_to_pre else []))
        else:
            go_route(r_start_up, TR, g_now, "raise")
        if push is not None:  # grasp/tools/plan_push.py: closed jaws push the bar sideways, no grasp
            go(up, TR, G_CLOSE, "close gripper")
            go_route(r_up_pre, TR, G_CLOSE, "above the push")
            go(grasp, NR, G_CLOSE, "down beside the bar")
            for k, q in enumerate(push["q_path_deg"]):
                go(q, NR, G_CLOSE, f"push {k + 1}/{len(push['q_path_deg'])}", correct=False)
            go(lift, NR, G_CLOSE, "rise")
            go_route(r_lift_up, TR, G_CLOSE, "back to raise")
            done = True  # at raise: the finally block takes the planned route home
            raise PushDone
        ref_o = None
        if a.held_check == "camera":  # empty-jaw references in the air: closed, then open (the jaws' own pixels)
            import cv2
            go(up, TR, G_CLOSE, "close for the empty-jaw reference", correct=False); time.sleep(0.5)
            ref_c = wrist_frame("ref_closed")
            go(up, TR, G_OPEN, "open gripper")
            time.sleep(0.5); ref_o = wrist_frame("ref_open")
            if ref_c is not None and ref_o is not None:
                jm = ((ref_c > 0.03) & (ref_c < a.held_near)) | ((ref_o > 0.03) & (ref_o < a.held_near))
                jaw_mask = cv2.dilate(jm.astype(np.uint8), np.ones((31, 31), np.uint8)).astype(bool)
                print(f"[run] empty-jaw reference: jaws cover {int(jaw_mask.sum())} px", flush=True)
        elif not out_to_pre:
            go(up, TR, G_OPEN, "open gripper")
        if not out_to_pre:
            go_route(r_up_pre, TR, G_OPEN, "pregrasp")
        go(grasp, NR, G_OPEN, "approach")
        if a.probe:
            import cv2, pyrealsense2 as rs
            time.sleep(1.0)
            video_pause(a.cap_dir, video)
            pipe = rs.pipeline(); rc = rs.config(); rc.enable_device(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
            rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            prof = pipe.start(rc); sc = prof.get_device().first_depth_sensor().get_depth_scale(); al = rs.align(rs.stream.color)
            for _ in range(20):
                pipe.wait_for_frames()
            fs = al.process(pipe.wait_for_frames())
            cv2.imwrite(os.path.join(a.cap_dir, "probe_color.png"), np.asanyarray(fs.get_color_frame().get_data()))
            np.save(os.path.join(a.cap_dir, "probe_depth.npy"), np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * sc)
            o = robot.get_observation(); np.save(os.path.join(a.cap_dir, "probe_q.npy"), np.array([o[f"{j}.pos"] for j in J]))
            pipe.stop()
            video_resume(a.cap_dir)
            # left wrist D405: look at the jaws and the block up close
            try:
                pw = rs.pipeline(); rw = rs.config(); rw.enable_device(os.environ.get("GRASP_WRIST_SERIAL", "409122272149"))
                rw.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rw.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pr = pw.start(rw); scw = pr.get_device().first_depth_sensor().get_depth_scale(); alw = rs.align(rs.stream.color)
                for _ in range(20):
                    pw.wait_for_frames()
                fw = alw.process(pw.wait_for_frames())
                cv2.imwrite(os.path.join(a.cap_dir, "probe_wrist_color.png"), np.asanyarray(fw.get_color_frame().get_data()))
                np.save(os.path.join(a.cap_dir, "probe_wrist_depth.npy"), np.asanyarray(fw.get_depth_frame().get_data()).astype(np.float32) * scw)
                pw.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[run] wrist camera failed: {e}", flush=True)
            print("[run] probe: saved the chest and wrist frames at the grasp pose", flush=True)
            go(pre, 10, G_OPEN, "back to pregrasp"); go_route(r_pre_up, 12, G_OPEN, "back to raise")
            done = True
            raise KeyboardInterrupt
        if a.wrist_snap or a.wrist_center:  # what the wrist D405 sees right before closing
            try:
                import cv2, pyrealsense2 as rs
                pw = rs.pipeline(); rw = rs.config(); rw.enable_device(os.environ.get("GRASP_WRIST_SERIAL", "409122272149"))
                rw.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rw.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pr = pw.start(rw); scw = pr.get_device().first_depth_sensor().get_depth_scale(); alw = rs.align(rs.stream.color)
                for _ in range(15):
                    pw.wait_for_frames()
                fw = alw.process(pw.wait_for_frames())
                cv2.imwrite(os.path.join(a.cap_dir, "wrist_at_grasp.jpg"), np.asanyarray(fw.get_color_frame().get_data()))
                np.save(os.path.join(a.cap_dir, "wrist_at_grasp_depth.npy"), (np.asanyarray(fw.get_depth_frame().get_data()).astype(np.float32) * scw).astype(np.float16))
                pw.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[run] wrist snapshot failed: {e}", flush=True)
        Tg_plan = np.array(plan["T_grasp_base"]) if "T_grasp_base" in plan else None

        def q_for(Tg, seed):
            from grasp.plan import T_L7_G
            Mt = Tg @ np.linalg.inv(T_L7_G)
            q, ok, _, _ = ik.solve(pin.SE3(Mt[:3, :3].copy(), Mt[:3, 3].copy()), np.deg2rad(seed), position_only=False)
            return list(np.rad2deg(q)) if ok and ik.within_limits(q) else None
        if a.wrist_center and ARM == "left" and Tg_plan is not None:
            try:
                import cv2
                from grasp.wrist_center import measure_edges
                colw = cv2.imread(os.path.join(a.cap_dir, "wrist_at_grasp.jpg"))
                depw = np.load(os.path.join(a.cap_dir, "wrist_at_grasp_depth.npy")).astype(np.float32)
                refw = ref_o if ref_o is not None else None
                m = measure_edges(colw, refw, depw, debug=True) if (colw is not None and refw is not None) else {"ok": False, "why": "no wrist frames"}
                if "debug" in m:
                    cv2.imwrite(os.path.join(a.cap_dir, "wrist_center.jpg"), m.pop("debug"))
                json.dump({k: v for k, v in m.items()}, open(os.path.join(a.cap_dir, "wrist_center.json"), "w"), default=float)
                if not m.get("ok"):
                    print(f"[run] wrist centre: not measured ({m.get('why')}); closing where planned", flush=True)
                elif abs(m["dc_m"]) < 0.003 or abs(m["dc_m"]) > 0.035 or m.get("outside_gap"):
                    print(f"[run] wrist centre: bar {m['dc_m']*1000:+.0f} mm from the jaw centre"
                          + (" (outside the jaws / too far: not shifting)" if abs(m["dc_m"]) >= 0.003 else " (ok)"), flush=True)
                else:
                    Xg, Zg = Tg_plan[:3, 0], Tg_plan[:3, 2]
                    Tg_up = Tg_plan.copy(); Tg_up[:3, 3] -= 0.015 * Zg
                    qgu = q_for(Tg_up, grasp); qu = qc = None
                    # 18:12: the full shift needed J2 / J3 past their limits (the grasp sat at J2 8 / J3 87): take the
                    # largest part of it that the arm can reach
                    for frac in (1.0, 0.75, 0.5):
                        Tc = Tg_plan.copy(); Tc[:3, 3] += frac * m["dc_m"] * Xg
                        Tu = Tc.copy(); Tu[:3, 3] -= 0.015 * Zg
                        qu, qc = q_for(Tu, grasp), q_for(Tc, grasp)
                        if qu is not None and qc is not None and qgu is not None:
                            if frac < 1.0:
                                print(f"[run] wrist centre: only {frac*100:.0f} % of the shift is reachable", flush=True)
                            break
                    if None in (qgu, qu, qc):
                        print(f"[run] wrist centre: bar {m['dc_m']*1000:+.0f} mm, but IK failed for the shifted grasp; closing where planned", flush=True)
                    else:
                        check(grasp, qgu, "wrist shift: up", allow_low=True); check(qgu, qu, "wrist shift: across", allow_low=True)
                        check(qu, qc, "wrist shift: down", allow_low=True); check(qc, lift, "wrist shift: lift")
                        print(f"[run] wrist centre: bar {m['dc_m']*1000:+.0f} mm toward the {'right' if m['dc_m'] > 0 else 'left'} jaw -> shifting the hand", flush=True)  # noqa
                        go(qgu, NR, G_OPEN, "wrist shift: up 15 mm"); go(qu, NR, G_OPEN, "wrist shift: across")
                        go(qc, NR, G_OPEN, "wrist shift: down")
                        grasp = qc; Tg_plan = Tc
            except Blocked as e:
                print(f"[run] wrist centre: shifted path not clear ({e}); closing where planned", flush=True)
            except Exception as e:  # noqa: BLE001  never stop the trial over the measurement
                print(f"[run] wrist centre failed ({type(e).__name__}: {e}); closing where planned", flush=True)
        q_close = grasp
        if a.close_press > 0 and Tg_plan is not None:
            Tp = Tg_plan.copy(); Tp[:3, 3] += a.close_press * Tg_plan[:3, 2]
            qp = q_for(Tp, grasp)
            if qp is not None:
                q_close = qp
                print(f"[run] closing while pressing {a.close_press*1000:.0f} mm deeper", flush=True)
        go(q_close, max(NR, 15), G_CLOSE, "close gripper", correct=False); time.sleep(a.close_wait)
        holding = True  # 離すまで: 途中で止めたら元の場所に下ろしてから戻る（finally）
        go(lift, NR, G_CLOSE, "lift"); time.sleep(2.0); rec("hold")
        # Held check after the lift (2026-11-21 Sota): jaws that closed on nothing end near the closed command
        # (iter_051: -1.0 deg); anything between them keeps the jaws more open. Not held -> open, go back down and grasp
        # again with the same plan, up to --retries times. Still empty -> skip the drop and go home.
        held = True
        for attempt in range(a.retries + 1):
            g_hold = float(robot.get_observation()["gripper.pos"])
            by_angle = g_hold <= a.held_below
            by_cam = held_by_camera(f"hold_{attempt + 1}") if a.held_check == "camera" else None
            held = by_angle if by_cam is None else by_cam  # the camera decides; the angle only when it can't tell
            print(f"[run] held check {attempt + 1}/{a.retries + 1}: gripper {g_hold:+.1f} deg ("
                  + ("held" if by_angle else "empty") + f" by angle), camera: "
                  + ({None: "n/a", True: "held", False: "empty"}[by_cam]) + " -> " + ("HELD" if held else "EMPTY"), flush=True)
            if held or attempt == a.retries:
                break
            # 2026-11-22: open only after going back down to the grasp pose. A bar held by a thin edge read as "empty"
            # (left arm, -1.1/-1.3 deg vs -1.0 empty) and opening at the lift height dropped it onto the floor beside the robot
            go(grasp, NR, G_CLOSE, f"back down before opening ({attempt + 1})")
            go(grasp, NR, G_OPEN, f"open to retry ({attempt + 1})", correct=False)
            holding = False
            go(pre, NR, G_OPEN, f"back to pregrasp ({attempt + 1})")
            go(grasp, NR, G_OPEN, f"approach again ({attempt + 1})")
            go(grasp, max(NR, 15), G_CLOSE, f"close gripper again ({attempt + 1})", correct=False); time.sleep(a.close_wait)
            holding = True
            go(lift, NR, G_CLOSE, f"lift again ({attempt + 1})"); time.sleep(2.0); rec("hold")
        if not held:
            print("[run] nothing held after the retries: skipping the drop, going home", flush=True)
            go(grasp, NR, G_CLOSE, "back down before opening (empty)")  # if something is held after all, it lands where it was
            go(grasp, NR, G_OPEN, "open (empty)", correct=False)
            holding = False
            go(pre, NR, G_OPEN, "back to pregrasp (empty)")
            if CH_RET_EMPTY is not None:
                ret_chain = CH_RET_EMPTY  # the finally block moves pregrasp → raise → start → home without stopping
            else:
                go_route(r_pre_up, TR, G_OPEN, "back to raise")
        elif drop is not None:  # throw aside: carry it beside the left shoulder and open there (2026-11-21 Sota)
            if CH_HOLD is not None:
                go_chain(CH_HOLD, ["raise the object high", "raise holding", "carry to the drop point"])
            else:
                go(lhigh, NR, G_CLOSE, "raise the object high")  # straight up first, then turn toward home and the shoulder
                go_route(r_lhigh_up, NR, G_CLOSE, "raise holding")
                go_route(r_up_dcarry, NR, G_CLOSE, "carry to the drop point")
            go(drop, NR, G_CLOSE, "down to the drop point")
            go(drop, max(NR, 15), G_OPEN, "drop", correct=False)
            holding = False
            go(dcarry, NR, G_OPEN, "up from the drop point")
            if CH_RET_DROP is not None:
                ret_chain = CH_RET_DROP  # the finally block moves drop carry → raise → start → home without stopping
            else:
                go_route(r_dcarry_up, TR, G_OPEN, "back to raise")
        elif carry is not None:  # self-reset: put it down at a new spot (grasp/tools/plan_place.py), next trial starts there
            if a.avoid:  # 割り込む物を見ながら運び、最後に位置を合わせる
                carry_watch([lift] + r_lift_carry, G_CLOSE, "carry", holding_now=True)
                go(carry, NR, G_CLOSE, "carry to the new spot")
            else:
                go_route(r_lift_carry, NR, G_CLOSE, "carry to the new spot")
            go(place, NR, G_CLOSE, "put down at the new spot")
            go(place, max(NR, 15), G_OPEN, "release", correct=False)
            holding = False
            go(carry, NR, G_OPEN, "up from the new spot")
            if a.avoid:
                carry_watch([carry] + r_carry_up, G_OPEN, "back to raise")
                go(up, TR, G_OPEN, "back to raise")
            else:
                go_route(r_carry_up, TR, G_OPEN, "back to raise")
        else:
            go(grasp, NR, G_CLOSE, "put back down")
            go(grasp, max(NR, 15), G_OPEN, "release", correct=False)
            holding = False
            go(pre, NR, G_OPEN, "back to pregrasp")
            go_route(r_pre_up, TR, G_OPEN, "back to raise")
        done = True
    except PushDone:
        pass
    except TrackingLost as e:
        print(f"[run] STOP: {e}", flush=True)
        TR = min(TR, 10.0)
        try:  # only move back if fresh readings look sane (the arm is where the last command left it, within the limit)
            o = robot.get_observation(); now = [float(o[f"{j}.pos"]) for j in J]
            print("[run] after stop, joints " + " ".join(f"{v:+.1f}" for v in now), flush=True)
        except Exception as e2:  # noqa: BLE001
            print(f"[run] motors not answering ({e2}); disconnecting without moving", flush=True)
            no_return = True
    except SystemExit as e:
        # 非常停止・電源断（ここの MotorLost と grasp/reactive.py の MotorLost の両方）: 戻る動きもしない
        # （指令が届かない・電源が戻った瞬間に跳ぶ）。ほかの SystemExit は今までどおり戻りの処理へ
        if type(e).__name__ != "MotorLost":
            raise
        print(f"{e}。戻る動きはしない。腕を支えて電源を確かめる", flush=True)
        motor_lost = True
    except KeyboardInterrupt:
        TR = min(TR, 20.0)  # an aborted trial may be next to the object: no fast moves on the way back
        print("[run] stopping: returning to the start pose", flush=True)
    finally:
        try:
            if motor_lost:
                raise SystemExit(1)
            if no_return:
                raise SystemExit(2)
            g_end = g_now if g_now > -10 else G_OPEN
            a.max_track_err = 1e9  # on the way back, never raise TrackingLost again
            chained_home = False
            if done and ret_chain is not None:  # --blend: from pregrasp / drop carry straight on to home
                go_chain(ret_chain, ["back to raise", "back to start"] + (["back to home"] if standby is not None else []))
                chained_home = True
            elif done:  # raise にいる: 作っておいた経路で戻る
                go_route(r_up_start, TR, g_end, "back to start")
            else:  # 途中で止めた: 今の位置から経路を作り直す。作れなければ今までどおり直線で戻る
                o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
                if holding:  # 物を持ったまま: 掴んだ場所の上へ戻り、元の場所に下ろして離してから戻る（持った物を机に押し付けない）
                    print("[run] 物を持っている: 元の場所に下ろしてから戻る", flush=True)
                    if watcher is not None:
                        carry_watch([q_now] + route(q_now, lift, "now→lift"), G_CLOSE, "back to the grasp spot", holding_now=True)
                    else:
                        go_route(route(q_now, lift, "now→lift"), TR, G_CLOSE, "back to the grasp spot")
                    go(lift, NR, G_CLOSE, "over the grasp spot")
                    go(grasp, NR, G_CLOSE, "put back down (abort)")
                    go(grasp, max(NR, 15), G_OPEN, "release (abort)", correct=False)
                    go(pre, NR, G_OPEN, "back to pregrasp (abort)")
                    o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
                    g_end = G_OPEN
                try:
                    back = route(q_now, start, "now→start")
                except SystemExit as e:
                    print(f"[run] 戻りの経路を作れない（{e}）。直線で戻る", flush=True)
                    back = [list(start)]
                go_route(back, TR, g_end, "back to start")
            if standby is not None and not chained_home:
                go_route(r_sb_home, TR, g_end, "back to home")
            if home is not None and a.hold_home:
                print("[run] holding at home with torque ON (release: grasp/tools/release_arm.py)", flush=True)
            elif home is not None and not at_home:  # the home pose drops when limp (11/20: J7 swung 63 deg): back to the resting pose first
                go_route(r_home_up0, TR, g_end, "back to raise over rest")
                go(rest, min(TR, 20.0), g_end, "back to rest")
        finally:
            if watcher is not None:
                watcher.stop()
            release_and_disconnect(robot, grav, skip=motor_lost or no_return); log.close()
            if motor_lost or no_return:
                print("[run] モーターの返事が途切れたので、戻る動きはしていない（腕は止まった所）", flush=True)
            else:
                print("[run] back at start, " + ("holding (torque on)" if (a.hold_home and home is not None) else "torque off"), flush=True)
            video_stop(video)
            s3_upload(a.cap_dir)
    return 0


def video_start(cap_dir, avoid=False):
    """2026-11-23 (Sota: 試行は動画も含めてクラウドに): record the chest camera during the motion in a separate process
    (grasp/tools/record_video.py -> <cap_dir>/video_chest.mp4 + video_chest_ts.csv). Off with GRASP_VIDEO=0, and with
    --avoid (its Watcher holds the chest camera the whole time). Never fails the run."""
    import subprocess
    if os.environ.get("GRASP_VIDEO", "1") == "0":
        print("[run] video off (GRASP_VIDEO=0)", flush=True); return None
    if avoid:
        print("[run] video off: --avoid uses the chest camera the whole time", flush=True); return None
    for f in (".video_pause", ".video_paused"):
        if os.path.exists(os.path.join(cap_dir, f)):
            os.remove(os.path.join(cap_dir, f))
    try:
        log = open(os.path.join(cap_dir, "video.log"), "a")
        p = subprocess.Popen([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "record_video.py"),
                              cap_dir], stdout=log, stderr=log)
        print(f"[run] recording the chest camera to {os.path.join(cap_dir, 'video_chest.mp4')}", flush=True)
        return p
    except Exception as e:  # noqa: BLE001
        print(f"[run] video not started: {e}", flush=True); return None


def video_pause(cap_dir, p, wait_s=5.0):
    """Release the chest camera for another reader (run_grasp --probe). Waits until the recorder says it let go."""
    if p is None or p.poll() is not None:
        return
    open(os.path.join(cap_dir, ".video_pause"), "w").close()
    t0 = time.time()
    while not os.path.exists(os.path.join(cap_dir, ".video_paused")) and time.time() - t0 < wait_s and p.poll() is None:
        time.sleep(0.05)


def video_resume(cap_dir):
    f = os.path.join(cap_dir, ".video_pause")
    if os.path.exists(f):
        os.remove(f)


def video_stop(p, wait_s=10.0):
    """Stop the recorder and wait until the mp4 is closed (before the upload, and before the next capture opens the camera)."""
    if p is None:
        return
    import signal
    import subprocess
    try:
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
            p.wait(timeout=wait_s)
    except subprocess.TimeoutExpired:
        p.kill(); p.wait()
        print("[run] video recorder did not stop in time: killed (the mp4 may be cut)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[run] video stop: {e}", flush=True)


def s3_upload(*paths):
    """2026-11-23 (Sota: 実行の結果は毎回 S3 に上げる): upload this run's folder in the background (grasp/tools/s3_sync.sh;
    never blocks or fails the run; without valid credentials the folder is queued in ~/grasp_runs/S3_PENDING)."""
    import subprocess
    sh = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tools", "s3_sync.sh")
    try:
        log = open(os.path.expanduser("~/grasp_runs/s3_sync.out"), "a")
        subprocess.Popen(["bash", sh, *[os.path.abspath(p) for p in paths]], stdout=log, stderr=log, start_new_session=True)
        print(f"[run] uploading {', '.join(os.path.basename(os.path.normpath(p)) for p in paths)} to S3 in the background "
              "(grasp/tools/s3_sync.sh)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[run] S3 upload not started: {e}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
