#!/usr/bin/env python3
"""止まった姿勢の PD の誤差を、重力のずれと摩擦・ガタに分けて測る。**--go で腕が動く**（無ければ確かめだけ）。
2026-11-23 Munenori（Kota の PC の Claude Code から）。

置くときのずれを分ける試し（grasp/tools/pd_check.py の続き）。run_grasp.py と同じ柔らかい PD（kp 50/10）と
重力のフィードフォワード（OpenArmGravityModel・scale 倍）で、記録の高い姿勢（持ち上げ・運ぶ・掴む手前）へ
**補正の送り直しをせずに** + 側と − 側から近づき、止めて関節角を読む。
  - 重力のずれ（どちらから来ても同じ向き）= (e+ + e-) / 2
  - 摩擦・ガタ（来た向きで変わる）       = (e+ - e-) / 2
e = 読み − 指令 [deg]。kp を掛けて Nm にも直し、重力モデルの何倍が足りないかを関節ごとに当てはめる。

道筋: 今の姿勢（休み）→ 上げた姿勢 →（姿勢ごとに）+側 → 姿勢 → −側 → 姿勢 → 上げた姿勢 → … → 休み → 脱力。
動かす前に全部の区間を床・keep-out の箱・cuRobo（GRASP_CUROBO(_RIGHT) があれば）で確かめ、通らない姿勢は外す。
Ctrl-C で止めると、上げた姿勢を通って休みへ戻ってから脱力する。

    set -a; . grasp/config/sites/kitakyushu-collect.env; set +a
    PYTHONPATH=. ~/projects/lerobot/.venv/bin/python grasp/tools/hold_test.py --arm right            # 確かめだけ
    PYTHONPATH=. ~/projects/lerobot/.venv/bin/python grasp/tools/hold_test.py --arm right --go       # 動かす
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)

URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
KP = np.array([50, 50, 50, 50, 10, 10, 10.0])  # lerobot の OPENARM_ALIGN_KP（soft_move_to_position の既定）
TCP_LEN = 0.2145


def pick_poses(side: str, n: int, runs_glob: str) -> List[List[float]]:
    """記録の plan.json から、その腕の高い姿勢（持ち上げ・運ぶ・掴む手前）を集め、互いに遠いものを n 個選ぶ。"""
    qs = []
    for p in sorted(glob.glob(os.path.join(runs_glob, "plan.json"))):
        try:
            plan = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        if plan.get("arm", "left") != side:
            continue
        for k in ("q_lift_deg", "q_carry_deg", "q_pre_deg"):
            if plan.get(k) is not None:
                qs.append(list(map(float, plan[k])))
    if not qs:
        return []
    Q = np.array(qs)
    chosen = [int(np.argmin(np.linalg.norm(Q - Q.mean(0), axis=1)))]  # 真ん中に近いものから、遠い順に足す
    while len(chosen) < min(n, len(Q)):
        d = np.min(np.linalg.norm(Q[:, None] - Q[chosen][None], axis=2), axis=1)
        chosen.append(int(np.argmax(d)))
    return [qs[i] for i in chosen]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--go", action="store_true", help="腕を動かす（無ければ道筋を確かめるだけ）")
    ap.add_argument("--poses", type=int, default=4, help="測る姿勢の数")
    ap.add_argument("--runs", default=os.path.join(REPO, "grasp", "runs", "cap_resident_*"), help="姿勢を拾う記録（glob）")
    ap.add_argument("--offset", type=float, default=4.0, help="+側・−側の離れ [deg]（全部の関節を同じ向きにずらす）")
    ap.add_argument("--scales", default="1.0", help="重力のフィードフォワードの倍率（カンマ区切り・例 1.0,0.7）")
    ap.add_argument("--hold", type=float, default=2.0, help="止めてから読むまでの秒")
    ap.add_argument("--speed", type=float, default=8.0, help="動きの速さ [deg/s]")
    ap.add_argument("--grav-urdf", default=os.path.join(REPO, "urdf", "openarm_v10_bimanual_grav.urdf"), help="重力のフィードフォワードに使う URDF（既定は手と爪の質量入りの "
                    "grasp/tools/make_grav_urdf.py が作る物。手 0 kg の元の物と比べるなら urdf/openarm_v10_bimanual.urdf）。FK・道筋は元の URDF のまま")
    ap.add_argument("--start", default=None, help="確かめだけのときの休みの姿勢（無ければ GRASP_REST(_RIGHT)）")
    a = ap.parse_args(argv)
    side = a.arm
    scales = [float(v) for v in a.scales.split(",")]
    sfx = "_RIGHT" if side == "right" else ""
    if "GRASP_FLOOR_Z" not in os.environ:
        sys.exit("[hold] GRASP_FLOOR_Z が無い。置き場の env を読んでから打つ（床の確かめをしないで動かさない）")
    floor_z = float(os.environ["GRASP_FLOOR_Z"])
    keepout = [np.array([float(v) for v in b.split(",")]) for b in os.environ.get("GRASP_KEEPOUT_BOXES", "").split(";") if b.strip()]
    os.environ["GRASP_ARM"] = side

    import pinocchio as pin
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(URDF, side=side)
    curobo = None
    if os.environ.get("GRASP_CUROBO" + sfx):
        from grasp.curobo import client as curobo
        curobo.set_arm(side)
    elif a.go:
        sys.exit(f"[hold] GRASP_CUROBO{sfx} が無い。腕と周りの物の当たりを確かめずに動かさない")

    def fk(qd):
        return ik.fk(np.deg2rad(qd))

    def tcp(qd):
        M = fk(qd)
        return M.translation + M.rotation @ [0, 0, TCP_LEN]

    def seg_ok(q0, q1, name) -> Optional[str]:
        """区間を確かめる。通れば None、だめなら理由。床の線は run_grasp.py と同じ考え方（低い端だけ緩める）。"""
        tmin = min(floor_z + 0.02, tcp(q0)[2] - 0.01, tcp(q1)[2] - 0.01)
        lmin = min(floor_z + 0.05, fk(q0).translation[2] - 0.01, fk(q1).translation[2] - 0.01)
        for s in np.linspace(0, 1, 20):
            q = np.asarray(q0) + (np.asarray(q1) - np.asarray(q0)) * s
            M = fk(q); t = M.translation + M.rotation @ [0, 0, TCP_LEN]
            if M.translation[2] < lmin or t[2] < tmin:
                return f"{name}: s={s:.2f} で床に近い（link7 z={M.translation[2]:.3f}・TCP z={t[2]:.3f}）"
            for b in keepout:
                if b[0] <= t[0] <= b[1] and b[2] <= t[1] <= b[3] and b[4] <= t[2] <= b[5]:
                    return f"{name}: s={s:.2f} で keep-out の箱に入る"
        if curobo is not None:
            d_end = min(curobo.check_path([q], n=1)[0]["table/arm"][0] for q in (q0, q1))
            bad = curobo.violations([list(q0), list(q1)], names=[name], table_margin=min(curobo.TABLE_MARGIN, d_end - 0.01))
            if bad:
                return f"cuRobo: {curobo.describe(bad)}"
        return None

    def raised(qd):
        """休みの姿勢から TCP を真上へ 8/5/3 cm 上げた姿勢（run_grasp.py の raised_some と同じ）。"""
        for dz in (0.08, 0.05, 0.03):
            M = fk(qd)
            q, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, dz])), np.deg2rad(qd),
                                   position_only=False, max_iter=300)
            if ok and ik.within_limits(q):
                return list(np.rad2deg(q))
        sys.exit("[hold] 休みの姿勢から持ち上げる姿勢が解けない")

    # ---- 道筋を作って確かめる ----
    poses = pick_poses(side, a.poses * 3, a.runs)
    if not poses:
        sys.exit(f"[hold] {a.runs} に {side} の腕の姿勢が無い")
    robot = None
    if a.go:
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
        port = os.environ.get("GRASP_CAN_PORT" + sfx) or sys.exit(f"[hold] GRASP_CAN_PORT{sfx} が無い")
        rid = os.environ.get("GRASP_ROBOT_ID" + sfx) or sys.exit(f"[hold] GRASP_ROBOT_ID{sfx} が無い")
        print(f"[hold] {side} の腕の CAN {port}・較正 {rid}", flush=True)
        robot = OpenArmFollower(OpenArmFollowerConfig(port=port, side=side, id=rid, align_on_connect=False, cameras={},
                                                      disable_torque_on_disconnect=True))
        robot.connect()
        o = robot.get_observation()
        start = [float(o[f"{j}.pos"]) for j in J]; grip = float(o["gripper.pos"])
    else:
        s = a.start or os.environ.get("GRASP_REST" + sfx)
        if not s:
            sys.exit(f"[hold] 確かめだけでも休みの姿勢が要る（--start か GRASP_REST{sfx}）")
        start = [float(v) for v in s.split(",")]; grip = 0.0
    print("[hold] 休みの姿勢 " + ",".join(f"{v:.1f}" for v in start), flush=True)
    up = raised(start)
    plan_ok = [("start→up", start, up)]
    why = seg_ok(start, up, "start→up")
    if why:
        sys.exit(f"[hold] 休みから上げる区間が通らない: {why}")
    targets = []  # (姿勢の番号, P, A+, A-)
    lo, hi = np.rad2deg(ik.lo), np.rad2deg(ik.hi)
    for P in poses:
        if len(targets) >= a.poses:
            break
        P = np.asarray(P)
        Ap = np.clip(P + a.offset, lo + 1, hi - 1); Am = np.clip(P - a.offset, lo + 1, hi - 1)
        chain = [("up→A+", up, Ap), ("A+→P", Ap, P), ("P→A-", P, Am), ("A-→P", Am, P), ("P→up", P, up)]
        bad = [w for w in (seg_ok(q0, q1, f"姿勢{len(targets) + 1} {nm}") for nm, q0, q1 in chain) if w]
        if bad:
            print(f"[hold] 姿勢を外す {np.round(P, 1).tolist()}: {bad[0]}", flush=True)
            continue
        targets.append((len(targets) + 1, P, Ap, Am))
        t = tcp(P)
        print(f"[hold] 姿勢{len(targets)} {np.round(P, 1).tolist()}  TCP {np.round(t, 3).tolist()}（机から {t[2] - floor_z:+.2f} m）", flush=True)
    if not targets:
        sys.exit("[hold] 通る姿勢が無い")
    n_moves = len(targets) * len(scales) * 4
    print(f"[hold] 道筋 ok: 姿勢 {len(targets)} 個 × 重力の倍率 {scales} × 2 方向（止める回数 {n_moves // 2}）", flush=True)
    if not a.go:
        print("[hold] 確かめだけ（--go を付けると動かす）")
        return 0

    # ---- 動かす ----
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
    gm = OpenArmGravityModel(a.grav_urdf, side, 1.0)
    print(f"[hold] 重力のモデル: {a.grav_urdf}", flush=True)
    out_dir = os.path.join(REPO, "grasp", "runs", time.strftime("hold_%Y%m%d_%H%M%S"))
    os.makedirs(out_dir, exist_ok=True)
    f = open(os.path.join(out_dir, "hold.csv"), "w", newline=""); w = csv.writer(f)
    w.writerow(["pose", "dir", "scale"] + [f"goal_{j}" for j in J] + [f"q_{j}" for j in J] + [f"std_{j}" for j in J]
               + [f"g_{j}" for j in J])

    def move(q, scale, speed):
        goal = {j: float(q[i]) for i, j in enumerate(J)}; goal["gripper"] = grip
        ff = (lambda p: {k: scale * v for k, v in gm.torque(p).items()})
        soft_move_to_position(robot.bus, goal, torque_ff_fn=ff, max_speed_dps=speed)
        age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
        if age > 0.3:
            raise SystemExit(f"[hold] モーターの返事が {age:.1f} 秒来ない（非常停止・電源・通信）。止める")

    def read_mean(n=10):
        Q = []
        for _ in range(n):
            o = robot.get_observation(); Q.append([float(o[f"{j}.pos"]) for j in J]); time.sleep(0.05)
        return np.mean(Q, 0), np.std(Q, 0)

    res: Dict = {}
    try:
        move(up, 1.0, a.speed)
        for scale in scales:
            for k, P, Ap, Am in targets:
                for d, A in (("+", Ap), ("-", Am)):
                    move(A, scale, a.speed)
                    move(P, scale, a.speed / 2)  # ゆっくり寄せて、行き過ぎで向きが変わらないようにする
                    time.sleep(a.hold)
                    q, sd = read_mean()
                    g = gm.torque({j: float(q[i]) for i, j in enumerate(J)})
                    w.writerow([k, d, scale] + list(np.round(P, 3)) + list(np.round(q, 3)) + list(np.round(sd, 3))
                               + [round(g[j], 4) for j in J]); f.flush()
                    e = q - P
                    res[(scale, k, d)] = (P, q, np.array([g[j] for j in J]))
                    dt = (tcp(q) - tcp(P)) * 1000
                    print(f"[hold] 倍率 {scale} 姿勢{k} {d}側から: 誤差 " + " ".join(f"{v:+.2f}" for v in e)
                          + f" deg  TCP {np.linalg.norm(dt):.1f} mm", flush=True)
                move(up, scale, a.speed)
    except KeyboardInterrupt:
        print("[hold] 止める: 上げた姿勢を通って休みへ戻る", flush=True)
    finally:
        try:
            o = robot.get_observation(); now = [float(o[f"{j}.pos"]) for j in J]
            if np.max(np.abs(np.asarray(now) - np.asarray(up))) > 1.0:
                move(up, 1.0, min(a.speed, 8.0))
            move(start, 1.0, min(a.speed, 8.0))
        finally:
            robot.disconnect(); f.close()
            print(f"[hold] 休みに戻して脱力した。記録 {out_dir}/hold.csv", flush=True)

    # ---- まとめ ----
    for scale in scales:
        G, Fr, Tg = [], [], []
        for k, P, _, _ in targets:
            if (scale, k, "+") not in res or (scale, k, "-") not in res:
                continue
            ep = res[(scale, k, "+")][1] - P; em = res[(scale, k, "-")][1] - P
            G.append((ep + em) / 2); Fr.append((ep - em) / 2); Tg.append((res[(scale, k, "+")][2] + res[(scale, k, "-")][2]) / 2)
        if not G:
            continue
        G, Fr, Tg = np.array(G), np.array(Fr), np.array(Tg)
        print(f"\n[hold] 倍率 {scale}（姿勢 {len(G)} 個）")
        print("  重力などの片寄り (e+ + e-)/2 の平均 [deg]: " + " ".join(f"J{j + 1} {G[:, j].mean():+.2f}" for j in range(7)))
        print("  摩擦・ガタ |e+ - e-|/2 の平均 [deg]   : " + " ".join(f"J{j + 1} {np.abs(Fr[:, j]).mean():.2f}" for j in range(7)))
        print("  摩擦・ガタをトルクにすると [Nm]       : " + " ".join(f"J{j + 1} {np.abs(Fr[:, j]).mean() * KP[j] / 57.3:.2f}" for j in range(7)))
        # 片寄りのトルク −kp·e を、モデルの重力の何倍が足りないかで当てはめる（フィードフォワードは scale 倍）
        Y = -KP * np.deg2rad(G)
        print("  実際の重さ ÷ モデル（関節ごと）       : " + " ".join(
            f"J{j + 1} {scale + (Tg[:, j] @ Y[:, j]) / max(Tg[:, j] @ Tg[:, j], 1e-9):.2f}" for j in range(7)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
