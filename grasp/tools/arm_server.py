#!/usr/bin/env python3
"""腕 1 本につないだまま常駐し、頼まれるたびに run_grasp.py の 1 回をこのプロセスで回す。**腕が動く。**

2026-11-25・作業者: Shreyan。選別を速くするため（run_pipeline.sh --go が使う）:
- 回ごとの起動（lerobot の読み込み・CAN の接続 約 1.3 秒）を払わない
- 回の間に休みの姿勢へ下ろして脱力しない。run_grasp.py --hover --keep-torque で、上げた姿勢でモーターを入れたまま待つ
- 左右で 1 つずつ立てる（run_grasp.py は腕を module の変数と環境変数で持つので、1 つのプロセスで 2 本を同時には回せない）。
  両腕を同時に動かすかは run_pipeline.sh が grasp/dual.py で決める

lerobot の venv で、置き場の env を読んでから立てる（run_pipeline.sh --go が立て、終わりに止める）:
    ~/projects/lerobot/.venv/bin/python grasp/tools/arm_server.py --arm left     # tcp://*:5562
    ~/projects/lerobot/.venv/bin/python grasp/tools/arm_server.py --arm right    # tcp://*:5563
要求（json）:
    {"op": "ping"} → {"ok": true}
    {"op": "state"} → {"q": [関節 1〜7 の度], "gripper", "holding": モーターを入れて支えているか, "lost": モーターの返事が途切れたか}
    {"op": "run", "argv": [run_grasp.py の引数], "env": {...}} → run_grasp.main(argv + --go --hover --keep-torque) → {"rc", "stdout"}
    {"op": "park"} → 休みの姿勢（つないだときの角度）へ関節の直線で戻し（cuRobo で当たらないと確かめてから・20 deg/s）、
        ゆっくり脱力する（grasp/soft_release.py）。当たるなら動かさずに支えたまま {"error"} を返す
    {"op": "quit"} → park して接続を切り、終わる
Ctrl-C（SIGINT）: 回の途中なら run_grasp.py の止め方（元の姿勢へ戻る）に任せる。待っている間なら park して終わる。
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
J = [f"joint_{i}" for i in range(1, 8)]
PARK_DPS = 20.0  # 休みの姿勢へ戻す速さ [deg/s]（机の近くへ下りるので、物の近くの決まりと同じ上限）


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--bind", default=None, help="既定: 左 tcp://*:5562・右 tcp://*:5563")
    a = ap.parse_args()
    side = a.arm
    bind = a.bind or f"tcp://*:{5562 if side == 'left' else 5563}"
    os.environ["GRASP_ARM"] = side  # run_grasp.py・grasp/curobo/client.py が読む前に
    if side == "left":
        port, rid = os.environ.get("GRASP_CAN_PORT", "can_fol_l"), os.environ.get("GRASP_ROBOT_ID", "follower_left")
    else:  # 右は既定の値を持たない（左の CAN・較正で右腕を動かさない・run_grasp.py と同じ）
        port, rid = os.environ.get("GRASP_CAN_PORT_RIGHT"), os.environ.get("GRASP_ROBOT_ID_RIGHT")
        if not port or not rid:
            sys.exit("[arm] 右腕の GRASP_CAN_PORT_RIGHT・GRASP_ROBOT_ID_RIGHT が置き場の env に無い")
    from grasp import run_grasp
    from grasp.gravity import gravity_torque_fn
    from grasp.motion import smooth_move_to_position
    from grasp.soft_release import soft_release
    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    robot = OpenArmFollower(OpenArmFollowerConfig(port=port, side=side, id=rid, align_on_connect=False, cameras={},
                                                  disable_torque_on_disconnect=True))
    robot.connect()
    st = {"holding": False, "lost": False}
    # モーターを入れているか（holding）は、実際に入れた・切った呼び出しで覚える（2026-11-25: 引数の誤りで run_grasp.py が
    # モーターを入れずに終わったのに「支えている」とみなし、終わりの park で脱力した腕を休みの姿勢の角度へ動かした）
    _en, _dis = robot.bus.enable_torque, robot.bus.disable_torque

    def _enable(*args, **kw):
        r = _en(*args, **kw)
        st["holding"] = True
        return r

    def _disable(*args, **kw):
        r = _dis(*args, **kw)
        st["holding"] = False
        return r
    robot.bus.enable_torque, robot.bus.disable_torque = _enable, _disable
    robot.bus.disable_torque()  # 最初の回まで休みの姿勢で脱力しておく（grasp_daemon.py と同じ）
    grav = gravity_torque_fn(side)
    obs = robot.get_observation()
    rest = [float(obs[f"{j}.pos"]) for j in J]
    print(f"[arm:{side}] CAN {port}・較正 {rid} につないだ。休みの姿勢 " + ",".join(f"{v:.1f}" for v in rest), flush=True)

    def q_now():
        o = robot.get_observation()
        return [float(o[f"{j}.pos"]) for j in J], float(o["gripper.pos"])

    def rest_up(dz: float = 0.08):
        """休みの姿勢の手先を真上へ dz 上げた姿勢 [deg]（park の経由点）。解けなければ None。"""
        import numpy as np
        import pinocchio as pin
        from ik.openarm_left_ik import OpenArmIK
        ik = OpenArmIK(side=side)
        q0 = ik.clamp(np.deg2rad(rest)); M = ik.fk(q0)
        q, ok, _, _ = ik.solve(pin.SE3(M.rotation, M.translation + np.array([0, 0, dz])), q0, position_only=False, max_iter=300)
        return list(np.rad2deg(q)) if ok and ik.within_limits(q) else None

    def park() -> dict:
        """休みの姿勢へ戻して脱力する。支えていなければ脱力だけ。当たるなら動かさない。"""
        if st["lost"]:
            return {"error": "モーターの返事が途切れている。動かさない（腕を支えて電源を確かめる）"}
        if not st["holding"]:
            robot.bus.disable_torque()
            return {"parked": True, "moved": False}
        q, g = q_now()
        if max(abs(x - y) for x, y in zip(q, rest)) > 1.0:
            client = run_grasp.CUROBO
            path = [rest]
            if client is not None:  # run_grasp.py の check と同じ考え方: 端がもともと机に近いときは、低い方の端 − 1 cm まで許す
                client.set_arm(side)

                def blocked(seq):
                    d_end = min(client.check_path([p], n=1)[0]["table/arm"][0] for p in seq)
                    return client.violations(seq, table_margin=min(client.TABLE_MARGIN, d_end - 0.01))
                bad = blocked([q, rest])
                if bad:  # 置いた所の上で止まっている（--hover）ときなど: 休みの姿勢の 8 cm 上を経由する
                    mid = rest_up()
                    if mid is not None and not blocked([q, mid, rest]):
                        path = [mid, rest]
                        bad = None
                if bad:
                    return {"error": f"休みの姿勢へ戻る道が当たる（{client.describe(bad)}）。動かさずに支えたまま。"
                                     "grasp/tools/release_arm.py で下ろす"}
            for tgt in path:
                goal = {j: float(tgt[i]) for i, j in enumerate(J)}
                goal["gripper"] = g
                smooth_move_to_position(robot.bus, goal, PARK_DPS, torque_ff_fn=grav)
        soft_release(robot, grav)  # 中で bus.disable_torque を呼ぶので holding は False になる
        st["holding"] = False
        return {"parked": True, "moved": True}

    import zmq
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(bind)
    print(f"[arm:{side}] {bind} で待つ", flush=True)
    try:
        while True:
            req = sock.recv_json()
            try:
                op = req.get("op")
                if op == "ping":
                    rep = {"ok": True}
                elif op == "state":
                    q, g = q_now()
                    rep = {"q": q, "gripper": g, "holding": st["holding"], "lost": st["lost"]}
                elif op == "run":
                    if st["lost"]:
                        raise RuntimeError("モーターの返事が途切れている。動かさない")
                    for k, v in (req.get("env") or {}).items():
                        os.environ[str(k)] = str(v)
                    argv = [str(x) for x in req["argv"]] + ["--go", "--hover", "--keep-torque"]
                    buf = io.StringIO()
                    t0 = time.time()
                    with contextlib.redirect_stdout(buf):
                        try:
                            rc = run_grasp.main(argv, robot=robot) or 0
                        except SystemExit as e:
                            rc = e.code if isinstance(e.code, int) else 1
                            print(e)
                            if type(e).__name__ == "MotorLost":
                                st["lost"] = True
                        except KeyboardInterrupt:
                            rc = 130
                            print("[arm] Ctrl-C")
                    if st["lost"]:
                        st["holding"] = False  # 返事の無いモーターには何も送らない
                    rep = {"rc": rc, "stdout": buf.getvalue()}
                    print(f"[arm:{side}] run rc={rc}（{time.time() - t0:.1f} 秒）", flush=True)
                elif op == "park":
                    rep = park()
                elif op == "quit":
                    rep = park()
                    sock.send_json(rep)
                    break
                else:
                    rep = {"error": f"知らない op: {op!r}"}
            except Exception as e:  # noqa: BLE001  1 つの失敗で口を止めない
                rep = {"error": f"{type(e).__name__}: {e}"}
                print(f"[arm:{side}] 失敗: {rep['error']}", flush=True)
            sock.send_json(rep, default=str)
    except KeyboardInterrupt:
        print(f"[arm:{side}] Ctrl-C: 休みの姿勢へ戻して終わる", flush=True)
        try:
            print(f"[arm:{side}] {park()}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[arm:{side}] park できない: {e}", flush=True)
    finally:
        if st["holding"] and not st["lost"]:
            print(f"[arm:{side}] 支えたまま接続を切らない（腕が落ちる）。release_arm.py で下ろす", flush=True)
        else:
            robot.disconnect()
        print(f"[arm:{side}] 終わり", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
