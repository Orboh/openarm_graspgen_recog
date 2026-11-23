#!/usr/bin/env python3
"""常駐して、合図（Enter）ごとに 1 回掴んで置く。**腕が動く。** 人がロボットの前にいて、非常停止を手に持っているときだけ使う。

2026-11-23・作業者: Munenori（Kota の PC の Claude Code から）。「計画して」から腕が動き出すまでを 1 秒以内にするため、
毎回払っていた起動（Python・lerobot の読み込み・モーターへの接続）と、人と Claude のやりとりを間から外す。

1 回の流れ:
  合図 → 腕の今の角度を読む → 計画の常駐（resident_plan.py --serve）に「撮る」を頼む（約 0.15 秒）
  → run_grasp.py --early: 休みの姿勢から真上へ上げて爪を開き始める（ここまで約 0.3 秒）。その間に裏で「計画する」を頼み、
    計画が来たら近づく軌道と当たりを確かめる → 掴んで置いて、休みの姿勢で脱力する
  計画が立たない・確かめを通らないときは、物の近くへは行かずに休みの姿勢へ戻って脱力する（tests/test_early_start.py）

lerobot の venv で、置き場の env を読んでから打つ。先に cuRobo のサーバーと計画の常駐を立てておく:
    ~/projects/lerobot/.venv/bin/python grasp/tools/grasp_daemon.py \\
        --run-args "--wrist-servo --transit-dps 75 --near-dps 12 --gripper-dps 45 --servo-dps 12 --servo-settle 0.25 --servo-step 0.06 --hold-s 0.5"
    # Enter で 1 回・q で終わる。--trigger-dir DIR なら DIR/go のファイルを合図にする（DIR/quit で終わる）
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
J = [f"joint_{i}" for i in range(1, 8)]


def request(addr: str, req: dict, timeout_s: float) -> dict:
    """計画の常駐に 1 つ頼む（頼みごとごとに口を作る。裏のスレッドからも呼ぶため）。返事が来なければ {"error"}。"""
    import zmq
    s = zmq.Context.instance().socket(zmq.REQ)
    s.setsockopt(zmq.LINGER, 0)
    s.setsockopt(zmq.RCVTIMEO, int(timeout_s * 1000))
    s.connect(addr)
    try:
        s.send_json(req)
        return s.recv_json()
    except zmq.Again:
        return {"error": f"計画の常駐（{addr}）から {timeout_s:.0f} 秒返事が無い"}
    finally:
        s.close()


def open_right_reader():
    """右腕の CAN を、トルクを入れずに状態を聞くだけで開く（grasp/tools/read_arm.py と同じ読み方）。開けなければ None。"""
    port, rid = os.environ.get("GRASP_CAN_PORT_RIGHT"), os.environ.get("GRASP_ROBOT_ID_RIGHT")
    if not port or not rid:
        return None
    try:
        import can
        cal = json.load(open(os.path.expanduser(f"~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/{rid}.json")))
        return can.Bus(channel=port, interface="socketcan", fd=True), cal
    except Exception as e:  # noqa: BLE001
        print(f"[daemon] 右腕の CAN を開けない（{e}）。右腕は GRASP_REST_RIGHT の姿勢とみなす", flush=True)
        return None


def read_right(right) -> str | None:
    """右腕の今の角度（lerobot の度・7 つの文字列）。読めなければ None（計画の常駐が GRASP_REST_RIGHT を使う）。"""
    if right is None:
        return None
    from grasp.tools.read_arm import read_once
    import numpy as np
    q = read_once(*right)[:7]
    if not np.all(np.isfinite(q)):
        print("[daemon] 右腕の返事が無いモーターがある。右腕は GRASP_REST_RIGHT の姿勢とみなす", flush=True)
        return None
    return ",".join(f"{v:.2f}" for v in q)


def wait_trigger(trigger_dir: str | None) -> bool:
    """合図を待つ。続けるなら True、終わるなら False。"""
    if not trigger_dir:
        try:
            s = input("[daemon] Enter で掴む（q で終わる）: ")
        except EOFError:
            return False
        return s.strip().lower() != "q"
    go, quit_ = os.path.join(trigger_dir, "go"), os.path.join(trigger_dir, "quit")
    while True:
        if os.path.exists(quit_):
            os.remove(quit_)
            return False
        if os.path.exists(go):
            os.remove(go)
            return True
        time.sleep(0.005)


def main(argv=None, robot=None) -> int:
    """robot: 試験用に偽のロボットを渡す口（渡せば接続しない）。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("--planner", default="tcp://localhost:5560", help="計画の常駐（resident_plan.py --serve）の口")
    ap.add_argument("--run-args", default="", help="run_grasp.py に足す引数（--go と --early は付ける）")
    ap.add_argument("--no-require-place", dest="require_place", action="store_false",
                    help="置き先が解けなくても掴む（掴んで元の場所に戻す）。既定は置き先が無ければ物の近くへ行かない")
    ap.add_argument("--plan-timeout", type=float, default=20.0, help="計画を待つ上限 [s]")
    ap.add_argument("--trigger-dir", default=None, help="Enter の代わりに DIR/go のファイルを合図にする")
    a = ap.parse_args(argv)
    if os.environ.get("GRASP_ARM", "left") != "left":
        sys.exit("[daemon] 今は左腕だけ（手首のカメラ・休みの姿勢は左で確かめた）")

    ping = request(a.planner, {"op": "ping"}, 5.0)
    if not ping.get("ok"):
        sys.exit(f"[daemon] 計画の常駐に届かない: {ping}。先に resident_plan.py --serve を立てる")
    from grasp import run_grasp  # 置き場の env（GRASP_CUROBO など）を読んでから読み込む
    if robot is None:
        from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
        port, rid = os.environ.get("GRASP_CAN_PORT", "can_fol_l"), os.environ.get("GRASP_ROBOT_ID", "follower_left")
        print(f"[daemon] 左腕の CAN {port}・較正 {rid} につなぐ", flush=True)
        robot = OpenArmFollower(OpenArmFollowerConfig(port=port, side="left", id=rid, align_on_connect=False, cameras={},
                                                      disable_torque_on_disconnect=True))
        robot.connect()
    robot.bus.disable_torque()  # 合図まで休みの姿勢で脱力しておく（run_grasp.py が試行ごとに入れて、終わりに切る）
    right = open_right_reader()  # 右腕は動かさない。角度だけ読んで、切り出しで右手を物と間違えないようにする
    if a.trigger_dir:
        os.makedirs(a.trigger_dir, exist_ok=True)
        for f in ("go", "quit"):
            if os.path.exists(os.path.join(a.trigger_dir, f)):
                os.remove(os.path.join(a.trigger_dir, f))
    k = 0
    try:
        while wait_trigger(a.trigger_dir):
            t0 = time.monotonic()
            k += 1
            obs = robot.get_observation()
            q = ",".join(f"{float(obs[f'{j}.pos']):.2f}" for j in J)
            q_right = read_right(right)
            cap = request(a.planner, {"op": "capture"}, 5.0)
            if "error" in cap:
                print(f"[daemon] #{k} 撮れない（{cap['error']}）。腕は動かさない", flush=True)
                continue
            t_cap = time.monotonic() - t0
            result: dict = {}

            def plan_req(name=cap["name"], q=q, q_right=q_right):
                result.update(request(a.planner, {"op": "plan", "name": name, "q": q, "q_right": q_right}, a.plan_timeout))
                result["t_done"] = time.monotonic() - t0

            th = threading.Thread(target=plan_req, daemon=True)
            th.start()

            def plan_wait():
                th.join()
                if "error" in result:
                    print(f"[daemon] 計画の常駐: {result['error']}", flush=True)
                    return False
                place = result.get("place") or {}
                print(f"[daemon] 計画: {result.get('summary')}（合図から {result['t_done']:.2f} 秒）"
                      + (f" | 置き先: {place.get('summary')}" if place else ""), flush=True)
                if not result.get("planned"):
                    return False
                if a.require_place and not place.get("ok"):
                    print("[daemon] 置き先が無いので物の近くへ行かない（--no-require-place で掴んで戻すだけにできる）", flush=True)
                    return False
                return True

            print(f"[daemon] #{k} {cap['name']}: 撮った（合図から {t_cap:.2f} 秒）", flush=True)
            try:
                rc = run_grasp.main([cap["cap"], "--go", "--early", *shlex.split(a.run_args)],
                                    robot=robot, plan_wait=plan_wait, t_trigger=t0)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
                print(f"[daemon] #{k} 止まった: {e}", flush=True)
                if type(e).__name__ == "MotorLost":
                    print("[daemon] モーターの返事が途切れた。常駐を終える（電源・非常停止を確かめてから立て直す）", flush=True)
                    break
            with open(os.path.join(cap["cap"], "daemon.json"), "w") as f:
                json.dump({"cap": cap["cap"], "rc": rc, "plan": result, "t_capture_s": t_cap}, f, ensure_ascii=False, indent=2, default=str)
            print(f"[daemon] #{k} 終わり（rc={rc}・合図から {time.monotonic() - t0:.1f} 秒）", flush=True)
    except KeyboardInterrupt:
        print("[daemon] Ctrl-C: 終わる", flush=True)
    finally:
        try:
            robot.bus.disable_torque()
        finally:
            robot.disconnect()
            if right is not None:
                right[0].shutdown()
        print("[daemon] 脱力して接続を切った", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
