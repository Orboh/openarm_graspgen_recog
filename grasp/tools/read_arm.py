#!/usr/bin/env python3
"""脱力した腕の関節 1〜7 とグリッパー（J8）の角度を読み続ける（lerobot の度）。**トルクを入れない・腕は動かさない。**

2026-11-22・作業者: Munenori。右腕を足すときの確かめ（関節の向き・グリッパーの全開・全閉・休みの姿勢）に使う。
読み方は grasp/enpire/live.py の READ_Q と同じ（モーターに状態を聞くだけ。enable も指令も送らない）。
角度 = モーターの位置 [deg] − 較正の homing_offset（run_grasp.py が lerobot で読む角度と同じ）。

    ~/projects/lerobot/.venv/bin/python grasp/tools/read_arm.py --arm right            # 0.2 秒ごとに 1 行・Ctrl-C で終わる
    ~/projects/lerobot/.venv/bin/python grasp/tools/read_arm.py --arm right --once     # 1 回だけ
    ~/projects/lerobot/.venv/bin/python grasp/tools/read_arm.py --arm right --minmax   # 終わるときに各関節の最小・最大を出す
CAN と較正は置き場の env の値（右は GRASP_CAN_PORT_RIGHT・GRASP_ROBOT_ID_RIGHT）か、--can・--id で渡す。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

DEFAULTS = {"left": ("can_lead_l", "leader_left_umi"), "right": ("can_lead_r", "leader_right_umi")}
NAMES = [f"joint_{i}" for i in range(1, 8)] + ["gripper"]


def read_once(bus, cal) -> list:
    """8 個のモーターの角度（返事の無いモーターは nan）。"""
    import can
    out = []
    for sid in range(1, 9):
        bus.send(can.Message(arbitration_id=0x7FF, data=bytes([sid, 0, 0xCC, 0, 0, 0, 0, 0]), is_extended_id=False))
        end, v = time.monotonic() + 0.1, float("nan")
        while time.monotonic() < end:
            m = bus.recv(timeout=0.01)
            if m and m.arbitration_id == 0x10 + sid:
                d = bytes(m.data)
                v = float(np.degrees(((d[1] << 8) | d[2]) / 65535 * 25 - 12.5) - cal[NAMES[sid - 1]]["homing_offset"])
                break
        out.append(v)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--arm", choices=["left", "right"], required=True)
    ap.add_argument("--can", default=None)
    ap.add_argument("--id", default=None, help="較正の名前（robots/openarm_follower/<ID>.json）")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--minmax", action="store_true")
    ap.add_argument("--period", type=float, default=0.2)
    a = ap.parse_args()
    suf = "_RIGHT" if a.arm == "right" else ""
    port = a.can or os.environ.get("GRASP_CAN_PORT" + suf) or DEFAULTS[a.arm][0]
    rid = a.id or os.environ.get("GRASP_ROBOT_ID" + suf) or DEFAULTS[a.arm][1]
    cal = json.load(open(os.path.expanduser(f"~/.cache/huggingface/lerobot/calibration/robots/openarm_follower/{rid}.json")))
    import can
    bus = can.Bus(channel=port, interface="socketcan", fd=True)
    print(f"[read-arm] {a.arm} 腕・CAN {port}・較正 {rid}（トルクは入れない）", flush=True)
    print("   " + " ".join(f"{n.replace('joint_', 'J'):>7}" for n in NAMES), flush=True)
    lo, hi = np.full(8, np.inf), np.full(8, -np.inf)
    try:
        while True:
            q = np.array(read_once(bus, cal))
            ok = np.isfinite(q)
            lo[ok], hi[ok] = np.minimum(lo[ok], q[ok]), np.maximum(hi[ok], q[ok])
            print("   " + " ".join("     --" if not np.isfinite(v) else f"{v:+7.1f}" for v in q), flush=True)
            if a.once:
                if not ok.all():
                    print(f"[read-arm] 返事の無いモーター: {[NAMES[i] for i in np.where(~ok)[0]]}", flush=True)
                break
            time.sleep(a.period)
    except KeyboardInterrupt:
        pass
    finally:
        bus.shutdown()
    if a.minmax:
        print("[read-arm] 最小: " + ",".join(f"{v:.1f}" for v in lo))
        print("[read-arm] 最大: " + ",".join(f"{v:.1f}" for v in hi))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
