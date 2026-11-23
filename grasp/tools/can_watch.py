#!/usr/bin/env python3
"""Watch which motors of one arm answer, while a person moves the arm by hand. **Read only: no enable, no command.**

2026-11-23 (Sota: 右手首の J5〜J7・爪が返事をしなくなった → 肘のコネクタを入れ直した → 手首を回して切れないか確かめる).
Same idea as the 9/21 Itabashi elbow check (J4->J5 CAN contact opening with the pose). Every 0.2 s: state-refresh
all 8 motors; a motor whose state did not update counts as a drop. Prints one line per second (angles of J5-J7 and
the motors that dropped) and a summary at the end.

    ~/projects/lerobot/.venv/bin/python grasp/tools/can_watch.py can_fol_r right [seconds=90]
"""
import sys
import time

from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig


def main():
    port, side = sys.argv[1], sys.argv[2]
    secs = float(sys.argv[3]) if len(sys.argv) > 3 else 90.0
    r = OpenArmFollower(OpenArmFollowerConfig(port=port, side=side, id=f"follower_{side}", align_on_connect=False,
                                              cameras={}, disable_torque_on_disconnect=False))
    b = r.bus
    b.connect()  # handshake only; no configure(), no enable
    motors = list(b.motors)
    drops = {m: 0 for m in motors}; first_drop = {}
    t0 = time.time(); last_print = 0.0; n = 0
    print(f"[watch] {port}: watching {motors} for {secs:.0f} s. Move the arm by hand now.", flush=True)
    while time.time() - t0 < secs:
        before = {m: b._last_update_ts.get(m, 0.0) for m in motors}
        try:
            st = b.sync_read_all_states()
        except Exception as e:  # noqa: BLE001
            st = None
            print(f"[watch] {time.time()-t0:5.1f}s read error: {e}", flush=True)
        n += 1
        lost = [m for m in motors if b._last_update_ts.get(m, 0.0) <= before[m]]
        for m in lost:
            drops[m] += 1
            first_drop.setdefault(m, time.time() - t0)
        now = time.time() - t0
        if now - last_print >= 1.0 or lost:
            ang = " ".join(f"{m[-1] if m != 'gripper' else 'g'}:{float(st[m]['position']):+6.1f}" for m in motors
                           if st is not None and m in ("joint_4", "joint_5", "joint_6", "joint_7", "gripper"))
            print(f"[watch] {now:5.1f}s {ang}" + (f"  DROP {lost}" if lost else "  ok"), flush=True)
            last_print = now
        time.sleep(0.2)
    b.disconnect(False)
    bad = {m: c for m, c in drops.items() if c}
    print(f"[watch] done: {n} reads. " + (f"DROPS {bad} (first at {({m: round(t, 1) for m, t in first_drop.items()})} s)"
                                          if bad else "no drops on any motor"), flush=True)


if __name__ == "__main__":
    main()
