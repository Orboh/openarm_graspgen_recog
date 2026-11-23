#!/usr/bin/env python3
"""Let an OpenArm arm go limp slowly ("damp mode"), then turn torque off. **The arm moves (it sinks under gravity).**

2026-11-22 (Sota, a rule for every tool): 「力を抜くときはゆっくり、ダンプモードみたいな感じでゆっくり下ろす。縁にぶつかったり、
モーターが台車の上で噛んだりして傷が入る」. Turning torque off at once drops the arm onto the cart edge.

What it does (MIT commands through the lerobot Damiao bus, 100 Hz):
  1. ramp  (ramp_s, ~4 s): stiffness kp and the gravity feed-forward fade from their holding values to 0 while the
     target is always the *current* angle (nothing is pulled back), damping kd stays on.
  2. damp  (damp_s, ~2 s+): kp = 0, kd only — the same idea as Unitree's Damp mode: the arm sinks slowly and settles.
     A joint moving faster than max_dps gets more damping. Ends when every joint moved < 1 deg in the last second
     (or after settle_max_s).
  3. torque off.
The gripper gets damping only (kp fades too). Nothing can be done when the motors stopped answering (MotorLost / e-stop).

    from grasp.soft_release import soft_release
    soft_release(robot, grav_fn)            # then robot.disconnect() (torque already off)
"""
from __future__ import annotations

import time
from typing import Callable, Dict, Optional

import numpy as np


def soft_release(robot, grav_fn: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None, ramp_s: float = 4.0,
                 damp_s: float = 2.0, settle_max_s: float = 8.0, kd: float = 1.0, kd_max: float = 3.0,
                 max_dps: float = 30.0, log: Optional[list] = None, verbose: bool = True) -> float:
    """Returns the fastest joint speed seen while releasing [deg/s]. Torque is off afterwards."""
    bus = robot.bus
    motors = list(bus.motors)
    joints = [m for m in motors if m != "gripper"]
    st = bus.sync_read_all_states()
    q = {m: float(st[m]["position"]) for m in motors}
    kp0 = {m: float(bus._gains.get(m, {}).get("kp", 10.0)) for m in motors}
    kd_now = {m: max(kd, float(bus._gains.get(m, {}).get("kd", 0.5))) for m in motors}
    dt = 0.01
    vmax = 0.0
    hist = []
    t0 = time.time()
    phase_end = t0 + ramp_s
    prev = t0
    while True:
        now = time.time(); step = max(1e-3, now - prev); prev = now
        s = min(1.0, (now - t0) / ramp_s)
        g = grav_fn({m: q[m] for m in joints}) if (grav_fn is not None and s < 1.0) else {}
        cmd = {}
        for m in motors:
            tau = float(g.get(m, 0.0)) * (1.0 - s)
            cmd[m] = (kp0[m] * (1.0 - s), kd_now[m], q[m], 0.0, tau)
        bus._mit_control_batch(cmd)
        st = bus._last_known_states
        qn = {m: float(st[m]["position"]) for m in motors}
        for m in joints:
            v = abs(qn[m] - q[m]) / step
            vmax = max(vmax, v) if s > 0.05 else vmax
            if v > max_dps:
                kd_now[m] = min(kd_max, kd_now[m] * 1.15)  # sinking too fast: more damping
        q = qn
        hist.append((now, [q[m] for m in joints]))
        if log is not None:
            log.append((now - t0, [q[m] for m in joints], [kd_now[m] for m in joints]))
        if now >= phase_end + damp_s:
            # settled: every joint moved < 1 deg over the last second
            old = [h for h in hist if h[0] >= now - 1.0]
            if old and np.max(np.abs(np.array(old[-1][1]) - np.array(old[0][1]))) < 1.0:
                break
            if now >= phase_end + settle_max_s:
                break
        time.sleep(max(0.0, dt - (time.time() - now)))
    bus.disable_torque()
    if verbose:
        print(f"[release] soft release done in {time.time() - t0:.1f} s (fastest joint {vmax:.0f} deg/s), torque off", flush=True)
    return vmax


def release_and_disconnect(robot, grav_fn=None, skip: bool = False, **kw) -> None:
    """robot.disconnect(), but when it would turn torque off, let the arm go limp slowly first (soft_release).
    skip=True (the motors stopped answering) or a held pose (disable_torque_on_disconnect=False) disconnect directly."""
    if not skip and getattr(robot.config, "disable_torque_on_disconnect", True):
        try:
            soft_release(robot, grav_fn, **kw)
        except Exception as e:  # noqa: BLE001  never block the disconnect
            print(f"[release] soft release failed ({type(e).__name__}: {e}); turning torque off directly", flush=True)
    robot.disconnect()
