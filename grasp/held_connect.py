#!/usr/bin/env python3
"""Connect to an OpenArm arm WITHOUT dropping it. **Sends motor commands (holds the arm where it is).**

2026-11-22 17:3x (Sota: 「回の間で腕が落ちて、台車にぶつかってる」). lerobot's OpenArmFollower.connect() calls configure(),
which wraps configure_motors() in bus.torque_disabled(): every new connection turned the torque OFF and back on, and
then nothing was commanded until the trial had built its routes (seconds). An arm held at home v5 fell onto the cart
at the start of every trial (run.log: "raise from rest"). This opens the CAN bus, sends ENABLE only (a no-op for a motor
that is already holding; a limp arm gets torque at its current angle) and at once one MIT hold command (current angle,
OPENARM_ALIGN_KP/KD, gravity feed-forward). Damiao keeps executing the last MIT command, so the arm stays put while
the caller plans.

    from grasp.held_connect import connect_holding
    connect_holding(robot, grav_fn)            # instead of robot.connect()
"""
from __future__ import annotations

from typing import Callable, Dict, Optional


def connect_holding(robot, grav_fn: Optional[Callable[[Dict[str, float]], Dict[str, float]]] = None,
                    verbose: bool = True) -> Dict[str, float]:
    """Returns the joint angles read right after connecting (lerobot deg, incl. gripper)."""
    from lerobot.motors.damiao.damiao_alignment import OPENARM_ALIGN_KD, OPENARM_ALIGN_KP
    from lerobot.motors.damiao.tables import CAN_CMD_ENABLE
    bus = robot.bus
    bus.connect()
    if not robot.is_calibrated:
        bus.disconnect(False)
        raise SystemExit("[connect] the motors' calibration does not match the calibration file: run lerobot-calibrate "
                         "(not done here: it needs the arm limp)")
    st = bus.sync_read_all_states()
    q = {m: float(st[m]["position"]) for m in bus.motors}
    for m in bus.motors:  # ENABLE only. No DISABLE first (that is what dropped the arm)
        bus._send_simple_command(m, CAN_CMD_ENABLE)
    st = bus.sync_read_all_states()
    q = {m: float(st[m]["position"]) for m in bus.motors}
    joints = [m for m in bus.motors if m != "gripper"]
    g = grav_fn({m: q[m] for m in joints}) if grav_fn is not None else {}
    cmd = {m: (float(OPENARM_ALIGN_KP.get(m, 10.0)), float(OPENARM_ALIGN_KD.get(m, 1.0)), q[m], 0.0, float(g.get(m, 0.0)))
           for m in bus.motors}
    bus._mit_control_batch(cmd)
    if verbose:
        print("[connect] connected without turning torque off; holding at " +
              ",".join(f"{q[m]:.1f}" for m in joints), flush=True)
    return q
