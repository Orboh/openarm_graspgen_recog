"""grasp/motion.py の follow_path（cuRobo の細かい軌道を止まらずにたどる）を、偽のモーターの口で確かめる。腕は動かさない。

2026-11-23・作業者: Munenori。実機で使う前に、どの関節も速さの上限を超えないこと・終点で終わること・始点から遠いと動かさないことを見る。
"""
from __future__ import annotations

import unittest
from unittest import mock

import numpy as np

try:
    from lerobot.motors.damiao.damiao_alignment import _ALIGN_STEP_S  # noqa: F401
    HAVE_LEROBOT = True
except Exception:  # noqa: BLE001  lerobot の venv でないとき
    HAVE_LEROBOT = False

J = [f"joint_{i}" for i in range(1, 8)]


class FakeBus:
    """送られた目標の角度を覚えるだけのモーターの口。"""

    def __init__(self, start, gripper=-3.0):
        self.motors = {m: None for m in J + ["gripper"]}
        self.pos = dict(zip(J, start)); self.pos["gripper"] = gripper
        self.sent = []

    def sync_read_all_states(self):
        return {m: {"position": v} for m, v in self.pos.items()}

    def _mit_control_batch(self, cmd):
        self.sent.append({m: c[2] for m, c in cmd.items()})


@unittest.skipUnless(HAVE_LEROBOT, "lerobot の venv で走らせる")
class FollowPathTest(unittest.TestCase):
    def setUp(self):
        from grasp import motion
        self.motion = motion
        self.p = mock.patch.object(motion.time, "sleep", lambda s: None)
        self.p.start()

    def tearDown(self):
        self.p.stop()

    def test_speed_limit_and_end(self):
        """曲がった軌道でも、どの関節も上限の速さを超えず、終点で終わる。"""
        from lerobot.motors.damiao.damiao_alignment import _ALIGN_STEP_S
        path = np.array([[0, 0, 0, 0, 0, 0, 0], [30, 5, 0, 0, 0, 0, 0], [30, 5, 40, -20, 0, 0, 10], [10, 5, 40, -20, 0, 0, 10]], float)
        bus = FakeBus(path[0])
        self.motion.follow_path(bus, J, path, max_speed_dps=15.0, gripper_deg=-60.0)
        Q = np.array([[s[m] for m in J] for s in bus.sent])
        v = np.max(np.abs(np.diff(np.vstack([path[:1], Q]), axis=0)), axis=1) / _ALIGN_STEP_S
        self.assertLessEqual(float(v.max()), 15.0 * 1.02)
        np.testing.assert_allclose(Q[-1], path[-1], atol=1e-6)
        self.assertAlmostEqual(bus.sent[-1]["gripper"], -60.0, places=6)

    def test_refuse_far_start(self):
        """今の角度が軌道の始点から 5° より離れていれば、何も送らずに止める。"""
        path = np.zeros((3, 7)); path[:, 0] = [0, 10, 20]
        bus = FakeBus([8, 0, 0, 0, 0, 0, 0])
        with self.assertRaises(RuntimeError):
            self.motion.follow_path(bus, J, path, max_speed_dps=15.0)
        self.assertEqual(bus.sent, [])


if __name__ == "__main__":
    unittest.main()
