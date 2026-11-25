"""run_grasp.py --hover --keep-torque（回の間に休みの姿勢へ戻らず、上げた所でモーターを入れたまま待つ）を偽のロボットで確かめる。腕は動かさない。

2026-11-25・作業者: Shreyan。選別を速くするため、1 つ置くたびに休みの姿勢へ下ろして脱力し、次の回にまた上げていたのをやめる。
安全に関わる所なので、先に次を確かめる:
- --hover --keep-torque: 終わりは上げた姿勢（raise）で、モーターを切らない・接続も切らない
- 上げた姿勢から始めた回は、もう上げずに（同じ姿勢を raise として）掴み、終わりはその姿勢に戻る
- 付けなければ今までどおり休みの姿勢へ戻って脱力する

    cd tests && PYTHONPATH=..:$HOME/projects/lerobot/src ~/projects/lerobot/.venv/bin/python -m unittest test_hover
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from test_early_start import HAVE_LEROBOT, J, PLAN, REST, FakeRobot


@unittest.skipUnless(HAVE_LEROBOT, "lerobot の venv で走らせる")
class HoverTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"GRASP_FLOOR_Z": "0.230", "GRASP_GRIPPER_OPEN": "-62", "GRASP_GRIPPER_CLOSE": "-3",
                                                "GRASP_ARM": "left"})
        self.env.start()
        for k in ("GRASP_CUROBO", "GRASP_CUROBO_RIGHT", "GRASP_SITE", "GRASP_KEEPOUT_BOXES", "GRASP_T_BASE_CAM", "GRASP_WRIST_JAW",
                  "GRASP_TIMING_LOG"):
            os.environ.pop(k, None)
        from grasp import run_grasp
        self.rg = importlib.reload(run_grasp)
        self.p_sleep = mock.patch.object(self.rg.time, "sleep", lambda s: None)
        self.p_sleep.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.cap = self.tmp.name
        with open(os.path.join(self.cap, "plan.json"), "w") as f:
            json.dump(PLAN, f)

    def tearDown(self):
        self.p_sleep.stop(); self.env.stop(); self.tmp.cleanup()

    def run_main(self, robot, *extra):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = self.rg.main([self.cap, "--go", "--transit-dps", "75", "--near-dps", "20", "--hold-s", "0", *extra], robot=robot)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
        return rc, out.getvalue()

    def pos(self, robot):
        return np.array([robot.bus.pos[j] for j in J])

    def test_default_goes_back_to_rest_and_releases(self):
        """付けなければ今までどおり: 休みの姿勢へ戻って脱力する。"""
        robot = FakeRobot(REST)
        rc, out = self.run_main(robot)
        self.assertEqual(rc, 0, out)
        self.assertIn("back to start done", out)
        np.testing.assert_allclose(self.pos(robot), REST, atol=1e-6)
        self.assertGreaterEqual(robot.bus.disabled, 1)

    def test_hover_ends_raised_with_torque_on(self):
        """--hover --keep-torque: 掴んで戻した後、休みの姿勢へ下ろさず上げた所で止まり、モーターを切らない。"""
        robot = FakeRobot(REST)
        rc, out = self.run_main(robot, "--hover", "--keep-torque")
        self.assertEqual(rc, 0, out)
        self.assertIn("approach done", out)
        self.assertIn("back to raise done", out)
        self.assertNotIn("back to start done", out)
        self.assertEqual(robot.bus.disabled, 0)  # モーターを入れたまま
        self.assertEqual(robot.disconnected, 0)
        end = self.pos(robot)
        self.assertGreater(np.max(np.abs(end - np.array(REST))), 1.0)  # 休みの姿勢ではない
        # 上げた姿勢は休みの姿勢より爪が上にある
        self.assertGreater(self.tcp_z(end), self.tcp_z(REST) + 0.03)

    def test_hover_start_does_not_raise_again(self):
        """上げた姿勢から始めた回は、もう上げずに掴み、終わりは始めた姿勢に戻る（上へずれていかない）。"""
        robot = FakeRobot(REST)
        rc, out = self.run_main(robot, "--hover", "--keep-torque")
        self.assertEqual(rc, 0, out)
        hover = self.pos(robot).copy()
        rc, out = self.run_main(robot, "--hover", "--keep-torque")
        self.assertEqual(rc, 0, out)
        self.assertIn("approach done", out)
        np.testing.assert_allclose(self.pos(robot), hover, atol=1e-6)
        self.assertEqual(robot.bus.disabled, 0)

    def tcp_z(self, q):
        from ik.openarm_left_ik import OpenArmIK
        ik = OpenArmIK(side="left")
        M = ik.fk(np.deg2rad(q))
        return float((M.translation + M.rotation @ [0, 0, 0.2145])[2])


if __name__ == "__main__":
    unittest.main()
