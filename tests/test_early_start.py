"""run_grasp.py --early（計画を待たずに真上へ上げ始める）を、偽のロボットで確かめる。腕は動かさない。

2026-11-23・作業者: Munenori。「計画して」から腕が動き出すまでを 1 秒以内にするため、撮った直後に休みの姿勢から真上へ上げ、
爪を開く間に裏で計画と確かめを済ませる。安全に関わる所なので、先に次を確かめる:
- 計画が立たない・計画を待つ口が例外を出す・計画の腕が違う → 上げて開いた後、物の近くへは行かずに休みの姿勢へ戻り、脱力する
- 計画が立つ → 計画を待つ口が返る前に腕が動き出し、掴む手順を最後まで通す
- 渡したロボットの接続は切らない（常駐のプロセスが使い回す）。終わりは脱力する
"""
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import tempfile
import threading
import unittest
from unittest import mock

import numpy as np

try:
    from lerobot.motors.damiao.damiao_alignment import _ALIGN_STEP_S  # noqa: F401
    HAVE_LEROBOT = True
except Exception:  # noqa: BLE001  lerobot の venv でないとき
    HAVE_LEROBOT = False

J = [f"joint_{i}" for i in range(1, 8)]
REST = [11.0, -4.4, -3.4, 16.9, -0.6, 2.0, -80.0]  # 2026-11-23 北九州の左腕の休みの姿勢（読んだ角度）
# 2026-11-23 cap_resident_20261123_150021 の計画（左腕・木材）
PLAN = {
    "arm": "left",
    "q_start_deg": REST,
    "q_pre_deg": [-57.0, -45.2, 70.6, 102.5, -61.0, 15.5, 88.2],
    "q_grasp_deg": [-25.7, -26.8, 40.4, 83.0, -30.3, -6.2, 84.2],
    "q_lift_deg": [-57.4, -44.4, 69.5, 99.2, -61.2, 12.7, 88.2],
}


class FakeBus:
    """送られた目標の角度にすぐ着くモーターの口。モーターを入れた・切った回数と、最初に指令が来た時刻を覚える。"""

    def __init__(self, start, gripper=-7.0):
        self.motors = {m: None for m in J + ["gripper"]}
        self.pos = dict(zip(J, start)); self.pos["gripper"] = gripper
        self.enabled = self.disabled = 0
        self.first_cmd = threading.Event()
        self._last_update_ts = {}

    def _touch(self):
        import time
        now = time.monotonic()
        for m in self.motors:
            self._last_update_ts[m] = now

    def sync_read_all_states(self):
        self._touch()
        return {m: {"position": v} for m, v in self.pos.items()}

    def _mit_control_batch(self, cmd):
        self.first_cmd.set()
        for m, c in cmd.items():
            self.pos[m] = c[2]
        self._touch()

    def enable_torque(self, *a, **k):
        self.enabled += 1

    def disable_torque(self, *a, **k):
        self.disabled += 1


class FakeRobot:
    def __init__(self, start):
        self.bus = FakeBus(start)
        self.disconnected = 0

    def get_observation(self):
        self.bus._touch()
        return {f"{m}.pos": v for m, v in self.bus.pos.items()}

    def disconnect(self):
        self.disconnected += 1


@unittest.skipUnless(HAVE_LEROBOT, "lerobot の venv で走らせる")
class EarlyStartTest(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"GRASP_FLOOR_Z": "0.230", "GRASP_GRIPPER_OPEN": "-62", "GRASP_GRIPPER_CLOSE": "-3",
                                                "GRASP_ARM": "left"})
        self.env.start()
        for k in ("GRASP_CUROBO", "GRASP_CUROBO_RIGHT", "GRASP_SITE", "GRASP_KEEPOUT_BOXES", "GRASP_T_BASE_CAM", "GRASP_WRIST_JAW"):
            os.environ.pop(k, None)
        from grasp import run_grasp
        self.rg = importlib.reload(run_grasp)  # CUROBO・G_OPEN はモジュールを読むときの env で決まる
        # 10 ms ごとの待ちだけを飛ばす（time は全体で 1 つなので、grasp/motion.py の待ちも飛ぶ）
        self.p_sleep = mock.patch.object(self.rg.time, "sleep", lambda s: None)
        self.p_sleep.start()
        self.tmp = tempfile.TemporaryDirectory()
        self.cap = self.tmp.name

    def tearDown(self):
        self.p_sleep.stop(); self.env.stop(); self.tmp.cleanup()

    def run_main(self, plan_wait, robot):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            try:
                rc = self.rg.main([self.cap, "--go", "--early", "--transit-dps", "75", "--near-dps", "20", "--hold-s", "0"],
                                  robot=robot, plan_wait=plan_wait)
            except SystemExit as e:
                rc = e.code if isinstance(e.code, int) else 1
        return rc, out.getvalue()

    def write_plan(self, **over):
        with open(os.path.join(self.cap, "plan.json"), "w") as f:
            json.dump({**PLAN, **over}, f)

    def assert_went_back_only(self, rc, out, robot):
        self.assertNotEqual(rc, 0, out)
        self.assertIn("[run] raise done", out)
        self.assertIn("[run] open gripper done", out)
        self.assertIn("back to start done", out)
        for near in ("pregrasp done", "approach done", "close gripper done"):
            self.assertNotIn(near, out)
        np.testing.assert_allclose([robot.bus.pos[j] for j in J], REST, atol=1e-6)
        self.assertGreaterEqual(robot.bus.disabled, 1)  # 休みの姿勢で脱力する
        self.assertEqual(robot.disconnected, 0)  # 常駐のプロセスの接続は切らない

    def test_plan_fails_goes_back(self):
        """計画が立たない: 上げて開いた後、物の近くへ行かずに休みの姿勢へ戻って脱力する。"""
        robot = FakeRobot(REST)
        rc, out = self.run_main(lambda: False, robot)
        self.assert_went_back_only(rc, out, robot)

    def test_plan_wait_raises_goes_back(self):
        """計画を待つ口が例外を出す（計画のプロセスが落ちた）: 同じく戻って脱力する。"""
        robot = FakeRobot(REST)

        def boom():
            raise RuntimeError("planner down")
        rc, out = self.run_main(boom, robot)
        self.assert_went_back_only(rc, out, robot)

    def test_wrong_arm_goes_back(self):
        """計画が別の腕（右）: 左腕で上げた後、物の近くへ行かずに戻る。"""
        robot = FakeRobot(REST)

        def wait():
            self.write_plan(arm="right"); return True
        rc, out = self.run_main(wait, robot)
        self.assert_went_back_only(rc, out, robot)

    def test_moves_before_plan_and_completes(self):
        """計画を待つ口が返る前に腕が動き出し、計画が立てば掴む手順を最後まで通して休みの姿勢で脱力する。"""
        robot = FakeRobot(REST)
        seen = {}

        def wait():
            seen["moved_before_plan"] = robot.bus.first_cmd.wait(timeout=10.0)
            self.write_plan(); return True
        rc, out = self.run_main(wait, robot)
        self.assertEqual(rc, 0, out)
        self.assertTrue(seen.get("moved_before_plan"), "計画を待つ間に腕が動き出していない")
        for ph in ("raise done", "open gripper done", "pregrasp done", "approach done", "close gripper done", "back to start done"):
            self.assertIn(ph, out)
        np.testing.assert_allclose([robot.bus.pos[j] for j in J], REST, atol=1e-6)
        self.assertGreaterEqual(robot.bus.enabled, 1)
        self.assertGreaterEqual(robot.bus.disabled, 1)
        self.assertEqual(robot.disconnected, 0)

    def test_early_refuses_without_go(self):
        """--early は --go と一緒にだけ使う（乾式の確かめには計画が要る）。"""
        robot = FakeRobot(REST)
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit):
            self.rg.main([self.cap, "--early"], robot=robot, plan_wait=lambda: True)
        self.assertFalse(robot.bus.first_cmd.is_set())


PLAN_PLACE = {  # 2026-11-23 cap_resident_20261123_171854（木材を青い重量計の上に置いた回）
    "arm": "left",
    "q_start_deg": [11.1, -1.6, 0.8, 14.9, -6.7, -1.7, -82.0],
    "q_pre_deg": [-35.8, -39.4, 50.0, 105.8, -74.6, -20.2, 89.8],
    "q_grasp_deg": [-13.8, -19.8, 39.0, 92.8, -46.2, -33.4, 86.9],
    "q_lift_deg": [-37.9, -34.1, 55.2, 110.9, -71.6, -9.5, 89.8],
    "q_carry_deg": [-55.1, -33.7, 61.5, 98.2, -77.1, 28.0, 75.5],
    "q_place_deg": [-31.0, -20.3, 38.4, 91.2, -50.0, 12.5, 83.5],
}


class ReturnAfterPlaceTest(EarlyStartTest):
    """置いた後の戻り（2026-11-23）: 爪の先が置いた物の上端より上に来るまで真上へ上げてから、最初に上げた姿勢へ戻る。"""

    def test_place_then_lift_clear_and_back(self):
        """置き先のある計画: 置く → 離す → 真上へ上げる（6 cm 以上）→ 最初に上げた姿勢 → 休みの姿勢、の順に通って脱力する。"""
        import numpy as np
        robot = FakeRobot(PLAN_PLACE["q_start_deg"])
        rng = np.random.default_rng(0)  # 拾う物の点（カメラの座標の代わりに付け根の座標・T_base_cam を単位行列にする）
        obj = np.c_[rng.uniform(-0.025, 0.025, 500), rng.uniform(-0.025, 0.025, 500), rng.uniform(0.23, 0.33, 500)]
        np.save(os.path.join(self.cap, "object.npy"), obj.astype(np.float32))

        def wait():
            with open(os.path.join(self.cap, "plan.json"), "w") as f:
                json.dump(PLAN_PLACE, f)
            return True
        rc, out = self.run_main(wait, robot)
        self.assertEqual(rc, 0, out)
        order = [ph for ph in ("put down at the new spot", "release", "up from the new spot", "back to raise", "back to start")
                 if f"[run] {ph} done" in out]
        self.assertEqual(order, ["put down at the new spot", "release", "up from the new spot", "back to raise", "back to start"], out)
        self.assertIn("置いた後の戻り", out)
        self.assertGreaterEqual(robot.bus.disabled, 1)

    def test_lift_amount(self):
        """真上へ上げる量: 爪の先が物の上端 + 3 cm に来るまで。少なくとも 6 cm（今までの運ぶ高さ）。"""
        self.assertAlmostEqual(self.rg.clear_lift_dz(tip_z=0.30, obj_top_z=0.38), 0.11)
        self.assertAlmostEqual(self.rg.clear_lift_dz(tip_z=0.30, obj_top_z=0.31), 0.06)

    def test_shift_box(self):
        """置いた物の箱 = 拾う物の箱を、掴んだ所から置いた所への手の動きだけずらした物。"""
        b = {"name": "target", "dims": [0.05, 0.05, 0.1], "pose": [0.3, 0.1, 0.28, 1.0, 0.0, 0.0, 0.0]}
        s = self.rg.shift_box(b, [0.01, -0.2, 0.07], name="placed")
        self.assertEqual(s["name"], "placed")
        self.assertEqual(s["dims"], [0.05, 0.05, 0.1])
        np.testing.assert_allclose(s["pose"][:3], [0.31, -0.1, 0.35])


if __name__ == "__main__":
    unittest.main()
