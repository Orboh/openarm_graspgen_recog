"""grasp/wrist_servo.py（手首のカメラで爪から見た物の位置）を合成した深度で確かめる（実機・カメラなし）。

    PYTHONPATH=. ~/projects/lerobot/.venv/bin/python -m unittest tests.test_wrist_servo -q
"""
import tempfile
import unittest

import numpy as np

from grasp.wrist_servo import JawModel, descend_travel, find_target, lateral_step, offsets

F = 400.0
K = np.array([[F, 0, 320], [0, F, 240], [0, 0, 1.0]])
FLOOR = 0.30  # カメラから机まで [m]（真下を向いたカメラ）


def jaw():
    """カメラが近づく向きを向き、爪の先の中点がカメラの 13 cm 前・閉じる向きが画像の右・厚みが画像の下の爪。"""
    return JawModel(mid=np.array([0.0, 0.0, 0.13]), C=np.array([1.0, 0, 0]), T=np.array([0, 1.0, 0]), A=np.array([0, 0, 1.0]),
                    K=K, half_open=0.04, tips=(np.array([-0.04, 0, 0.13]), np.array([0.04, 0, 0.13])))


def scene(boxes):
    """机（深さ FLOOR）の上に、(中心 x, 中心 y, 幅 x, 幅 y, 高さ) [m] の箱を置いた深度。"""
    v, u = np.mgrid[0:480, 0:640]
    d = np.full((480, 640), FLOOR, np.float32)
    for cx, cy, wx, wy, h in boxes:
        z = FLOOR - h
        x, y = (u - 320) / F * z, (v - 240) / F * z
        inside = (np.abs(x - cx) <= wx / 2) & (np.abs(y - cy) <= wy / 2)
        d = np.where(inside & (z < d), z, d)
    return d


class WristServoTest(unittest.TestCase):
    def test_offset_of_a_box(self):
        """閉じる向きに +20 mm・厚みに -10 mm ずれた箱は、そのずれで出る。"""
        t = find_target(scene([(0.02, -0.01, 0.05, 0.05, 0.06)]), jaw())
        o = offsets(t, jaw())
        self.assertTrue(o["c_ok"] and o["t_ok"])
        self.assertAlmostEqual(o["e_c"], 0.02, delta=0.003)
        self.assertAlmostEqual(o["e_t"], -0.01, delta=0.003)
        self.assertAlmostEqual(o["tip_h"], FLOOR - 0.13, delta=0.002)
        step, done = lateral_step(o)
        self.assertFalse(done)
        np.testing.assert_allclose(step, [0.016, -0.008, 0.0], atol=0.003)  # 0.8 倍に弱めて直す

    def test_hint_picks_the_right_one_of_two_touching(self):
        """高さの違う 2 つの箱が接して写っても、深さの飛びで分け、計画の位置に近い方を選ぶ。"""
        d = scene([(-0.03, 0.0, 0.05, 0.05, 0.06), (0.025, 0.0, 0.06, 0.05, 0.03)])
        for hint, want in (((-0.03, 0.0, 0.1), -0.03), ((0.025, 0.0, 0.1), 0.025)):
            o = offsets(find_target(d, jaw(), hint_jaw=np.array(hint)), jaw())
            self.assertAlmostEqual(o["e_c"], want, delta=0.004)

    def test_cut_side_peeks_toward_the_cut(self):
        """画像の上の端で切れた向き（厚み）は、切れた見え方の中心では直さず、切れた側へ 15 mm 寄せて全体を写しにいく。
        （この合成の爪は厚み +T が画像の下なので、上で切れたら -T へ。2026-11-23 北九州の 1 回目で 25〜29 mm ずれたまま閉じた）"""
        o = offsets(find_target(scene([(0.0, -0.20, 0.05, 0.30, 0.06)]), jaw()), jaw())
        self.assertFalse(o["t_ok"])
        self.assertEqual(o["t_dir"], -1.0)
        step, done = lateral_step(o)
        self.assertFalse(done)
        self.assertAlmostEqual(step[1], -0.015, places=6)

    def test_descend_stops_above_the_floor(self):
        """下ろす量は、爪の先の角が机から floor_clear より下がらない所まで。"""
        o = {"tip_h": 0.10, "drop": 1.0, "a_wide": 0.12}  # 一番太い所はずっと先（机より下になる）
        travel, why = descend_travel(o, depth_target=0.03, floor_clear=0.008)
        self.assertAlmostEqual(travel, 0.10 - 0.008, delta=1e-6)
        self.assertIn("机", why)
        o = {"tip_h": 0.10, "drop": 1.0, "a_wide": 0.02}
        travel, _ = descend_travel(o, depth_target=0.03)
        self.assertAlmostEqual(travel, 0.05, delta=1e-6)


class TShiftTest(unittest.TestCase):
    """--servo-t-shift（手を爪の厚みの向きへ寄せる）の向き。2026-11-23: 毎回、爪の奥側（−Yg）の長辺で木材を挟んでいた。"""

    def test_converges_with_hand_shifted(self):
        """目標（bias_t = −t_shift）を引いた差で横を直すと、手は物に対して +Yg へ t_shift だけずれた所で止まる。"""
        for t_shift in (-0.02, 0.015):
            bias_t = -t_shift
            hand = 0.0  # 物に対する手の位置（Yg）
            for _ in range(30):
                e_t_raw = -hand  # 手が +Yg にあれば物は −Yg に見える
                step, done = lateral_step({"e_c": 0.0, "e_t": e_t_raw - bias_t, "c_ok": True, "t_ok": True}, tol=0.0005)
                if done:
                    break
                hand += step[1]
            self.assertAlmostEqual(hand, t_shift, delta=0.001)


class OneShotTest(unittest.TestCase):
    """--servo-one-shot: 1 回測って、横の直しと下ろす動きを 1 回の動きにまとめる。大きくずれていれば今までどおり何度も直す。
    2026-11-23: 直した量への応答はほぼ 1 倍（閉じる 0.86・厚み 1.13）で、測りの揺れが 4〜5 mm あり、くり返しは揺れを追っていた。"""

    def run_servo(self, measured, one_shot=True, t_shift=0.0):
        from unittest import mock
        from grasp import wrist_approach as wa

        def fk(q):  # 関節角の頭 3 つを位置 [mm] とみなす偽の腕（向きは単位行列）
            T = np.eye(4); T[:3, 3] = np.asarray(q[:3], float) / 1000.0
            return T

        def solve(T, seed):
            return [*(T[:3, 3] * 1000.0), 0, 0, 0, 0]
        moves, seen = [], iter(measured)

        def fake_offsets(tgt, jaw):
            e_c, e_t = next(seen)
            return {"e_c": e_c, "e_t": e_t, "c_ok": True, "t_ok": True, "c_dir": 0.0, "t_dir": 0.0,
                    "a_wide": 0.01, "tip_h": 0.20, "drop": 1.0, "px": 100}

        class Cam:
            def grab(self, *a, **k):
                return np.zeros((4, 4, 3), np.uint8), np.zeros((4, 4), np.float32)
        tgt = {"mask": np.zeros((4, 4), bool), "pick": "test"}
        with tempfile.TemporaryDirectory() as d, \
                mock.patch.object(wa, "find_target", lambda *a, **k: tgt), mock.patch.object(wa, "offsets", fake_offsets), \
                mock.patch.dict("sys.modules", {"cv2": mock.MagicMock()}):
            q = wa.servo_approach(grasp_q=[0, 0, 0, 0, 0, 0, 0], pre_q=[0, 0, -100, 0, 0, 0, 0], fk=fk, solve=solve,
                                  go=lambda q, *a, **k: moves.append((a[-1], list(q))), check=lambda *a, **k: None, cam=Cam(),
                                  jaw=None, obj_base=np.zeros(3), obj_extent=0.05, g_open=-62, out_dir=d, near=0.04,
                                  depth_target=0.03, one_shot=one_shot, t_shift=t_shift, log=lambda m: None)
        return q, moves

    def test_one_move_after_one_measure(self):
        """測るのは 1 回、動くのは「近い所へ」と「直して下ろす」の 2 回。横は測った差を全部直す。"""
        from grasp.wrist_approach import AX_L7_G
        q, moves = self.run_servo([(0.008, -0.006)])
        self.assertEqual([m[0] for m in moves], ["servo near", "servo one-shot"])
        want = AX_L7_G @ np.array([0.008, -0.006, 0.04 + 0.0])  # 近い所（−near）から: 横 (8, −6) mm・下ろす a_wide + depth = 40 mm
        np.testing.assert_allclose(np.asarray(q[:3]) / 1000.0, AX_L7_G @ np.array([0, 0, -0.04]) + want, atol=1e-6)

    def test_t_shift_applies_in_one_shot(self):
        """厚みの向きの寄せ（--servo-t-shift）も 1 回の動きに入る。"""
        from grasp.wrist_approach import AX_L7_G
        q, _ = self.run_servo([(0.0, -0.010)], t_shift=-0.02)
        lat = AX_L7_G.T @ (np.asarray(q[:3]) / 1000.0)
        self.assertAlmostEqual(lat[1], -0.010 + (-0.02), places=6)  # 物の見え（−10 mm）に手を −20 mm 寄せる分を足す

    def test_large_closing_offset_falls_back(self):
        """閉じる向きに 12 mm を超えてずれていれば、斜めに下ろさず今までどおり直してから下ろす。"""
        q, moves = self.run_servo([(0.020, 0.0), (0.002, 0.0), (0.001, 0.0)])
        names = [m[0] for m in moves]
        self.assertNotIn("servo one-shot", names)
        self.assertIn("servo align 0", names)


if __name__ == "__main__":
    unittest.main()
