"""両腕を同時に動かしてよいかの確かめ（grasp/dual.py の左右の通り道）。実機・GPU なし。

2026-11-25・作業者: Shreyan。cuRobo のサーバーは、もう片方の腕を休みの姿勢で止まっている物としてしか見ないので、両腕が同時に動くときの
腕どうしの当たりは見ない。ここで、左腕の回の通り道（関節 3〜7・指先・爪の先を、計画の姿勢の間の関節の直線でなぞった点）が、右腕の回の
通り道より、付け根の座標の y で gap 以上左にあるときだけ、同時に動かしてよいとする。

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python -m unittest tests.test_dual -q
"""
from __future__ import annotations

import os
import unittest

import numpy as np

try:
    import pinocchio  # noqa: F401
    HAVE_PIN = True
except ImportError:
    HAVE_PIN = False

REST_L = [13.3, -1.1, -0.2, 12.1, 0.3, 1.4, -87.7]   # kitakyushu-collect.env の GRASP_REST
REST_R = [-15.6, -0.4, 1.5, 11.6, 1.0, -1.9, 89.5]   # GRASP_REST_RIGHT


def reach(side, xyz):
    """side の腕の hand_tcp（URDF の手先・実物の指先より約 17 cm 上）を xyz（付け根の座標）へ持っていく関節角 [deg]（位置だけ・テスト用）。"""
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(side=side)
    best = None
    for seed in np.random.default_rng(0).uniform(ik.lo, ik.hi, (60, 7)):  # 決まった乱数の初期値（毎回同じ）
        q, ok, _, _ = ik.solve_position(np.asarray(xyz, float), seed, max_iter=300)
        if ok and ik.within_limits(q):
            best = q
            break
    assert best is not None, f"{side} が {xyz} に届かない"
    return np.rad2deg(best).tolist()


def plan_at(side, rest, q):
    return {"arm": side, "q_start_deg": rest, "q_pre_deg": q, "q_grasp_deg": q, "q_lift_deg": q}


@unittest.skipUnless(HAVE_PIN, "pinocchio の入った venv で走らせる")
class LaneTest(unittest.TestCase):
    def setUp(self):
        os.environ.pop("GRASP_TIMING_LOG", None)
        from grasp import dual
        self.dual = dual

    def test_own_sides_are_ok(self):
        """左腕は左の物・右腕は右の物（中央から 25 cm ずつ）: 同時に動かしてよい。"""
        pl = plan_at("left", REST_L, reach("left", [0.35, 0.25, 0.47]))
        pr = plan_at("right", REST_R, reach("right", [0.35, -0.25, 0.47]))
        ok, gap = self.dual.lanes_ok(pl, pr, gap=0.15)
        self.assertTrue(ok, f"gap {gap:.3f}")
        self.assertGreaterEqual(gap, 0.15)

    def test_both_reach_middle_is_refused(self):
        """両腕が中央のすぐ脇へ行く（この高さで左右が届くいちばん中央寄り・y ±5 cm）: 同時には動かさない。"""
        pl = plan_at("left", REST_L, reach("left", [0.35, 0.05, 0.47]))
        pr = plan_at("right", REST_R, reach("right", [0.35, -0.05, 0.47]))
        ok, gap = self.dual.lanes_ok(pl, pr, gap=0.15)
        self.assertFalse(ok, f"gap {gap:.3f}")
        self.assertLess(gap, 0.15)

    def test_place_crossing_the_middle_is_refused(self):
        """掴む所は離れていても、左腕の置き先（運ぶ姿勢・置く姿勢）が右腕の通り道に入るなら同時には動かさない。"""
        pl = plan_at("left", REST_L, reach("left", [0.35, 0.25, 0.47]))
        q_mid = reach("left", [0.35, 0.05, 0.47])
        pl.update(q_carry_deg=q_mid, q_place_deg=q_mid)
        pr = plan_at("right", REST_R, reach("right", [0.35, -0.05, 0.47]))
        ok, _ = self.dual.lanes_ok(pl, pr, gap=0.15)
        self.assertFalse(ok)

    def test_wrong_arm_labels_are_refused(self):
        """計画の腕が左右そろっていない（両方左など）なら同時には動かさない。"""
        pl = plan_at("left", REST_L, reach("left", [0.35, 0.25, 0.47]))
        ok, _ = self.dual.lanes_ok(pl, dict(pl), gap=0.15)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
