"""grasp/depth.py の挟む深さ（実機・カメラなし）。

    PYTHONPATH=. python3 -m unittest tests.test_depth -q
"""
import unittest

import numpy as np

from grasp.depth import grasp_depth

TIP = 0.14
HALF = 0.04


def cylinder_across(depth_from_tip, n=3000, r=0.025, half_circle=True, seed=0):
    """爪の座標で、爪の間に横たわる直径 5 cm の円柱（軸は爪の厚みの向き）の面の点。中心は z = TIP - depth_from_tip。
    half_circle なら手首寄りの半周だけ（胸のカメラは片側しか見ない）。"""
    rng = np.random.default_rng(seed)
    th = rng.uniform(np.pi, 2 * np.pi, n) if half_circle else rng.uniform(0, 2 * np.pi, n)
    y = rng.uniform(-0.03, 0.03, n)
    return np.c_[r * np.cos(th), y, TIP - depth_from_tip + r * np.sin(th)]


class GraspDepthTest(unittest.TestCase):
    def test_contact_at_cylinder_centre(self):
        """爪が当たるのは円柱の一番太い所（中心の深さ）。片側しか見えなくても同じ。"""
        for d in (0.030, 0.045):
            for half in (True, False):
                got = grasp_depth(np.eye(4), cylinder_across(d, half_circle=half), HALF, TIP)
                # 手首寄りの半周しか見えないと、太い所のうち手首側の区間だけで測るので少し深く出る（5 mm 以内）
                self.assertTrue(d - 0.003 <= got <= d + (0.005 if half else 0.003), (d, half, got))

    def test_centre_past_tip_is_shallow(self):
        """円柱の中心が指先より先なら、当たる所は指先の近く（2026-11-22 の 18:11 の抜けた回の形）。"""
        self.assertLess(grasp_depth(np.eye(4), cylinder_across(-0.005, half_circle=False), HALF, TIP), 0.012)

    def test_tall_object_from_above(self):
        """幅の一定な縦長の物を上から深く挟むと、爪の面に入っている範囲の真ん中。"""
        rng = np.random.default_rng(1)
        z = rng.uniform(TIP - 0.050, TIP + 0.050, 4000)  # 指先から 50 mm 奥まで物の上の端がある
        x = rng.choice([-0.025, 0.025], 4000)
        P = np.c_[x, rng.uniform(-0.01, 0.01, 4000), z]
        self.assertAlmostEqual(grasp_depth(np.eye(4), P, HALF, TIP), 0.025, delta=0.006)

    def test_pose_is_applied(self):
        """爪を回して動かしても、物を同じだけ動かせば同じ深さ。"""
        P = cylinder_across(0.040)
        a = np.deg2rad(35)
        G = np.eye(4)
        G[:3, :3] = [[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]]
        G[:3, 3] = [0.3, -0.1, 0.4]
        Pw = P @ G[:3, :3].T + G[:3, 3]
        self.assertAlmostEqual(grasp_depth(G, Pw, HALF, TIP), grasp_depth(np.eye(4), P, HALF, TIP), places=9)

    def test_object_outside_jaws_is_zero(self):
        """物が指先より前（爪の間に入らない）なら 0。"""
        self.assertEqual(grasp_depth(np.eye(4), cylinder_across(-0.06), HALF, TIP), 0.0)


if __name__ == "__main__":
    unittest.main()
