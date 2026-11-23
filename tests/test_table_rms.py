"""grasp/tools/table_rms.py の平面の当てはめと残差（実機・カメラなし）。

    PYTHONPATH=. python3 -m unittest tests.test_table_rms -q
"""
import unittest

import numpy as np

from grasp.tools import table_rms as tr

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])


def plane_depth(sigma=0.001, seed=0):
    """カメラから 0.5 m・少し傾いた机の深度（σ の雑音つき）と、上の 1/5 に机から 1 cm 手前の別の面（奥の棚）。"""
    rng = np.random.default_rng(seed)
    v, u = np.mgrid[0:480, 0:640]
    ray = np.stack([(u - 320) / 600.0, (v - 240) / 600.0, np.ones_like(u, float)], -1)
    n = np.array([0.0, -0.5, -1.0]); n /= np.linalg.norm(n)   # 机の法線（カメラ座標）
    d0 = 0.5
    z = d0 / -(ray @ n)
    z = z + rng.normal(0, sigma, z.shape)
    z[:90] -= 0.010                                       # 棚: 机とほぼ平行で 1 cm 手前
    z[90:100] = 0                                         # 棚と机の間の深度の抜け
    return z.astype(np.float32), n


class TableRmsTest(unittest.TestCase):
    def test_rms_and_shelf_excluded(self):
        depth, n_true = plane_depth(sigma=0.001)
        pts, vu = tr.backproject(depth, K, 0.15, 2.0)
        n, d, inl = tr.fit_plane(pts, 0.015)
        inl = tr.largest_region(depth.shape, vu, inl)
        n, d, _ = tr.fit_plane(pts[inl], 0.015)
        inl = tr.largest_region(depth.shape, vu, np.abs(pts @ n + d) < 0.015)
        self.assertFalse((vu[0][inl] < 90).any(), "棚の画素が机に入った")
        res = pts[inl] @ n + d
        rms = np.sqrt(np.mean(res ** 2))
        self.assertAlmostEqual(abs(n @ n_true), 1.0, places=4)
        self.assertGreater(rms, 0.0007)   # 深度の σ 1 mm → 面の法線方向では cos 分だけ小さい
        self.assertLess(rms, 0.0012)

    def test_up_constraint_rejects_wrong_plane(self):
        depth, n_true = plane_depth(sigma=0.0005)
        pts, _ = tr.backproject(depth, K, 0.15, 2.0)
        wrong_up = np.cross(n_true, [1.0, 0, 0]); wrong_up /= np.linalg.norm(wrong_up)  # 机に垂直な向き
        with self.assertRaises(RuntimeError):
            tr.fit_plane(pts, 0.015, up_cam=wrong_up, max_tilt_deg=20)


if __name__ == "__main__":
    unittest.main()
