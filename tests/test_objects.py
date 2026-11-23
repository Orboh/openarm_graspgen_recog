"""grasp/objects.py の対象の選び方（実機・カメラなし）。

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python -m unittest discover -s tests -q
"""
import unittest

import numpy as np

from grasp import objects

K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])
BROWN = (60, 120, 180)  # BGR。HSV で H≈15・S≈170（木片の色）
GRAY = (90, 90, 90)


def box(center, size, n=400, seed=0):
    """中心と大きさ [m] の箱の中の点（カメラの座標）。"""
    rng = np.random.default_rng(seed)
    return (np.asarray(center) + (rng.random((n, 3)) - 0.5) * np.asarray(size)).astype(np.float32)


def paint(img, pts, bgr):
    """点の写る画素の周りを塗る（色の画像を作る）。"""
    uv = pts @ K.T
    uv = np.round(uv[:, :2] / uv[:, 2:]).astype(int)
    for u, v in uv:
        img[max(v - 3, 0):v + 4, max(u - 3, 0):u + 4] = bgr


class PickTest(unittest.TestCase):
    def setUp(self):
        self.img = np.full((480, 640, 3), 40, np.uint8)  # 暗いマット
        self.big_brown = box([0.0, -0.15, 0.6], [0.26, 0.12, 0.09], n=1300)   # 奥の段ボール（色は木片・大きすぎる）
        self.gray = box([0.12, 0.10, 0.35], [0.06, 0.05, 0.03], n=300, seed=1)  # 灰色の部品
        self.log = box([-0.03, 0.07, 0.30], [0.10, 0.05, 0.05], n=600, seed=2)  # 手前の木片
        for pts, c in ((self.big_brown, BROWN), (self.gray, GRAY), (self.log, BROWN)):
            paint(self.img, pts, c)
        self.clusters = [self.big_brown, self.log, self.gray]  # 大きい順（segment_clusters と同じ）

    def test_wood_picks_the_log_not_the_biggest(self):
        best, table = objects.pick(self.clusters, K, self.img, objects.load("wood"))
        self.assertEqual(best, 1)
        self.assertTrue(table[0]["color_ok"])
        self.assertFalse(table[0]["size_ok"])  # 段ボールは大きさで外れる

    def test_missing_target_returns_none(self):
        best, table = objects.pick(self.clusters, K, self.img, objects.load("alframe"))
        self.assertIsNone(best)
        self.assertIn("target alframe not found", objects.not_found(objects.load("alframe"), table))

    def test_unknown_name_stops(self):
        with self.assertRaises(SystemExit):
            objects.load("no_such_object")


if __name__ == "__main__":
    unittest.main()
