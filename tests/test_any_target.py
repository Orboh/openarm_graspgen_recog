"""どんな物でも拾うための部品を確かめる（2026-11-23・作業者: Munenori）。腕もカメラも使わない。

- 形で選ぶ（--target any・grasp/objects.py の "mode": "shape"）: 色を見ず、机からの高さと机に沿った大きさで選ぶ。
  寝ているケーブル（低い）・爪に入らない大きな物は選ばない
- 机に沿った外形（grasp/objects.py の footprint）: 丸い物は縦横がそろい、長い物は長い辺が出る（置く向きを一回り試すかの判断に使う）
- 長い物の厚みの向き（grasp/wrist_servo.py の t_covered）: 爪の幅が全部物の上に乗っていれば、画面の端で切れていても直さない
"""
from __future__ import annotations

import unittest

import numpy as np

from grasp import objects

PLANE = [0.0, 0.0, 1.0, 0.0]  # 机の面: 高さ = z（カメラの座標の代わり）


def box(center, size, n=2000, seed=0):
    """中心・大きさ (x, y, z) の箱の中の点（底は机の上）。"""
    rng = np.random.default_rng(seed)
    c = np.asarray(center, float); s = np.asarray(size, float)
    p = rng.random((n, 3)) * s + (c - s / 2)
    p[:, 2] = rng.random(n) * s[2]
    return p.astype(np.float32)


def cylinder(center_xy, r, h, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.random(n) * 2 * np.pi; rr = r * np.sqrt(rng.random(n))
    return np.c_[center_xy[0] + rr * np.cos(a), center_xy[1] + rr * np.sin(a), rng.random(n) * h].astype(np.float32)


class ShapePickTest(unittest.TestCase):
    def setUp(self):
        self.spec = objects.load("any")
        self.K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
        self.img = np.zeros((480, 640, 3), np.uint8)

    def pick(self, clusters):
        # 机の面を z にした仮の座標なので、カメラへの写り方（画面の端の判定）は切る。画面の端は test_skips_cluster_at_image_border で見る
        return objects.pick(clusters, self.K, self.img, {**self.spec, "border_px": 0}, plane=PLANE)

    def test_skips_lying_cable(self):
        """寝ているケーブル（高さ 8 mm）は選ばず、立てた丸太を選ぶ。"""
        cable = box([0.3, 0.0, 0], [0.40, 0.01, 0.008], n=4000)
        wood = cylinder([0.3, -0.1], 0.025, 0.09, n=1500, seed=2)
        best, table = self.pick([cable, wood])
        self.assertEqual(best, 1, table)
        self.assertFalse(table[0]["ok"])

    def test_wide_objects_ok(self):
        """全体が爪より太い物（テープの輪・上の開いた箱）も選ぶ。縁や壁が掴めるかは GraspGenX と確かめに任せる（2026-11-23）。"""
        ring = box([0.3, 0.0, 0], [0.126, 0.116, 0.048], n=2000)
        best, table = self.pick([ring])
        self.assertEqual(best, 0, table)

    def test_skips_cluster_at_image_border(self):
        """画面の端に掛かるかたまり（画面の外から続くケーブルなど）は選ばない。"""
        rng = np.random.default_rng(3)  # カメラの座標（z が奥）: 机は z = 0.55、物は机から 2〜5 cm 手前
        cam = np.c_[rng.uniform(-0.025, 0.025, 800), rng.uniform(-0.025, 0.025, 800), rng.uniform(0.50, 0.53, 800)]
        edge = cam.copy(); edge[:, 0] += 0.39  # 右の端（u ≈ 320 + 400·0.39/0.5 ≈ 632〜）
        best, table = objects.pick([edge, cam], self.K, self.img, self.spec, plane=[0.0, 0.0, -1.0, 0.55])
        self.assertEqual(best, 1, table)
        self.assertIn("画面の端", table[0]["why"])

    def test_long_bar_ok(self):
        """寝かせたアルミのバー（27 × 3 × 3 cm）は、長い辺が長くても短い辺が爪に入るので選ぶ。"""
        bar = box([0.3, 0.0, 0], [0.27, 0.03, 0.03], n=3000)
        best, table = self.pick([bar])
        self.assertEqual(best, 0, table)

    def test_none_when_nothing_fits(self):
        """当てはまる物が無ければ選ばない（いちばん大きいかたまりに戻らない）。"""
        best, _ = self.pick([box([0.3, 0.0, 0], [0.40, 0.01, 0.008])])
        self.assertIsNone(best)


class JawClearanceTest(unittest.TestCase):
    """開いた爪と物のすき間は、爪の板の近くの点だけで見る（輪の向こう側の壁などは数えない・2026-11-23）。"""

    def test_far_wall_ignored(self):
        from grasp.depth import jaw_clearance
        rng = np.random.default_rng(0)
        wall = np.c_[rng.uniform(-0.015, 0.015, 500), rng.uniform(-0.01, 0.01, 500), rng.uniform(-0.05, 0.0, 500)]
        far = wall + [0.10, 0, 0]  # 爪の外 10 cm（輪の向こう側）
        self.assertAlmostEqual(jaw_clearance(np.vstack([wall, far]), half=0.04), 0.025, delta=0.002)

    def test_point_on_finger_blocks(self):
        from grasp.depth import jaw_clearance
        on_finger = np.array([[0.05, 0.0, -0.02]] * 30)  # 爪の板（内側の面 40 mm の外 1 cm）に物
        self.assertLess(jaw_clearance(on_finger, half=0.04), 0)


class FitsBetweenJawsTest(unittest.TestCase):
    def test_wood_fits_ring_does_not(self):
        from grasp.wrist_approach import fits_between_jaws
        self.assertTrue(fits_between_jaws(cylinder([0.3, 0.1], 0.025, 0.09)))
        self.assertTrue(fits_between_jaws(box([0.3, 0.0, 0], [0.27, 0.03, 0.03])))  # 長いバー（短い辺 3 cm）
        self.assertFalse(fits_between_jaws(box([0.3, 0.0, 0], [0.126, 0.116, 0.048])))  # テープの輪


class FootprintTest(unittest.TestCase):
    def test_round_and_long(self):
        long_, short = objects.footprint(cylinder([0.3, 0.1], 0.025, 0.09), up=[0, 0, 1])
        self.assertLess(long_ / short, 1.3)
        long_, short = objects.footprint(box([0.3, 0.0, 0], [0.27, 0.03, 0.03]), up=[0, 0, 1])
        self.assertGreater(long_ / short, 5)
        self.assertAlmostEqual(long_, 0.27, delta=0.02)


class TCoveredTest(unittest.TestCase):
    def test_long_object_covers_pad(self):
        from grasp.wrist_servo import t_covered
        t = np.linspace(-0.08, 0.06, 200)  # 長い物（画面の端で切れて見える）
        self.assertTrue(t_covered(t, bias_t=0.0))
        self.assertTrue(t_covered(t, bias_t=0.02))   # 目標を 2 cm 寄せても、まだ爪の幅は物の上

    def test_short_or_off_center_not_covered(self):
        from grasp.wrist_servo import t_covered
        self.assertFalse(t_covered(np.linspace(-0.025, 0.025, 100), bias_t=0.02))  # 丸太（直径 5 cm）を 2 cm 寄せる → 乗りきらない
        self.assertFalse(t_covered(np.linspace(0.0, 0.10, 100), bias_t=0.0))      # 物が片側にしか無い


class SelfFingersTest(unittest.TestCase):
    """腕の点を消す線に、左右の爪の板（全開の位置）も入る（2026-11-23: 開いた外側の爪が消えずに、ケーブルとつながって選ばれた）。"""

    def test_outer_finger_point_is_near_a_segment(self):
        try:
            from grasp.tools.thor_snap import self_segments
            from grasp import plan as P
            from ik.openarm_left_ik import OpenArmIK
        except Exception as e:  # noqa: BLE001  pinocchio の無い venv
            self.skipTest(str(e))
        q = [11.0, -4.4, -3.4, 16.9, -0.6, 2.0, -80.0]
        segs = self_segments(q, side="left")
        ik = OpenArmIK(side="left")
        import pinocchio as pin
        pin.forwardKinematics(ik.model, ik.data, np.deg2rad(q))
        Tg = P.t_l7_g("left")
        MG = ik.data.oMi[ik.model.getJointId("openarm_left_joint7")].act(pin.SE3(Tg[:3, :3], Tg[:3, 3]))
        for sx in (-1, 1):  # 全開の爪の板の真ん中（閉じる向き ±45 mm・指先から 4 cm 奥）
            p = MG.act(np.array([sx * 0.045, 0.0, P.CFG["flange_to_tip_m"] - 0.04]))
            d = min(np.linalg.norm(p - (p0 + np.clip((p - p0) @ (p1 - p0) / max((p1 - p0) @ (p1 - p0), 1e-9), 0, 1) * (p1 - p0)))
                    for p0, p1 in segs)
            self.assertLess(d, 0.01)


class PlaceOnSurfaceTest(unittest.TestCase):
    """青い重量計の上に置く（2026-11-23）: 置き場の色のかたまりは拾わない・板の上の置き先の候補。"""

    def test_scale_is_not_picked(self):
        """--target any は、置き場（blue_scale）の色のかたまりを拾う物にしない。"""
        import cv2
        K = np.array([[400.0, 0, 320], [0, 400.0, 240], [0, 0, 1]])
        rng = np.random.default_rng(4)
        scale = np.c_[rng.uniform(0.02, 0.12, 3000), rng.uniform(-0.05, 0.05, 3000), rng.uniform(0.49, 0.53, 3000)]
        wood = np.c_[rng.uniform(-0.12, -0.08, 800), rng.uniform(-0.02, 0.02, 800), rng.uniform(0.46, 0.53, 800)]
        img = np.zeros((480, 640, 3), np.uint8)
        hsv = np.zeros_like(img); hsv[..., 0] = 100; hsv[..., 1] = 250; hsv[..., 2] = 130  # 青
        blue = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        uv = scale @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
        img[uv[:, 1].min():uv[:, 1].max() + 1, uv[:, 0].min():uv[:, 0].max() + 1] = blue[0, 0]
        spec = {**objects.load("any"), "border_px": 0}
        best, table = objects.pick([scale, wood], K, img, spec, plane=[0.0, 0.0, -1.0, 0.55])
        self.assertEqual(best, 1, table)
        self.assertIn("置き場", table[0]["why"])

    def test_surface_spots_inside_and_near_first(self):
        from grasp.tools.resident_plan import surface_spots
        rng = np.random.default_rng(5)
        P = np.c_[rng.uniform(0.26, 0.47, 3000), rng.uniform(-0.16, 0.01, 3000), np.full(3000, 0.285)]
        spots = surface_spots(P, near_xy=(0.22, 0.10), inset=0.04, step=0.03)
        self.assertGreater(len(spots), 4)
        S = np.array(spots)
        self.assertTrue(np.all((S[:, 0] >= 0.26 + 0.04 - 1e-6) & (S[:, 0] <= 0.47 - 0.04 + 1e-6)))
        self.assertTrue(np.all((S[:, 1] >= -0.16 + 0.04 - 1e-6) & (S[:, 1] <= 0.01 - 0.04 + 1e-6)))
        d = np.hypot(S[:, 0] - 0.22, S[:, 1] - 0.10)
        self.assertTrue(np.all(np.diff(d) >= -1e-9))  # 物に近い順


if __name__ == "__main__":
    unittest.main()
