"""右腕を足したときの決まり（grasp/arm.py・plan.py の腕の選び方・cuRobo の鏡写し）。実機・カメラ・GPU なし。

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python -m unittest tests.test_arm -q
"""
import os
import unittest
from unittest import mock

import numpy as np

from grasp import arm

URDF = os.path.join(os.path.dirname(__file__), "..", "urdf", "openarm_v10_bimanual.urdf")


def _quat_to_R(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class MirrorTest(unittest.TestCase):
    def test_all_links_mirror(self):
        """q_右 = MIRROR_SIGN * q_左 で、左右の同じ名前の link が T_右 = M T_左 M になる（link4 だけ肘の軸に沿って 63 mm ずれる）。"""
        import pinocchio as pin
        full = pin.buildModelFromUrdf(URDF)
        d = full.createData()
        rng = np.random.default_rng(0)
        for _ in range(5):
            qL = rng.uniform(-0.6, 0.6, 7) + [0, 0, 0, 1, 0, 0, 0]
            q = pin.neutral(full)
            for i in range(7):
                q[full.joints[full.getJointId(f"openarm_left_joint{i + 1}")].idx_q] = qL[i]
                q[full.joints[full.getJointId(f"openarm_right_joint{i + 1}")].idx_q] = arm.MIRROR_SIGN[i] * qL[i]
            pin.forwardKinematics(full, d, q)
            pin.updateFramePlacements(full, d)
            for k in range(1, 8):
                a = d.oMf[full.getFrameId(f"openarm_left_link{k}")].homogeneous
                b = d.oMf[full.getFrameId(f"openarm_right_link{k}")].homogeneous
                D = np.eye(4)
                if k == 4:  # link4 の枠だけ、右は肘の軸（自分の y）に沿って 63 mm ずれた所に置いてある（動きは同じ）
                    D[1, 3] = 0.063
                np.testing.assert_allclose(arm.mirror_T(a) @ D, b, atol=1e-9)
            for fr in ("hand_tcp", "link7"):
                a = d.oMf[full.getFrameId(f"openarm_left_{fr}")].homogeneous
                b = d.oMf[full.getFrameId(f"openarm_right_{fr}")].homogeneous
                np.testing.assert_allclose(arm.mirror_T(a), b, atol=1e-9)

    def test_right_ik_solves_mirrored_target(self):
        """右腕の IK が、左の解の鏡写しの姿勢を解ける（左右で同じ IK の作り）。"""
        import pinocchio as pin
        from ik.openarm_left_ik import OpenArmIK
        L, R = OpenArmIK(URDF, side="left"), OpenArmIK(URDF, side="right")
        qL = np.deg2rad([-17.7, 2.0, 4.7, 28.1, -8.3, -15.5, -28.9])
        T = arm.mirror_T(L.fk(qL).homogeneous)
        q, ok, _, _ = R.solve(pin.SE3(T[:3, :3], T[:3, 3]), arm.mirror_q(qL) + 0.1, position_only=False)
        self.assertTrue(ok)
        np.testing.assert_allclose(R.fk(q).homogeneous, T, atol=1e-3)
        self.assertTrue(R.within_limits(arm.mirror_q(qL)))  # 左の好む姿勢の鏡写しは右の可動域の中

    def test_mirror_pose_matches_mirror_T(self):
        rng = np.random.default_rng(1)
        q = rng.normal(size=4); q /= np.linalg.norm(q)
        pose = [0.1, 0.2, 0.3, *q]
        T = np.eye(4); T[:3, :3] = _quat_to_R(q); T[:3, 3] = pose[:3]
        m = arm.mirror_pose_wxyz(pose)
        Tm = np.eye(4); Tm[:3, :3] = _quat_to_R(m[3:]); Tm[:3, 3] = m[:3]
        np.testing.assert_allclose(Tm, arm.mirror_T(T), atol=1e-12)

    def test_right_env_does_not_fall_back_to_left(self):
        with mock.patch.dict(os.environ, {"GRASP_CAN_PORT": "can_lead_l"}, clear=False):
            os.environ.pop("GRASP_CAN_PORT_RIGHT", None)
            self.assertEqual(arm.env("GRASP_CAN_PORT", "left"), "can_lead_l")
            self.assertIsNone(arm.env("GRASP_CAN_PORT", "right"))


class MirroredArmTest(unittest.TestCase):
    def test_spheres_and_link7(self):
        """右腕の球 = 左を q_左 = MIRROR_SIGN*q_右 にした球の y を反転。link7 は M T M。"""
        from grasp.curobo.model import MirroredArm

        class FakeLeft:
            sphere_links = ["openarm_left_link1"]

            def spheres(self, q):
                q = np.atleast_2d(q)
                return np.stack([q[:, 0], q[:, 1], q[:, 2], np.full(len(q), 0.05)], -1)[:, None, :]

            def link7_pose(self, q):
                q = np.atleast_2d(q)
                T = np.tile(np.eye(4), (len(q), 1, 1)); T[:, :3, 3] = q[:, :3]
                return T

        m = MirroredArm(FakeLeft())
        qR = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]])
        S = m.spheres(qR)
        np.testing.assert_allclose(S[0, 0], [-0.1, 0.2, -0.3, 0.05])  # 左の q = (-0.1, -0.2, -0.3, ...)、y を反転
        T = m.link7_pose(qR)
        np.testing.assert_allclose(T[0, :3, 3], [-0.1, 0.2, -0.3])


class RightSphereTest(unittest.TestCase):
    def test_right_center_lands_on_urdf_right_link(self):
        """右用に置き直した中心を左の運動学で動かして y 反転すると、URDF の右腕の link の同じ局所点に来る（link4 のずれも含む）。"""
        from grasp.curobo.model import right_local_center
        from grasp.urdf_fk import LeftArmFK
        L, R = LeftArmFK(URDF, "left"), LeftArmFK(URDF, "right")
        rng = np.random.default_rng(2)
        for _ in range(3):
            qL = rng.uniform(-0.5, 0.5, 7) + [0, 0, 0, 1, 0, 0, 0]
            PL, PR = L.link_poses(qL), R.link_poses(arm.mirror_q(qL))
            for k in range(8):
                c = rng.normal(scale=0.05, size=3)
                cp = right_local_center(f"openarm_left_link{k}", c)
                got = arm.mirror_points(PL[f"openarm_left_link{k}"][:3, :3] @ cp + PL[f"openarm_left_link{k}"][:3, 3])
                want = PR[f"openarm_right_link{k}"][:3, :3] @ c + PR[f"openarm_right_link{k}"][:3, 3]
                np.testing.assert_allclose(got, want, atol=1e-9, err_msg=f"link{k}")

    def test_gripper_is_mirrored(self):
        from grasp.curobo.model import right_local_center
        np.testing.assert_allclose(right_local_center("openarm_left_gripper_body", [0.01, 0.02, 0.1]), [0.01, 0.02, 0.1])


class OtherArmObstacleTest(unittest.TestCase):
    def test_sphere_distance_and_ignore(self):
        """動かない球の集まりまでの距離（表面どうし）と、ignore_links の球を外すこと。"""
        from grasp.curobo.model import obstacle_distances, sphere_spheres_distance
        O = np.array([[0.0, 0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.2]])
        S = np.array([[[0.2, 0.0, 0.0, 0.05], [0.6, 0.0, 0.0, 0.1]]])  # (1, 2, 4)
        np.testing.assert_allclose(sphere_spheres_distance(S, O), [[0.10, 0.10]], atol=1e-6)
        world = {"cuboid": [], "spheres": [{"name": "other_arm", "spheres": O, "ignore_links": ["openarm_left_link0"]}]}
        D = obstacle_distances(S, world, None, ["openarm_left_link0", "openarm_left_link3"])["other_arm"]
        self.assertTrue(np.isinf(D[0, 0]))
        self.assertAlmostEqual(float(D[0, 1]), 0.10, places=5)

    def test_mirror_world_mirrors_other_arm(self):
        from grasp.curobo.model import mirror_world
        w = {"cuboid": [], "mesh": [], "spheres": [{"name": "other_arm", "spheres": np.array([[0.1, 0.2, 0.3, 0.05]]),
                                                    "boxes": [{"name": "b", "dims": [1, 1, 1], "pose": [0, 0.2, 0, 1, 0, 0, 0]}]}]}
        m = mirror_world(w)["spheres"][0]
        np.testing.assert_allclose(m["spheres"], [[0.1, -0.2, 0.3, 0.05]])
        self.assertAlmostEqual(m["boxes"][0]["pose"][1], -0.2)


class ChooseArmTest(unittest.TestCase):
    def r(self, side, score, travel):
        return {"arm": side, "score": score, "q_start_deg": [0] * 7, "q_pre_deg": [travel] + [0] * 6}

    def test_higher_score_wins(self):
        from grasp.plan import choose_arm
        got = choose_arm({"left": self.r("left", 0.60, 10), "right": self.r("right", 0.80, 90)}, tie=0.05)
        self.assertEqual(got["arm"], "right")

    def test_tie_goes_to_smaller_motion(self):
        from grasp.plan import choose_arm
        got = choose_arm({"left": self.r("left", 0.80, 70), "right": self.r("right", 0.78, 20)}, tie=0.05)
        self.assertEqual(got["arm"], "right")

    def test_one_side_missing(self):
        from grasp.plan import choose_arm
        self.assertEqual(choose_arm({"left": None, "right": self.r("right", 0.5, 5)})["arm"], "right")
        self.assertIsNone(choose_arm({"left": None, "right": None}))


class ClientTest(unittest.TestCase):
    def test_right_without_right_server_stops(self):
        from grasp.curobo import client
        with mock.patch.dict(os.environ, {"GRASP_CUROBO": "localhost:5558"}, clear=False):
            os.environ.pop("GRASP_CUROBO_RIGHT", None)
            with self.assertRaises(SystemExit):
                client.set_arm("right")
            client.set_arm("left")
            self.assertEqual(client.ADDR, "localhost:5558")


if __name__ == "__main__":
    unittest.main()
