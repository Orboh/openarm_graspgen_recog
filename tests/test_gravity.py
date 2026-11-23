"""重力のフィードフォワードの URDF（grasp/gravity.py・grasp/tools/make_grav_urdf.py）。実機なし。

    PYTHONPATH=.:$HOME/projects/lerobot/src ~/projects/lerobot/.venv/bin/python tests/test_gravity.py  # lerobot の venv は tests が別の物に当たるのでファイルで
"""
import os
import unittest
from unittest import mock

from grasp import gravity
from grasp.tools import make_grav_urdf

BASE = os.path.join(os.path.dirname(__file__), "..", "urdf", "openarm_v10_bimanual.urdf")
POSE = [-40.1, -24.1, 62.2, 93.7, -47.3, 38.7, 76.5]  # hold_test の姿勢 1（2026-11-23）


def _pin_model(urdf, side):
    import pinocchio as pin
    full = pin.buildModelFromUrdf(urdf)
    arm = [f"openarm_{side}_joint{i}" for i in range(1, 8)]
    for n in full.names:  # OpenArmGravityModel と同じく指は 0 kg
        if n.endswith(("_finger_joint1", "_finger_joint2")):
            full.inertias[full.getJointId(n)] = pin.Inertia.Zero()
    return pin.buildReducedModel(full, [j for j in range(1, full.njoints) if full.names[j] not in arm], pin.neutral(full))


class TestGravUrdf(unittest.TestCase):
    def test_link7_carries_hand_and_jaws(self):
        """link7 に固定した質量が、元の URDF より手と爪の分（0.48 kg）だけ重い。左右とも。"""
        for side in ("left", "right"):
            m0, m1 = _pin_model(BASE, side), _pin_model(gravity.GRAV_URDF, side)
            jid = m0.getJointId(f"openarm_{side}_joint7")
            self.assertAlmostEqual(m1.inertias[jid].mass - m0.inertias[jid].mass,
                                   make_grav_urdf.HAND_MASS + make_grav_urdf.JAW_MASS, places=6)

    def test_urdf_is_up_to_date(self):
        """置いてある URDF が make_grav_urdf.py の値と同じ（値を直したら作り直す）。"""
        import xml.etree.ElementTree as ET
        root = ET.parse(gravity.GRAV_URDF).getroot()
        hand = root.find("link[@name='openarm_left_hand']/inertial/mass")
        jaw = root.find("link[@name='openarm_left_jaws_mass']/inertial/mass")
        self.assertEqual(float(hand.get("value")), make_grav_urdf.HAND_MASS)
        self.assertEqual(float(jaw.get("value")), make_grav_urdf.JAW_MASS)

    def test_torque_fn_uses_grav_urdf(self):
        """gravity_torque_fn は手と爪の分だけ大きいトルクを返す（姿勢 1 で J1 が約 1.3 Nm 増える）。"""
        from lerobot.motors.damiao.openarm_gravity import OpenArmGravityModel
        q = {f"joint_{i + 1}": v for i, v in enumerate(POSE)}
        t0 = OpenArmGravityModel(BASE, "left").torque(q)
        t1 = gravity.gravity_torque_fn("left")(q)
        self.assertAlmostEqual(t1["joint_1"] - t0["joint_1"], -1.33, delta=0.05)
        self.assertGreater(abs(t1["joint_4"]), abs(t0["joint_4"]))

    def test_missing_urdf_stops(self):
        """GRASP_GRAV_URDF が無いファイルなら動かす前に止める（手 0 kg のモデルに黙って戻らない）。"""
        with mock.patch.dict(os.environ, {"GRASP_GRAV_URDF": "/nonexistent/grav.urdf"}):
            with self.assertRaises(SystemExit):
                gravity.grav_urdf_path()


if __name__ == "__main__":
    unittest.main()
