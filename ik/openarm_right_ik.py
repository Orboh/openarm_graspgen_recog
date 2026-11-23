#!/usr/bin/env python3
"""OpenArm right-arm IK: the left-arm module with the right arm's joints (2026-11-21, Sota: 右腕を使う).
Same bimanual URDF and base frame as OpenArmLeftIK, so T_base_cam and plans share coordinates."""
from __future__ import annotations

import numpy as np
import pinocchio as pin

from ik.openarm_left_ik import DEFAULT_URDF, OpenArmLeftIK

RIGHT_ARM_JOINTS = [f"openarm_right_joint{i}" for i in range(1, 8)]
TCP_FRAME_R = "openarm_right_hand_tcp"


class OpenArmRightIK(OpenArmLeftIK):
    def __init__(self, urdf_path: str = DEFAULT_URDF, ee_frame: str = TCP_FRAME_R):  # noqa: D107
        full = pin.buildModelFromUrdf(urdf_path)
        lock = [j for j in range(1, full.njoints) if full.names[j] not in RIGHT_ARM_JOINTS]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        if self.model.existFrame(ee_frame):
            self.ee_fid = self.model.getFrameId(ee_frame); self.use_frame = True; self.ee_name = ee_frame
        else:
            self.ee_jid = self.model.getJointId("openarm_right_joint7"); self.use_frame = False
            self.ee_name = "openarm_right_joint7(fallback)"
        self.lo = np.where(np.isfinite(self.model.lowerPositionLimit), self.model.lowerPositionLimit, -np.pi)
        self.hi = np.where(np.isfinite(self.model.upperPositionLimit), self.model.upperPositionLimit, np.pi)
        # 2026-11-21 21:24: the real right arm sits 4.1 deg / 4.6 cm off the URDF (chest camera + marker on the right
        # hand: 15 mm median error as modelled, 1.7 mm with this correction). actual = C * model.
        self.C = pin.SE3.Identity()
        import json
        import os
        cp = os.environ.get("GRASP_RIGHT_CORR", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                              "grasp", "config", "right_arm_base_corr.json"))
        if os.path.exists(cp) and os.environ.get("GRASP_RIGHT_CORR", "") != "none":
            c = json.load(open(cp))
            from scipy.spatial.transform import Rotation
            self.C = pin.SE3(Rotation.from_rotvec(c["rotvec"]).as_matrix(), np.array(c["t_m"]))
            self.corr_path = cp

    def fk(self, q) -> pin.SE3:
        return self.C * OpenArmLeftIK.fk(self, q)
    # solve() is inherited: it calls self.fk (corrected) and the LOCAL-frame Jacobian, which a fixed left factor C
    # does not change, so the full-pose solve lands on the corrected target.


def arm_ik_class():
    """OpenArmLeftIK or OpenArmRightIK, by GRASP_ARM (default left)."""
    import os
    return OpenArmRightIK if os.environ.get("GRASP_ARM", "left") == "right" else OpenArmLeftIK


def mirror_q(q_deg):
    """Left-arm posture (lerobot deg) -> the mirror-image right-arm posture. Sign pattern from the right arm resting in
    the mirror of the left home on 11/21 (left 53.9,0,-4,118.7,-1,0.2,-33 / right -56.1,-0.8,6.6,106.1,6.9,6.0,48.2)."""
    s = [-1, -1, -1, 1, -1, 1, -1]
    return [v * k for v, k in zip(q_deg, s)]
