#!/usr/bin/env python3
"""Build a URDF with the real left gripper (so101 fin-ray finger x2, was UMI finger) attached (urdf/openarm_v10_bimanual_umi.urdf).

The original urdf/openarm_v10_bimanual.urdf is not modified. For the left arm:
  - drops the stock finger visuals/collisions (the joints stay, for compatibility)
  - adds openarm_left_umi_finger_{l,r} (prismatic, parent link7, 0 = closed to 0.055 m = 110 mm open, mimic)
  - adds the openarm_left_umi_tcp frame (centre of the grasp region; IK target)

Geometry (2026-11-20, measured on the real arm):
  - grasp-region centre in link7 = (0.022, -0.010, 0.210) m (block placed centred between the open jaws by hand, joint angles read)
  - jaw tips = TCP + 15 mm, finger length 89.3 mm (umi_finger.stl) → finger root z = 0.1357
  - umi_finger.stl (mm): flat face x=0 is the contact face, x = thickness direction (26 mm, the closing direction),
    y = width (20 mm), z = 0 at the root → -89.3 at the tip
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
DST = os.path.join(REPO, "urdf", "openarm_v10_bimanual_umi.urdf")
MESH = "package://openarm_orboh_grasp/meshes/so101_finger.stl"  # resolved to grasp/meshes/ (see grasp/urdf_fk.py)

# 2026-11-20 night: the real jaw is 齊藤's so101-finger.step (fin-ray finger 110 x 35.3 x 39 mm), not the UMI finger.
# so101_finger.stl is that STEP rotated (no mirror) into the umi_finger.stl convention: flat contact face x=0,
# x = back of the fin (closing direction), y = width 35.3 mm, z = 0 at the root, -110 mm at the tip.
TCP = np.array([0.0082, -0.0031, 0.2203])   # taught grasp centre, recomputed with the re-calibrated chest camera
TIP_Z = 0.2295                              # jaw tip in link7 (flange z 89.5 mm + measured flange->tip 140 mm)
FINGER_LEN = 0.110
OPEN_HALF = 0.055


def rpy_from_R(R):
    sy = -R[2, 0]
    pitch = np.arcsin(np.clip(sy, -1, 1))
    roll = np.arctan2(R[2, 1], R[2, 2]); yaw = np.arctan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def main():
    tree = ET.parse(SRC); root = tree.getroot()
    for ln in ("openarm_left_left_finger", "openarm_left_right_finger"):
        link = root.find(f"link[@name='{ln}']")
        for tag in ("visual", "collision"):
            for el in link.findall(tag):
                link.remove(el)
    z_root = TIP_Z - FINGER_LEN
    # STL axes → link7: left jaw (+y side, contact face toward -y): x_s→+y, y_s→+x, z_s→-z
    R_l = np.array([[0, 1, 0], [1, 0, 0], [0, 0, -1]], float)
    # right jaw (-y side, contact face toward +y): x_s→-y, y_s→-x, z_s→-z
    R_r = np.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]], float)
    for side, R, mimic in (("l", R_l, False), ("r", R_r, True)):
        name = f"openarm_left_umi_finger_{side}"
        link = ET.SubElement(root, "link", name=name)
        for tag in ("visual", "collision"):
            el = ET.SubElement(link, tag)
            ET.SubElement(el, "origin", xyz="0 0 0", rpy="0 0 0")
            g = ET.SubElement(el, "geometry"); ET.SubElement(g, "mesh", filename=MESH, scale="0.001 0.001 0.001")
        inert = ET.SubElement(link, "inertial")
        ET.SubElement(inert, "origin", xyz="0.02 0 -0.05", rpy="0 0 0"); ET.SubElement(inert, "mass", value="0.05")
        ET.SubElement(inert, "inertia", ixx="1e-5", ixy="0", ixz="0", iyy="1e-5", iyz="0", izz="1e-5")
        j = ET.SubElement(root, "joint", name=f"{name}_joint", type="prismatic")
        r, p, y = rpy_from_R(R)
        ET.SubElement(j, "origin", xyz=f"{TCP[0]} {TCP[1]} {z_root:.4f}", rpy=f"{r:.6f} {p:.6f} {y:.6f}")
        ET.SubElement(j, "parent", link="openarm_left_link7"); ET.SubElement(j, "child", link=name)
        ET.SubElement(j, "axis", xyz="1 0 0")  # opening = thickness direction (moves away from the contact face)
        ET.SubElement(j, "limit", lower="0", upper=str(OPEN_HALF), effort="100", velocity="1")
        if mimic:
            ET.SubElement(j, "mimic", joint="openarm_left_umi_finger_l_joint", multiplier="1", offset="0")
    tcp = ET.SubElement(root, "link", name="openarm_left_umi_tcp")
    jt = ET.SubElement(root, "joint", name="openarm_left_umi_tcp_joint", type="fixed")
    ET.SubElement(jt, "origin", xyz=f"{TCP[0]} {TCP[1]} {TCP[2]}", rpy="0 0 0")
    ET.SubElement(jt, "parent", link="openarm_left_link7"); ET.SubElement(jt, "child", link="openarm_left_umi_tcp")
    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print("wrote", DST)


if __name__ == "__main__":
    main()
