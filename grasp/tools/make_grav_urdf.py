#!/usr/bin/env python3
"""重力補正用の URDF を作る（urdf/openarm_v10_bimanual_grav.urdf）。2026-11-23 Munenori（Kota の PC の Claude Code から）。

元の urdf/openarm_v10_bimanual.urdf は openarm_<side>_hand が質量の無い空のリンクで、重力補正
（lerobot の OpenArmGravityModel）は link7 より先を 0 kg として計算していた。実機で止めた姿勢の試し
（grasp/tools/hold_test.py・2026-11-23 左腕）で、TCP が 12〜18 mm 垂れ、link7 の z 0.10〜0.13 m に 0.33〜0.38 kg が
足りないと当てはまった。そこで両腕に次を足す（元のファイルは変えない）:
  - 純正の手（モーター・レール・取り付け部）: 0.35 kg・重心 link7 の z 0.1301 m（上流 openarm_description の
    hand の値: link7→hand z 0.1001 + 重心 z 0.03。慣性も上流の値）
  - 爪（so101 のフィンレイ指 2 本・北九州で量って 2 本で 130 g）: 重心は爪の根元 z 0.1195 m と先 z 0.2295 m の
    真ん中 z 0.175 m とする（make_umi_urdf.py の TIP_Z・FINGER_LEN）。左右対称なので x・y は 0
指の関節（finger_joint）は OpenArmGravityModel が 0 kg にするので、爪は link7 に固定した別のリンクで足す。

    python3 grasp/tools/make_grav_urdf.py
"""
from __future__ import annotations

import os
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
DST = os.path.join(REPO, "urdf", "openarm_v10_bimanual_grav.urdf")

HAND_MASS = 0.35                       # 純正の手（上流の値・2026-11-23 Munenori の指示）
HAND_COM_Z = 0.1001 + 0.03             # link7 の座標
HAND_INERTIA = dict(ixx="0.0002473", ixy="1e-06", ixz="1e-06", iyy="1.763e-05", iyz="1e-06", izz="0.0002521")  # 上流の値
JAW_MASS = 0.130                       # 爪 2 本（2026-11-23 北九州で量った）
JAW_COM_Z = (0.1195 + 0.2295) / 2      # 爪の根元と先の真ん中
JAW_INERTIA = dict(ixx="2e-04", ixy="0", ixz="0", iyy="2e-04", iyz="0", izz="5e-05")  # 長さ 110 mm の棒 2 本の目安


def add_inertial(link: ET.Element, mass: float, z: float, inertia: dict) -> None:
    """link に慣性を入れる（すでにあれば置き換える）。"""
    for old in link.findall("inertial"):
        link.remove(old)
    el = ET.SubElement(link, "inertial")
    ET.SubElement(el, "origin", xyz=f"0 0 {z:.4f}", rpy="0 0 0")
    ET.SubElement(el, "mass", value=f"{mass}")
    ET.SubElement(el, "inertia", **inertia)


def main() -> None:
    tree = ET.parse(SRC); root = tree.getroot()
    for side in ("left", "right"):
        hand = root.find(f"link[@name='openarm_{side}_hand']")
        j = root.find(f"joint[@name='openarm_{side}_hand_joint']")
        if hand is None or j is None or j.find("parent").get("link") != f"openarm_{side}_link7":
            raise SystemExit(f"[grav-urdf] openarm_{side}_hand が link7 に固定されていない（元の URDF が変わった）")
        o = j.find("origin")
        if o is not None and any(abs(float(v)) > 1e-9 for v in o.get("xyz", "0 0 0").split()):
            raise SystemExit(f"[grav-urdf] openarm_{side}_hand_joint が link7 の原点に無い。重心の z を見直す")
        add_inertial(hand, HAND_MASS, HAND_COM_Z, HAND_INERTIA)
        jaw = ET.SubElement(root, "link", name=f"openarm_{side}_jaws_mass")
        add_inertial(jaw, JAW_MASS, JAW_COM_Z, JAW_INERTIA)
        jj = ET.SubElement(root, "joint", name=f"openarm_{side}_jaws_mass_joint", type="fixed")
        ET.SubElement(jj, "origin", xyz="0 0 0", rpy="0 0 0")
        ET.SubElement(jj, "parent", link=f"openarm_{side}_link7"); ET.SubElement(jj, "child", link=f"openarm_{side}_jaws_mass")
    root.insert(0, ET.Comment(" grasp/tools/make_grav_urdf.py が作った重力補正用の URDF。手で直さない "))
    ET.indent(tree, space="  ")
    tree.write(DST, encoding="utf-8", xml_declaration=True)
    print(f"[grav-urdf] {DST} に書いた（手 {HAND_MASS} kg・z {HAND_COM_Z:.4f} / 爪 {JAW_MASS} kg・z {JAW_COM_Z:.4f}）")


if __name__ == "__main__":
    main()
