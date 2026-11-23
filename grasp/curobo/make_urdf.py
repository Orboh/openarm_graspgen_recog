#!/usr/bin/env python3
"""cuRobo 用の左腕だけの URDF を作る（grasp/curobo/openarm_left_umi.urdf）。

元は urdf/openarm_v10_bimanual_umi.urdf（Pinocchio の IK と同じ根の座標 = world）。元のファイルは変えない。
- 右腕の link・joint を外す（北九州の収集機で左腕が当たる恐れは無いと 2026-11-21 Munenori が判断）
- 左の元の平行リンクの指（openarm_left_left_finger・right_finger）を外す。実物は so101 の指（openarm_left_umi_finger_*）
- 本体（openarm_body_link0）は link だけ残し、形は周りの物のメッシュとして入れる（meshes/body_link0_symp.stl を写す）。
  北九州の本体のフレームは純正の本体で、胸のカメラもこれに付いている（2026-11-21 Munenori）
- visual を collision と同じ形に置き換え（cuRobo の球の当てはめは visual を使う）、collision のメッシュを grasp/curobo/meshes/ に写したファイルへ向け直す
  （腕は ~/projects/openarm-orboh/openarm_description（Apache-2.0）の v10 の簡略メッシュ、指は grasp/meshes/so101_finger.stl）
- 手の本体（指を動かすモーターの箱）は URDF に無いので、箱の link（openarm_left_gripper_body）を足す（寸法は実物と合うと 2026-11-21 Munenori が確かめた）

    python3 grasp/curobo/make_urdf.py
"""
from __future__ import annotations

import copy
import os
import shutil
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
SRC = os.path.join(REPO, "urdf", "openarm_v10_bimanual_umi.urdf")
DST = os.path.join(HERE, "openarm_left_umi.urdf")
MESH_DIR = os.path.join(HERE, "meshes")
OPENARM_DESC = os.path.expanduser(os.environ.get("OPENARM_DESC", "~/projects/openarm-orboh/openarm_description/meshes"))

# 手の本体の箱（link7 の座標・[m]）。2026-11-21 Munenori が実物と合うと確かめた: 手の取り付け面 z=0.0895 から指の付け根 z=0.1195 の少し先まで、
# 閉じる向き（link7 の y）は全開の指の外側まで
GRIPPER_BODY_XYZ = (0.0, 0.0, 0.107)
GRIPPER_BODY_SIZE = (0.060, 0.100, 0.035)


def mesh_source(fn: str) -> str:
    """package:// のパスを手元のファイルへ。"""
    if fn.startswith("package://openarm_orboh_grasp/"):
        return os.path.join(REPO, "grasp", fn.split("openarm_orboh_grasp/", 1)[1])
    if "/mesh/arm/collision/" in fn:
        return os.path.join(OPENARM_DESC, "arm", "v10", "collision", os.path.basename(fn))
    if "/mesh/body/collision/" in fn:
        return os.path.join(OPENARM_DESC, "body", "v10", "collision", os.path.basename(fn))
    raise SystemExit(f"[make_urdf] 知らないメッシュのパス: {fn}")


def main() -> int:
    tree = ET.parse(SRC)
    root = tree.getroot()
    drop_links = {"openarm_left_left_finger", "openarm_left_right_finger"}
    for el in list(root):
        name = el.get("name", "")
        if el.tag == "link" and (name.startswith("openarm_right") or name in drop_links):
            root.remove(el)
        elif el.tag == "joint":
            parent = el.find("parent").get("link")
            child = el.find("child").get("link")
            if name.startswith("openarm_right") or child in drop_links or parent in drop_links or child.startswith("openarm_right"):
                root.remove(el)
    os.makedirs(MESH_DIR, exist_ok=True)
    for link in root.iter("link"):
        for v in link.findall("visual"):
            link.remove(v)
        if link.get("name") == "openarm_body_link0":
            # 本体の形は腕の球に入れず、周りの物（メッシュ）として入れる。球で当てると覆われない所とはみ出しが 2〜6 cm になる
            # （2026-11-21 測った）。メッシュは写しておき、grasp/config/sites/*.world.yml の mesh から読む
            for c in link.findall("collision"):
                fn = c.find("geometry/mesh").get("filename")
                shutil.copyfile(mesh_source(fn), os.path.join(MESH_DIR, os.path.basename(fn)))
                link.remove(c)
        for m in link.iter("mesh"):
            src = mesh_source(m.get("filename"))
            shutil.copyfile(src, os.path.join(MESH_DIR, os.path.basename(src)))
            m.set("filename", "meshes/" + os.path.basename(src))
    # 手の本体の箱
    body = ET.SubElement(root, "link", name="openarm_left_gripper_body")
    col = ET.SubElement(body, "collision")
    ET.SubElement(col, "origin", xyz="0 0 0", rpy="0 0 0")
    ET.SubElement(ET.SubElement(col, "geometry"), "box", size=" ".join(str(v) for v in GRIPPER_BODY_SIZE))
    # cuRobo の球の当てはめ（RobotBuilder）は visual の形を使うので、visual に collision と同じ形を入れる
    for link in root.iter("link"):
        for c in link.findall("collision"):
            v = copy.deepcopy(c); v.tag = "visual"; link.append(v)
    j = ET.SubElement(root, "joint", name="openarm_left_gripper_body_joint", type="fixed")
    ET.SubElement(j, "origin", xyz=" ".join(str(v) for v in GRIPPER_BODY_XYZ), rpy="0 0 0")
    ET.SubElement(j, "parent", link="openarm_left_link7")
    ET.SubElement(j, "child", link="openarm_left_gripper_body")
    ET.indent(tree)
    with open(DST, "wb") as f:
        f.write("<?xml version='1.0' encoding='utf-8'?>\n<!-- grasp/curobo/make_urdf.py が作った。手で直さない -->\n".encode())
        tree.write(f, encoding="utf-8", xml_declaration=False)
    links = [l.get("name") for l in root.iter("link")]
    print(f"[make_urdf] {DST} に書いた（link {len(links)}: {', '.join(links)}）")
    print(f"[make_urdf] メッシュ {len(os.listdir(MESH_DIR))} 個を {MESH_DIR} に写した")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
