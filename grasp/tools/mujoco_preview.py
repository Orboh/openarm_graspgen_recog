#!/usr/bin/env python3
"""Render a trajectory preview of both OpenArm arms with the cart / camera / clamp in MuJoCo. **Does not move the arm.**
2026-11-22 (Sota: 無条項（MuJoCo）でシミュレーションして見せて、目視で OK を出す). Reads grasp/runs/sim/standby_traj.json
(grasp/tools/standby_preview_traj.py) and the obstacles in tokyo-thor.world.yml; writes an mp4 + a contact sheet.

    MUJOCO_GL=egl ~/dimos/.venv/bin/python grasp/tools/mujoco_preview.py
"""
import json, os, xml.etree.ElementTree as ET
import cv2, mujoco, numpy as np, yaml
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SIM = os.path.join(REPO, "grasp/runs/sim"); MD = os.path.join(REPO, "grasp/curobo/meshes")


def build():
    r = ET.parse(os.path.join(REPO, "urdf/openarm_v10_bimanual_umi.urdf")).getroot()
    for l in r.findall("link"):
        for v in l.findall("visual"):
            l.remove(v)
        for c in list(l.findall("collision")):
            m = c.find("geometry/mesh")
            if m is None:
                continue
            f = os.path.join(MD, os.path.basename(m.get("filename")))
            if os.path.exists(f):
                m.set("filename", f)
            else:
                l.remove(c)
    mj = ET.SubElement(r, "mujoco"); ET.SubElement(mj, "compiler", {"discardvisual": "false", "fusestatic": "false", "strippath": "false", "balanceinertia": "true"})
    spec = mujoco.MjSpec.from_string(ET.tostring(r).decode())
    w = yaml.safe_load(open(os.path.join(REPO, "grasp/config/sites/tokyo-thor.world.yml")))
    wb = spec.worldbody
    spec.visual.headlight.ambient = [0.55, 0.55, 0.55]; spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    lt = wb.add_light(); lt.pos = [0.4, 0.3, 2.0]; lt.dir = [-0.2, -0.15, -1]; lt.diffuse = [0.6, 0.6, 0.6]
    g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_PLANE; g.size = [2, 2, 0.01]; g.pos = [0, 0, 0.28]; g.rgba = [0.75, 0.72, 0.65, 1]
    col = {"cart_tray": [0.9, 0.55, 0.1, 0.55], "camera_clamp": [0.9, 0.1, 0.1, 0.8], "chest_camera": [0.15, 0.15, 0.15, 0.9], "right_arm_home": [0, 0, 0, 0]}
    for n, c in {**(w.get("cuboid") or {}), **(w.get("cuboid_extra") or {})}.items():
        if col.get(n, [0, 0, 0, 1])[3] == 0:
            continue
        g = wb.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.size = [d / 2 for d in c["dims"]]; g.pos = c["pose"][:3]; g.quat = c["pose"][3:]
        g.rgba = col.get(n, [0.5, 0.5, 0.5, 0.6]); g.contype = 0; g.conaffinity = 0
    for arm in ("left", "right"):  # gripper body + jaws as boxes (grasp/curobo/make_urdf.py sizes), as arm_collision.py
        b = spec.body(f"openarm_{arm}_link7")
        for pos, size in (((0, 0, 0.107), (0.03, 0.05, 0.0175)), ((0, 0.041, 0.175), (0.019, 0.011, 0.055)), ((0, -0.041, 0.175), (0.019, 0.011, 0.055))):
            g = b.add_geom(); g.type = mujoco.mjtGeom.mjGEOM_BOX; g.pos = pos; g.size = size; g.rgba = [0.35, 0.35, 0.4, 1]; g.contype = 0; g.conaffinity = 0
    return spec.compile()


def main():
    import sys
    tin = sys.argv[1] if len(sys.argv) > 1 else os.path.join(SIM, "standby_traj.json")
    tag = sys.argv[2] if len(sys.argv) > 2 else "standby"
    m = build(); d = mujoco.MjData(m)
    traj = json.load(open(tin))["frames"]
    jid = {arm: [m.joint(f"openarm_{arm}_joint{i}").qposadr[0] for i in range(1, 8)] for arm in ("left", "right")}
    W, H = 640, 480
    ren = mujoco.Renderer(m, H, W)
    cams = []
    for az, el, dist, look in ((140, -22, 1.25, (0.12, 0, 0.5)), (90, -8, 1.2, (0.08, 0, 0.5)), (180, -75, 1.1, (0.12, 0, 0.45))):
        c = mujoco.MjvCamera(); c.azimuth, c.elevation, c.distance, c.lookat = az, el, dist, look; cams.append(c)
    names = ["front-left", "side (from the right)", "top"]
    vw = cv2.VideoWriter(os.path.join(SIM, f"{tag}_preview.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W * 3 // 2, H // 2 + 60))
    sheet = []
    for k, f in enumerate(traj):
        for arm, q in f["q"].items():
            for a, v in zip(jid[arm], q):
                d.qpos[a] = np.deg2rad(v)
        mujoco.mj_forward(m, d)
        tiles = []
        for c, nm in zip(cams, names):
            ren.update_scene(d, c); im = ren.render()[:, :, ::-1].copy()
            im = cv2.resize(im, (W // 2, H // 2)); cv2.putText(im, nm, (6, 16), 0, 0.45, (20, 20, 20), 1); tiles.append(im)
        row = np.hstack(tiles)
        bar = np.full((60, row.shape[1], 3), 255, np.uint8)
        cm = f["clear_cm"]; worst = min(cm, key=cm.get)
        cv2.putText(bar, f["label"], (8, 22), 0, 0.6, (0, 0, 0), 2)
        colr = (0, 0, 220) if cm[worst] < 1.0 else ((0, 140, 255) if cm[worst] < 2.0 else (0, 140, 0))
        cv2.putText(bar, f"closest: {worst} {cm[worst]:.1f} cm   (cart {cm.get('cart_tray', 99):.1f} / clamp {cm.get('camera_clamp', 99):.1f} / body {cm.get('body', 99):.1f})",
                    (8, 48), 0, 0.5, colr, 1)
        fr = np.vstack([row, bar]); vw.write(fr)
        if k == 0 or traj[k - 1]["label"] != f["label"] and "->" not in f["label"]:
            sheet.append(fr)
    vw.release()
    cv2.imwrite(os.path.join(SIM, f"{tag}_sheet.jpg"), np.vstack(sheet[:12]))
    print("wrote", os.path.join(SIM, f"{tag}_preview.mp4"), len(traj), "frames;", len(sheet), "key poses")


if __name__ == "__main__":
    main()
