#!/usr/bin/env python3
"""Capture one frame with the chest D435i on Thor → pull it to this PC → cut out the object → show it on this PC's screen.

Each run overwrites grasp/runs/latest.jpg, and that window (eog) follows it automatically.
Left = color + the cut-out object (red) + ROI box (light blue); right = depth.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/thor_snap.py
    # ROI (image rows and distance) can be narrowed:
    ... thor_snap.py --rows 120:350 --zmin 0.40 --zmax 0.75
    # 拾う物を名前で選ぶ（grasp/config/objects/<名前>.json・2026-11-22）:
    ... thor_snap.py --target wood        # alframe（アルミのバー）も。--list-targets で一覧
    # YOLOE-26 + SigLIP2（本 repo で追加）: 孤立度・ロボットからの距離・積みの高さの規則で対象を決め、その 1 つだけ材質を見る
    ... thor_snap.py --yoloe

The arm does not move (it only uses the camera). Can't run while record / stream_chest holds the camera.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grasp import remote  # noqa: E402
from grasp.capture import depth_to_points  # noqa: E402
from grasp import objects  # noqa: E402
from grasp.segment import SegParams, segment_clusters  # noqa: E402
from grasp import timing  # noqa: E402  段ごとの時間（記録だけ・2026-11-25 Shreyan）

THOR = remote.HOST  # 既定は東京の Thor。腕とカメラがこの PC にあるなら GRASP_ROBOT_HOST=local（grasp/remote.py）


def self_segments(q_deg, side="left"):
    """その腕の関節 3〜7・指先の中心・爪の先を結ぶ線分（腕の付け根の座標）。--self-q・--self-q-right で腕の点を消すのに使う。"""
    from grasp import plan as P  # pinocchio・爪の寸法（GRASP_GRIPPER_JSON）を使う
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(side=side)
    pts = P.arm_points(ik, np.deg2rad(q_deg))  # 関節 3〜7・その間の中点・指先の中心・爪の先
    chain = [pts[0], pts[1], pts[2], pts[3], pts[4], pts[-2], pts[-1]]
    segs = list(zip(chain[:-1], chain[1:]))
    # 左右の爪の板（全開の位置・閉じる向き ±45 mm）も線にする（2026-11-23: 腕の線から 3 cm だけ消すと、開いた外側の爪が残り、
    # 画面の外から来るケーブルとつながって拾う物に選ばれた）。爪の座標の原点は手首の先の面、+Z が近づく向き、+X が閉じる向き
    import pinocchio as pin
    Tg = P.t_l7_g(side)
    MG = ik.data.oMi[ik.model.getJointId(f"openarm_{side}_joint7")].act(pin.SE3(Tg[:3, :3], Tg[:3, 3]))
    for sx in (-1.0, 1.0):
        segs.append((MG.act(np.array([sx * 0.045, 0.0, 0.0])), MG.act(np.array([sx * 0.045, 0.0, P.CFG["flange_to_tip_m"]]))))
    return segs
RUNS = os.path.join(os.path.dirname(__file__), "..", "runs")


def perceive_target(a, img, depth, K, tbc, d):
    """Aluminium frame in a pile (2026-11-21): ask perceive_server for candidate masks, keep those scored as aluminium,
    take the one lying highest above the floor (top of the pile first), and return its points in the camera frame.
    Where the shiny top has no depth, the mask pixels are put on the floor plane raised by --alu-height."""
    import glob
    import cv2
    import msgpack
    import msgpack_numpy as mn
    import zmq
    host, port = (a.perceive.split(":") + ["5558"])[:2]
    sock = zmq.Context.instance().socket(zmq.REQ); sock.setsockopt(zmq.RCVTIMEO, 60000); sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host}:{port}")
    try:
        sock.send(msgpack.packb({"bgr": img}, default=mn.encode, use_bin_type=True))
        rep = msgpack.unpackb(sock.recv(), object_hook=mn.decode, raw=False)
        if a.perceive_target == "wood" and "error" not in rep:
            # 2026-11-21: clear a rigid neighbour first. The aluminium answer above marks the bars (kept as rigid
            # obstacles); a second request finds the wood block, which becomes the target.
            bars = rep["cands"]
            sock.send(msgpack.packb({"bgr": img, "prompts": ["wooden block", "wood", "brown block", "cardboard box"]},
                                    default=mn.encode, use_bin_type=True))
            rep = msgpack.unpackb(sock.recv(), object_hook=mn.decode, raw=False)
            if "error" not in rep:
                hsv0 = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                for c_ in rep["cands"]:  # wood score: brown pixel share in the mask, compact, not the robot / bag
                    mm = np.asarray(c_["mask"], bool)
                    brown = ((hsv0[..., 0] >= 8) & (hsv0[..., 0] <= 30) & (hsv0[..., 1] >= 45))[mm].mean() if mm.any() else 0
                    c_["alu"] = float(brown) * (1.0 if mm.sum() < 0.03 * img.shape[0] * img.shape[1] else 0.2)
                rep["cands"].sort(key=lambda c_: -c_["alu"])
                rep["bars"] = [b for b in bars if b["alu"] >= 0.3]
    finally:
        sock.close()
    if "error" in rep:
        raise RuntimeError(rep["error"])
    cands = [c for c in rep["cands"] if c["alu"] >= a.alu_thr]
    summary = [{k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items() if k != "mask"} for c in rep["cands"]]
    if not os.path.exists(tbc):
        raise RuntimeError("no hand-eye (T_base_cam) to place the floor")
    T = np.array(json.load(open(tbc))["T_base_cam"])
    # floor plane from the whole depth image (largest plane), in the camera frame
    from grasp.segment import SegParams, segment_largest_object  # noqa: F401  (open3d via the same env)
    import open3d as o3d
    from grasp.plan import floor_plane_base
    nb, db = floor_plane_base(d, T)  # the fixed floor (config/floor_plane.json), same as plan.py (2026-11-21 23:15)
    n = T[:3, :3].T @ nb; dd = float(db + nb @ T[:3, 3])  # in the camera frame: height = n·p + dd
    best = None; found = []
    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    P_all = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)
    h_all = P_all @ n + dd
    for c in cands:
        m = np.asarray(c["mask"], bool)
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        z = depth[m]; uu, vv = u[m], v[m]
        P = np.stack([(uu - K[0, 2]) * z / K[0, 0], (vv - K[1, 2]) * z / K[1, 1], z], 1)
        h = P @ n + dd
        ok = (z > 0) & (h > 0.005) & (h < a.hmax)
        if ok.sum() >= max(60, 0.25 * m.sum()):
            pts = P[ok]; src = "depth"
        else:  # shiny top: rays through the mask pixels onto the floor plane raised by the bar height
            r = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(len(uu))], 1)
            t = (a.alu_height - dd) / (r @ n)
            pts = r * t[:, None]; pts = pts[(t > 0)]; src = "plane+height"
        if len(pts) < 40:
            continue
        pb_ = pts @ T[:3, :3].T + T[:3, 3]
        top = float(np.percentile(pts @ n + dd, 90))
        if a.band != "none":
            x0, x1, y0, y1 = map(float, a.band.split(","))
            cx, cy = pb_[:, 0].mean(), pb_[:, 1].mean()
            if not (x0 < cx < x1 and y0 < cy < y1):
                continue
        # 2026-11-21 (Sota: 隙間のあるバーを優先): count rigid things beside the bar where the open jaws go. Points not in
        # this mask, at least as high as the bar's middle (soft sacks underneath are lower), within the jaw opening
        # (±a.gap_half m across the bar's long axis, past its own half-width) along the bar's length.
        hm = float(np.median(pts @ n + dd))
        xy = pb_[:, :2]; mu = xy.mean(0); w_, V_ = np.linalg.eigh(np.cov((xy - mu).T)); ax = V_[:, 1]; nx = np.array([-ax[1], ax[0]])
        al, ac = (xy - mu) @ ax, (xy - mu) @ nx
        hw = float(np.percentile(np.abs(ac), 90))
        o = (~m) & (depth > 0) & (h_all > hm + 0.005) & (h_all < 0.25)
        ob = P_all[o] @ T[:3, :3].T + T[:3, 3]
        oa, oc = (ob[:, :2] - mu) @ ax, (ob[:, :2] - mu) @ nx
        a0, a1, c0 = al.min() - 0.02, al.max() + 0.02, hw + 0.006
        inside = (oa > a0) & (oa < a1) & (np.abs(oc) > c0) & (np.abs(oc) < a.gap_half)
        # share of 1 cm cells in the jaw corridor that hold something rigid (point counts depend on distance)
        cells = {(int((x - a0) // 0.01), int(np.floor(y / 0.01))) for x, y in zip(oa[inside], oc[inside])}
        n_cells = max(1, int(np.ceil((a1 - a0) / 0.01)) * 2 * int(np.ceil((a.gap_half - c0) / 0.01)))
        crowd = round(len(cells) / n_cells, 3)
        c["crowd"] = crowd
        found.append((top, pts, c, src, crowd))
    # clear bars first (few rigid points beside them), then the highest; crowded bars stay as later fallbacks
    # when every bar is crowded, try the least crowded first (11/21 23:14: the pipe at 0.14 waited behind a bar at 0.30)
    found.sort(key=lambda f: (f[4] > a.crowd_max, f[4] if f[4] > a.crowd_max else 0.0, -f[0]))
    best = found[0][:4] if found else None
    for f in found:
        summary[[id(x) for x in rep["cands"]].index(id(f[2]))]["crowd"] = f[4]
    json.dump({"cands": summary, "alu_thr": a.alu_thr, "seconds": rep.get("seconds"),
               "chosen": None if best is None else {k: v for k, v in summary[[id(x) for x in rep["cands"]].index(id(best[2]))].items()},
               "points_from": None if best is None else best[3]}, open(os.path.join(d, "perceive.json"), "w"), indent=1)
    if best is not None and a.skip_crowded and found[0][4] > a.crowd_max:
        raise RuntimeError(f"every aluminium frame in reach is crowded (jaw corridor filled {[f[4] for f in found]} > "
                           f"{a.crowd_max}): skip this round, do not move")
    if best is None:
        raise RuntimeError(f"no aluminium frame in reach ({len(cands)} aluminium-like of {len(rep['cands'])} candidates)")
    top, pts, c, src = best
    cv2.imwrite(os.path.join(d, "mask.png"), (np.asarray(c["mask"], np.uint8) * 255))
    # rigid things that are NOT the target (other bar-like candidates, wood-coloured pixels): plan.py --soft-below keeps
    # them as obstacles even when they lie lower than the target (11/21 19:47: the target bar rested on a second bar and
    # the jaw would have landed on it; height alone called it "soft")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rigid = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 30) & (hsv[..., 1] >= 45)).astype(np.uint8)
    rigid = cv2.morphologyEx(rigid, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)).astype(bool)
    for o in rep["cands"] + rep.get("bars", []):
        if o is not c and (o["alu"] >= 0.3 or o.get("aspect", 0) >= 3.0 or o in rep.get("bars", [])) and np.asarray(o["mask"]).sum() < 0.08 * img.shape[0] * img.shape[1]:
            rigid |= np.asarray(o["mask"], bool)
    rigid &= ~cv2.dilate(np.asarray(c["mask"], np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    cv2.imwrite(os.path.join(d, "rigid.png"), rigid.astype(np.uint8) * 255)
    # perceive.jpg (2026-11-21, for the panel): every candidate the detector found, its aluminium score and crowd;
    # red = chosen target, orange = other aluminium (fallbacks), grey = not aluminium, magenta = rigid neighbours
    try:
        vis = img.copy(); ov = vis.copy(); ov[rigid] = (255, 0, 255); vis = cv2.addWeighted(ov, 0.35, vis, 0.65, 0)
        alts = [f[2] for f in found if f[2] is not c]
        for o in rep["cands"]:
            col = (0, 0, 255) if o is c else (0, 140, 255) if any(o is x for x in alts) else (160, 160, 160)
            cnts, _ = cv2.findContours(np.asarray(o["mask"], np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(vis, cnts, -1, col, 2)
            x0, y0 = int(o["box"][0]), int(o["box"][1])
            txt = f"alu {o['alu']:.2f}" + (f" crowd {o['crowd']:.2f}" if o.get("crowd") is not None else "") + (" TARGET" if o is c else "")
            cv2.putText(vis, txt, (x0, max(12, y0 + 12)), 0, 0.42, col, 2)
        cv2.imwrite(os.path.join(d, "perceive.jpg"), vis)
    except Exception as e:  # noqa: BLE001  (picture only)
        print(f"[snap] perceive.jpg failed: {e}")
    # the other aluminium frames, highest first: plan.py tries them when the first one has no reachable grasp
    # (11/21: near the robot only closing directions 0-60 deg from the base x axis solve IK, so many bars are skipped)
    for f in glob.glob(os.path.join(d, "object_alt_*.npy")):
        os.remove(f)
    for i, (_, p_, _, _, _) in enumerate([f for f in found if f[2] is not c][:4]):
        np.save(os.path.join(d, f"object_alt_{i}.npy"), p_.astype(np.float32))
    ext = pts.max(0) - pts.min(0)
    return pts.astype(np.float32), (f"alu {c['alu']:.2f} ({len(cands)} of {len(rep['cands'])}) {len(pts)} pts [{src}] "
                                    f"size {ext[0]*100:.0f}x{ext[1]*100:.0f}x{ext[2]*100:.0f} cm top {top*100:.1f} cm crowd {c.get('crowd')}"
                                    f" (others: {[f[4] for f in found[1:]]})")


def clean_points(pts: np.ndarray, p: SegParams = SegParams()) -> np.ndarray:
    """mask から出した点（カメラの座標）を、segment_clusters と同じ手順で整える（2026-11-23 Shreyan）。
    mask の縁の深度の飛び（背景と混ざった画素）が残ると、外形が 10 cm 以上ふくらみ、GraspGenX の候補が爪の隙間・床の確かめで全部落ちた。"""
    import open3d as o3d
    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts.astype(np.float64))).voxel_down_sample(p.voxel)
    if len(pc.points) < p.dbscan_min_pts:
        return np.asarray(pc.points, dtype=np.float32)
    lab = np.asarray(pc.cluster_dbscan(p.dbscan_eps, p.dbscan_min_pts))
    if lab.max() >= 0:
        pc = pc.select_by_index(np.where(lab == np.argmax(np.bincount(lab[lab >= 0])))[0])
    pc, _ = pc.remove_statistical_outlier(p.sor_nb, p.sor_std)
    return np.asarray(pc.points, dtype=np.float32)


def pick_target(a, img, depth, K, tbc, d):
    """Generalized heap sorter (2026-11-23): YOLOE-26 prompt-free gives every object's mask in one shot (no text
    prompts to tune, unlike --perceive's Grounding DINO), and each is back-projected to 3D the same way --perceive
    does it: real depth where there is enough of it, floor-plane + assumed height where the top is shiny/missing.
    Rule (pure geometry, no material involved): an object isolated from the rest of the heap beats picking off the
    top of the heap; among isolated candidates the one FARTHEST from the robot (base-frame origin) goes first; among
    non-isolated ones, whichever sits highest above the floor goes first (same "clear ones, then highest" idea as
    --perceive, just not hardcoded to bar shapes). Material plays no part in picking the target.
    Material (grasp.classification.classify_material, SigLIP zero-shot) is found only AFTER the target is decided --
    the winner of the geometric ranking is classified once, to label it for whichever bin it gets routed to; SigLIP
    never sees, and never influences the choice of, any other candidate."""
    import glob
    import cv2
    from grasp.segment import segment_object_frames
    from grasp.classification import classify_material
    from grasp.plan import floor_plane_base

    if not os.path.exists(tbc):
        raise RuntimeError("no hand-eye (T_base_cam) to place the floor / rank by distance to the robot")
    T = np.array(json.load(open(tbc))["T_base_cam"])
    _t = time.time()
    nb, db = floor_plane_base(d, T)
    timing.event("snap/floor plane (RANSAC)", _t, time.time())
    n = T[:3, :3].T @ nb; dd = float(db + nb @ T[:3, 3])  # camera frame: height = n·p + dd

    _t = time.time()
    r = segment_object_frames(d)[0]
    timing.event("snap/yoloe (load+predict)", _t, time.time())
    if r.masks is None or len(r.masks.data) == 0:
        raise RuntimeError("YOLOE found no objects in this frame")
    masks = r.masks.data.cpu().numpy() > 0.5
    confs = r.boxes.conf.cpu().numpy() if r.boxes is not None else np.ones(len(masks))

    v, u = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
    P_all = np.stack([(u - K[0, 2]) * depth / K[0, 0], (v - K[1, 2]) * depth / K[1, 1], depth], -1)

    _t_masks = time.time()
    full = [cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
            if m.shape != depth.shape else m for m in masks]
    # 2 つ以上の mask が取り合う画素は、いちばん小さい mask のものにする（物はマット・台の上に載っている。2026-11-23: 前は
    # 「どちらにも入れない」にしていて、マットの mask に丸ごと覆われた木の丸太が消えた）
    owner = np.full(depth.shape, -1, int)
    for i in sorted(range(len(full)), key=lambda i: full[i].sum(), reverse=True):
        owner[full[i]] = i
    cands = []
    n_plane = 0
    n_border = 0

    def _why(i, reason):  # GRASP_SNAP_DEBUG=1 で、どの mask を何で外したかを出す
        if os.environ.get("GRASP_SNAP_DEBUG"):
            ys_, xs_ = np.where(full[i])
            print(f"[snap] mask {i} u {xs_.min()}-{xs_.max()} v {ys_.min()}-{ys_.max()}: 外した（{reason}）", flush=True)
    for i, m in enumerate(full):
        if m.shape != depth.shape:
            m = cv2.resize(m.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        m = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        m &= owner == i
        # 画面の縁に掛かる物（外から来るケーブル・画面の外へはみ出した物）は全体が見えないので選ばない（2026-11-23: 腕のケーブルを選んだ）
        b, fm = a.border_px, full[i]  # 縮める前の mask で見る
        if b > 0 and (fm[:b].any() or fm[-b:].any() or fm[:, :b].any() or fm[:, -b:].any()):
            n_border += 1
            _why(i, "border")
            continue
        # いちばん大きいつながった領域だけを残す（2026-11-23: バーの mask に重量計の縁の細い切れ端が付き、縁と縁の距離が 0 になった）
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        if nl > 2:
            m = lab == (1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA])))
        if m.sum() < 60:
            _why(i, f"too small after cleanup ({int(m.sum())} px)")
            continue
        z = depth[m]; uu, vv = u[m], v[m]
        P = P_all[m]
        h = P @ n + dd
        valid = z > 0
        # 2026-11-23 Shreyan: RANSAC の床（floor_plane_base）を、対象を選ぶ前の絞り込みにも使う。深度のある画素の大半が床の面の上
        # （|高さ| < --plane-dist）なら、床・マット・床の模様なので候補にしない。前は深度があっても上に点が足りないと下の
        # 「光って深度が無い」扱いに落ち、床から 2 cm 浮いた物として候補に戻っていた
        if valid.sum() >= 0.5 * m.sum() and np.mean(np.abs(h[valid]) < a.plane_dist) > a.plane_frac:
            n_plane += 1
            _why(i, "plane")
            continue
        ok = valid & (h > a.plane_dist) & (h < a.hmax)
        if ok.sum() >= max(60, 0.25 * m.sum()):
            pts, src = P[ok], "depth"
        elif valid.sum() >= 0.5 * m.sum():  # 深度はあるのに、床の上の高さの範囲に点が足りない（--hmax より高い物・腕など）
            _why(i, f"height (ok {int(ok.sum())}/{int(m.sum())} px between {a.plane_dist}..{a.hmax} m)")
            continue
        else:  # shiny/no-depth top: rays through the mask pixels onto the floor plane raised by the assumed height
            ray = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones(len(uu))], 1)
            t = (a.obj_height - dd) / (ray @ n)
            pts = ray * t[:, None]; pts, src = pts[t > 0], "plane+height"
        if src == "depth":
            pts = clean_points(pts)  # 旧い切り出し（segment_clusters）と同じ後始末: 4 mm の格子・DBSCAN のいちばん大きいかたまり・外れ値
        if len(pts) < 40:
            _why(i, f'too few 3D points ({len(pts)})')
            continue
        pb = pts @ T[:3, :3].T + T[:3, 3]
        centroid = pb[:, :2].mean(0)
        cands.append({"mask": m, "pts": pts, "pb": pb, "src": src,
                     "top": float(np.percentile(pts @ n + dd, 90)),
                     "centroid": centroid, "radius": float(np.percentile(np.linalg.norm(pb[:, :2] - centroid, axis=1), 90)),
                     "material": None, "mat_conf": None, "det_conf": float(confs[i])})
    timing.event("snap/masks → 3D candidates", _t_masks, time.time(), masks=len(full), cands=len(cands))
    _t_filt = time.time()
    if os.environ.get("GRASP_SNAP_DEBUG"):
        for c in cands:
            ys_, xs_ = np.where(c["mask"])
            print(f"[snap] 候補 u {xs_.min()}-{xs_.max()} v {ys_.min()}-{ys_.max()} 中心 {np.round(c['centroid'], 3).tolist()} "
                  f"上端 {c['top']:.3f} 点 {len(c['pts'])} {c['src']}", flush=True)
    if n_border:
        print(f"[snap] yoloe: 画面の縁に掛かるので外した mask {n_border} 個", flush=True)
    if n_plane:
        print(f"[snap] yoloe: 床の面（RANSAC）の上の平らな領域として外した mask {n_plane} 個", flush=True)
    # 2026-11-23 Shreyan: 旧い切り出しと同じく、腕の届く範囲（--band）の外と、ロボット自身の腕・爪（--self-q）を候補から外す。
    # 外さないと、胸のカメラに写った自分のグリッパー（机から 13 cm）が「積みのいちばん上」として選ばれた
    if cands and a.band != "none":
        x0, x1, y0, y1 = map(float, a.band.split(","))
        cands = [c for c in cands if x0 < c["centroid"][0] < x1 and y0 < c["centroid"][1] < y1]
    if cands and a.exclude_xy:  # 2026-11-23 Shreyan: この回までに置いた所（run_pipeline.sh --n）。置いた物を拾い直さない
        # "x,y" は --exclude-r、"x,y,r" はその半径（拾えなかった物は 3 cm だけ外す。10 cm だと隣の物まで外れた）
        ex = [[float(v) for v in xy.split(",")] for xy in a.exclude_xy.split(";") if xy.strip()]
        done_xy = np.array([e[:2] for e in ex]); rad = np.array([e[2] if len(e) > 2 else a.exclude_r for e in ex])
        before = len(cands)
        cands = [c for c in cands if np.all(np.linalg.norm(done_xy - c["centroid"], axis=1) > rad)]
        if len(cands) < before:
            print(f"[snap] yoloe: 置いた所にある候補 {before - len(cands)} 個を外した", flush=True)
    segs = getattr(a, "_self_segs", None)
    if cands and segs:
        r = a.self_drop if a.self_drop > 0 else a.self_r
        keep = []
        for c in cands:
            dmin = np.full(len(c["pb"]), np.inf)
            for p0, p1 in segs:
                sg = p1 - p0; t = np.clip(((c["pb"] - p0) @ sg) / max(sg @ sg, 1e-9), 0, 1)
                dmin = np.minimum(dmin, np.linalg.norm(c["pb"] - (p0 + t[:, None] * sg), axis=1))
            if np.mean(dmin <= r) <= 0.5:
                keep.append(c)
        if len(keep) < len(cands):
            print(f"[snap] yoloe: 腕・爪として外した候補 {len(cands) - len(keep)} 個", flush=True)
        cands = keep
    timing.event("snap/band+exclude+self filters", _t_filt, time.time(), cands=len(cands))
    if not cands:
        raise RuntimeError("no candidates survived depth/height/band/self filtering")
    _t_iso = time.time()

    # 2026-11-23 Shreyan: 孤立の度合いは、中心と半径ではなく、切り出した物どうしの縁と縁の距離で測る（机に沿った XY の最短距離）。
    # 中心と半径では、大きい物（重量計・長いバー）の半径がほかの物まで届き、どれも孤立と見なされなかった
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(0)
    for c in cands:
        xy = c["pb"][:, :2]
        c["_xy"] = xy[rng.choice(len(xy), 3000, replace=False)] if len(xy) > 3000 else xy
        c["_tree"] = cKDTree(c["_xy"])
    for c in cands:
        gaps = [float(o["_tree"].query(c["_xy"], k=1)[0].min()) for o in cands if o is not c]
        c["isolation_gap"] = min(gaps) if gaps else float("inf")
        c["isolated"] = c["isolation_gap"] > a.isolation_gap
        # ロボットからの距離 = ハンドアイ（T_base_cam）で出した腕の付け根の座標（openarm_body_link0）の原点から、物の中心までの水平距離
        c["dist_robot"] = float(np.linalg.norm(c["centroid"]))
    for c in cands:
        del c["_xy"], c["_tree"]
    # 孤立した物が先。孤立した物の中では、ロボットからいちばん遠い物が先。孤立した物が無ければ、積みのいちばん上（床から高い物）が先
    cands.sort(key=lambda c: (not c["isolated"], -c["dist_robot"] if c["isolated"] else -c["top"]))
    best = cands[0]  # target is decided by geometry alone -- material has no vote
    timing.event("snap/isolation + ranking", _t_iso, time.time(), cands=len(cands))

    # NOW classify, once, only the winner -- the losers' material is never needed, so it's never computed
    ys, xs = np.where(best["mask"])
    # 2026-11-23 Shreyan: 箱で切り抜くと、斜めの細い物（バー）では箱の大半が背景（マット）になり材質を取り違えた。mask の外は灰色で塗る
    pad = 4
    y0, y1 = max(0, ys.min() - pad), min(img.shape[0], ys.max() + 1 + pad)
    x0, x1 = max(0, xs.min() - pad), min(img.shape[1], xs.max() + 1 + pad)
    crop = img[y0:y1, x0:x1].copy()
    crop[~best["mask"][y0:y1, x0:x1]] = 128
    cv2.imwrite(os.path.join(d, "material_crop.png"), crop)
    _t = time.time()
    best["material"], best["mat_conf"] = classify_material(crop)
    timing.event("snap/siglip (load+classify)", _t, time.time())
    best["mat_conf"] = float(best["mat_conf"])

    _t_save = time.time()
    cv2.imwrite(os.path.join(d, "mask.png"), (best["mask"].astype(np.uint8) * 255))
    np.savez_compressed(os.path.join(d, "cand_masks.npz"), masks=np.stack([c["mask"] for c in cands]))  # 候補の順（pick.json の cands と同じ）
    summary = [{k: v for k, v in c.items() if k not in ("mask", "pts", "pb")} for c in cands]
    json.dump({"cands": summary, "chosen": {k: v for k, v in best.items() if k not in ("mask", "pts", "pb")}},
              open(os.path.join(d, "pick.json"), "w"), indent=1,
              default=lambda o: o.tolist() if isinstance(o, np.ndarray) else (o.item() if isinstance(o, np.generic) else str(o)))
    for f in glob.glob(os.path.join(d, "object_alt_*.npy")):
        os.remove(f)
    for i, c in enumerate([c for c in cands if c is not best][:4]):
        np.save(os.path.join(d, f"object_alt_{i}.npy"), c["pts"].astype(np.float32))

    timing.event("snap/save pick outputs", _t_save, time.time())
    ext = best["pts"].max(0) - best["pts"].min(0)
    return best["pts"].astype(np.float32), (
        f"{best['material']} ({best['mat_conf']:.2f}) {'isolated' if best['isolated'] else 'heap'} "
        f"{len(best['pts'])} pts [{best['src']}] size {ext[0]*100:.0f}x{ext[1]*100:.0f}x{ext[2]*100:.0f} cm "
        f"top {best['top']*100:.1f} cm dist_robot {best['dist_robot']*100:.0f} cm "
        f"(of {len(cands)} candidates, {sum(c['isolated'] for c in cands)} isolated)")


def main(argv=None) -> int:
    timing.startup()
    _t = time.time()
    import cv2
    timing.event("snap/import cv2", _t, time.time())

    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", default="0:480", help="image rows to use v0:v1 (cuts out the arm's own hands at the bottom of the frame)")
    ap.add_argument("--cols", default="0:640")
    ap.add_argument("--zmin", type=float, default=0.30)
    ap.add_argument("--zmax", type=float, default=0.70, help="ignore anything farther than this [m] (stand legs at the back)")
    ap.add_argument("--hmax", type=float, default=0.04, help="only things within this height of the floor [m] (excludes the hands)")
    ap.add_argument("--cap", default=None, help="re-process an existing capture name without taking a new one")
    ap.add_argument("--band", default="0.33,0.56,-0.05,0.35",
                    help="left arm's reach (base frame x0,x1,y0,y1 [m]). Only points inside it are segmented (uses T_base_cam). 'none' to disable")
    ap.add_argument("--no-view", action="store_true", help="don't open a window")
    ap.add_argument("--chest-preset", default=None, help="D435 visual preset for capture.py (one word: Hand; use --capture-args for others)")
    ap.add_argument("--no-hole-fill", action="store_true", help="capture.py --no-hole-fill (keep low-confidence pixels empty)")
    ap.add_argument("--capture-args", default="", help="extra args for capture.py on Thor, e.g. \"--preset Hand --no-hole-fill\" (shiny aluminium)")
    ap.add_argument("--fast", action="store_true", help="short capture (8 warm-up + 4 temporal frames) for back-to-back trials")
    ap.add_argument("--exclude-bright", action="store_true", help="ignore bright unsaturated pixels (aluminium) -> target the wood block next to it")
    ap.add_argument("--exclude-wood", action="store_true", help="ignore wood-coloured pixels (pick the aluminium bar even when a wood block touches it)")
    ap.add_argument("--self-q", default=None, help="今の左腕の関節角（lerobot の度・7 つ）。腕のまわりの点を切り出しの前に消す（腕が写るとき）")
    ap.add_argument("--self-r", type=float, default=0.06, help="--self-q で消す、腕の線からの距離 [m]")
    ap.add_argument("--self-q-right", default=None,
                    help="右腕の関節角（lerobot の度）。右腕のまわりの点も消す（2026-11-23: --target any で休みの右手を拾わない）")
    ap.add_argument("--self-drop", type=float, default=0.0,
                    help=">0 なら、点の半分より多くが腕の線からこの距離 [m] 以内のかたまりを候補から外す（消し残した爪のかけら・"
                         "--self-r を小さくして腕のすぐ横の物を削らないとき）")
    ap.add_argument("--target", default=None,
                    help="拾う物の名前（grasp/config/objects/<名前>.json）。色と大きさが合うかたまりを選び、無ければ object.npy を作らない。"
                         "無指定なら今までどおりいちばん大きいかたまり")
    ap.add_argument("--no-separate", action="store_true", help="--target の separate_from（相手の色の画素を消す）を使わない")
    ap.add_argument("--list-targets", action="store_true", help="登録してある物の名前を出して終わる")
    ap.add_argument("--perceive", default=None, help="host:port of grasp/tools/perceive_server.py: pick the aluminium frame from the "
                    "pile by Grounding DINO + SAM 2 + shape (2026-11-21) instead of the largest cluster")
    ap.add_argument("--alu-thr", type=float, default=0.55, help="minimum aluminium score for --perceive")
    ap.add_argument("--gap-half", type=float, default=0.065, help="--perceive: half the jaw opening across the bar [m]; rigid things closer than this crowd the bar")
    ap.add_argument("--crowd-max", type=float, default=0.10, help="--perceive: bars whose jaw corridor is more than this share filled (1 cm cells) go after the clear ones")
    ap.add_argument("--perceive-target", dest="perceive_target", choices=["alu", "wood"], default="alu",
                    help="--perceive: what to pick. wood = the wood block next to the bars (clear a rigid neighbour first); "
                         "the bars stay rigid obstacles. (Renamed from the old --target, which --target now means for grasp/objects.py)")
    ap.add_argument("--skip-crowded", action="store_true", help="--perceive: fail (no target) when every bar is crowded, instead of trying the least bad")
    ap.add_argument("--alu-height", type=float, default=0.03, help="assumed bar height [m] when its depth is missing (shiny top)")
    ap.add_argument("--yoloe", action="store_true", help="pick the target with YOLOE-26, ranked by the isolated-from-heap / "
                    "nearest-robot / highest-in-heap rule (material plays no part in picking), instead of --perceive's "
                    "Grounding DINO aluminium-only pipeline; the winner is then labelled by material "
                    "(grasp/classification.py, SigLIP) for routing to a bin")
    ap.add_argument("--isolation-gap", type=float, default=0.03, help="--yoloe: min gap [m] to the nearest other candidate's "
                    "footprint (edge to edge, base-frame XY) to count as isolated from the heap")
    ap.add_argument("--exclude-xy", default="", help="--yoloe: 腕の付け根の座標 x,y を ; で並べる。中心がここから --exclude-r 以内の候補を外す（置いた物）")
    ap.add_argument("--border-px", type=int, default=8, help="--yoloe: 画面の縁からこの画素以内に掛かる mask は選ばない（0 で使わない）")
    ap.add_argument("--exclude-r", type=float, default=0.07, help="--exclude-xy の半径 [m]")
    ap.add_argument("--plane-dist", type=float, default=0.008,
                    help="--yoloe: RANSAC の床の面からこの距離 [m] 以内の点を床とみなす（segment.py の plane_dist と同じ）")
    ap.add_argument("--plane-frac", type=float, default=0.6,
                    help="--yoloe: 深度のある画素のうち床の面の上の割合がこれより多い mask は、床・マットとして対象を選ぶ前に外す")
    ap.add_argument("--obj-height", type=float, default=0.02, help="--yoloe: assumed object height [m] when its depth is missing")
    a = ap.parse_args(argv)
    if a.list_targets:
        print("\n".join(objects.names()))
        return 0
    spec = objects.load(a.target) if a.target else None  # 撮る前に名前を確かめる

    os.makedirs(RUNS, exist_ok=True)
    if a.cap:
        name = a.cap
    else:
        r = remote.run("cd ~/grasp_work && o=~/grasp_runs/cap_$(date +%Y%m%d_%H%M%S); "
                       f"{remote.CAMERA_PY} grasp/capture.py --out $o " + ("--warmup 8 --temporal 4 " if a.fast else "")
                       + (f"--preset {a.chest_preset} " if a.chest_preset else "") + ("--no-hole-fill " if a.no_hole_fill else "")
                       + a.capture_args + " >/dev/null && basename $o", timeout=None)
        if r.returncode != 0:
            raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
        name = r.stdout.strip()
        remote.get(f"grasp_runs/{name}", RUNS)
    d = os.path.join(RUNS, name)
    _t = time.time()
    depth = np.load(os.path.join(d, "depth.npy"))
    K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    img = cv2.imread(os.path.join(d, "color.png"))
    img_raw = img.copy()  # 色で見分けるのは描き込む前の画像

    for stale in ("object.npy", "target.json"):  # --cap で撮り直さずに切り直すとき、前の結果を残さない
        if os.path.exists(os.path.join(d, stale)):
            os.remove(os.path.join(d, stale))
    timing.event("snap/load capture", _t, time.time())
    _t = time.time()
    v0, v1 = map(int, a.rows.split(":")); u0, u1 = map(int, a.cols.split(":"))
    dm = np.zeros_like(depth); dm[v0:v1, u0:u1] = depth[v0:v1, u0:u1]
    if spec is not None and not a.no_separate and spec["separate_from"]:  # 触れている相手（例: 木片の隣のアルミ）と分ける
        dm[objects.separate_mask(img, spec)] = 0
    if a.exclude_wood:  # drop wood-coloured pixels (brown hue, saturated) so a touching wood block doesn't merge with the target
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        wood = ((hsv[..., 0] >= 8) & (hsv[..., 0] <= 35) & (hsv[..., 1] >= 30)).astype(np.uint8)
        wood = cv2.morphologyEx(wood, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        wood = cv2.dilate(wood, np.ones((7, 7), np.uint8)).astype(bool)
        dm[wood] = 0
    if a.exclude_bright:  # drop bright unsaturated pixels (aluminium) so the wood block next to the bar is the target
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        br = ((hsv[..., 2] >= 200) & (hsv[..., 1] <= 45)).astype(np.uint8)
        br = cv2.dilate(cv2.morphologyEx(br, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)), np.ones((5, 5), np.uint8)).astype(bool)
        dm[br] = 0
    pts, _ = depth_to_points(dm, K, zmin=a.zmin, zmax=a.zmax)
    tbc = os.environ.get("GRASP_T_BASE_CAM", os.path.join(os.path.dirname(__file__), "..", "config", "T_base_cam.json"))
    if os.environ.get("GRASP_SITE") and "GRASP_T_BASE_CAM" not in os.environ:  # 東京のハンドアイで腕の届く範囲を切らない
        tbc = ""
    timing.event("snap/depth→points", _t, time.time())
    _t = time.time()
    segs = []
    if (a.self_q or a.self_q_right) and os.path.exists(tbc):
        for qs, sd in ((a.self_q, "left"), (a.self_q_right, "right")):
            if qs:
                segs += self_segments([float(v) for v in qs.split(",")], side=sd)
    if segs:
        # ロボット自身の左腕を消す（2026-11-21 北九州）: 机に寝かせた休みの姿勢では左手がカメラに写り、近くに置いた物と
        # ひとかたまりで切り出された。今の関節角から腕の線（関節 3〜7・指先・爪の先）を出し、そこから --self-r 以内の点を消す
        T = np.array(json.load(open(tbc))["T_base_cam"])
        pb = pts @ T[:3, :3].T + T[:3, 3]
        keep = np.ones(len(pts), bool)
        for p0, p1 in segs:
            seg = p1 - p0; t = np.clip(((pb - p0) @ seg) / max(seg @ seg, 1e-9), 0, 1)
            keep &= np.linalg.norm(pb - (p0 + t[:, None] * seg), axis=1) > a.self_r
        print(f"[snap] self-filter: removed {int((~keep).sum())} points of the robot's own arm", flush=True)
        pts = pts[keep]
    if a.band != "none" and os.path.exists(tbc):
        T = np.array(json.load(open(tbc))["T_base_cam"]); x0, x1, y0, y1 = map(float, a.band.split(","))
        pb = pts @ T[:3, :3].T + T[:3, 3]
        floor_like = pb[:, 2] < np.percentile(pb[:, 2], 60)   # keep the floor for plane removal
        inb = (pb[:, 0] > x0) & (pb[:, 0] < x1) & (pb[:, 1] > y0) & (pb[:, 1] < y1)
        pts = pts[inb | floor_like]
    timing.event("snap/self-arm segments + point filters (pinocchio FK)", _t, time.time())
    msg = ""
    _t_pick = time.time()
    if a.yoloe:
        a._self_segs = segs  # pick_target で腕の線から近い候補を外す
        try:
            obj, msg = pick_target(a, img, depth, K, tbc, d)
            np.save(os.path.join(d, "object.npy"), obj)
            uv = obj @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
            for u, v in uv[:: max(1, len(uv) // 1500)]:
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        except RuntimeError as e:
            msg = f"pick failed: {e}"
            if os.path.exists(os.path.join(d, "object.npy")):
                os.remove(os.path.join(d, "object.npy"))
    elif a.perceive:
        try:
            obj, msg = perceive_target(a, img, depth, K, tbc, d)
            np.save(os.path.join(d, "object.npy"), obj)
            uv = obj @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
            for u, v in uv[:: max(1, len(uv) // 1500)]:
                if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                    cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        except RuntimeError as e:
            msg = f"perceive failed: {e}"
            if os.path.exists(os.path.join(d, "object.npy")):
                os.remove(os.path.join(d, "object.npy"))
    else:
      try:
        clusters, dbg = segment_clusters(pts, SegParams(zmax=a.zmax, above_plane_max=a.hmax))
        if segs and a.self_drop > 0:  # 消し残した腕（爪のかけら）のかたまりを外す
            T = np.array(json.load(open(tbc))["T_base_cam"])

            def near_arm(c):
                cb = c @ T[:3, :3].T + T[:3, 3]
                dmin = np.full(len(cb), np.inf)
                for p0, p1 in segs:
                    sg = p1 - p0; t = np.clip(((cb - p0) @ sg) / max(sg @ sg, 1e-9), 0, 1)
                    dmin = np.minimum(dmin, np.linalg.norm(cb - (p0 + t[:, None] * sg), axis=1))
                return float(np.mean(dmin <= a.self_drop))
            fr = [near_arm(c) for c in clusters]
            dropped = [i for i, f in enumerate(fr) if f > 0.5]
            if dropped:
                print(f"[snap] self-filter: 腕のかけらとして外したかたまり {len(dropped)} 個", flush=True)
            clusters = [c for c, f in zip(clusters, fr) if f <= 0.5]
            if not clusters:
                raise RuntimeError("腕のかけらのほかに物が無い")
        best = 0
        if spec is not None:
            for c in clusters:  # 選ばなかったかたまりは黄色（どれと見分けたかを図で見る）
                uv = c @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
                for u, v in uv:
                    if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                        cv2.circle(img, (u, v), 1, (0, 220, 255), -1)
            best, table = objects.pick(clusters, K, img_raw, spec, plane=dbg.get("plane"))
            with open(os.path.join(d, "target.json"), "w") as f:  # どのかたまりを何で選んだか・外したかの記録
                json.dump({"target": spec["name"], "picked": best, "clusters": table,
                           "separate_from": [] if a.no_separate else spec["separate_from"]}, f, indent=2)
            if best is None:
                raise RuntimeError(objects.not_found(spec, table))
        obj = clusters[best]
        uv = obj @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        ext = obj.max(0) - obj.min(0)
        msg = (f"{spec['name']} " if spec else "") + \
              f"object {len(obj)} pts  size {ext[0]*100:.0f}x{ext[1]*100:.0f}x{ext[2]*100:.0f} cm  dist {obj[:, 2].mean():.2f} m"
        np.save(os.path.join(d, "object.npy"), obj)
      except RuntimeError as e:
        msg = f"segment failed: {e}"
    timing.event("snap/target selection (yoloe・perceive・clusters)", _t_pick, time.time(), ok=os.path.exists(os.path.join(d, "object.npy")))
    _t = time.time()
    cv2.rectangle(img, (u0, v0), (u1 - 1, v1 - 1), (255, 200, 0), 1)
    cv2.putText(img, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, msg, (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 1.5), alpha=170), cv2.COLORMAP_JET)
    out = os.path.join(RUNS, "latest.jpg")
    cv2.imwrite(out, np.hstack([img, dv]))
    cv2.imwrite(os.path.join(d, "overlay.jpg"), np.hstack([img, dv]))
    timing.event("snap/overlay images", _t, time.time())
    print(name, "|", msg)
    if not a.no_view:
        running = subprocess.run(["pgrep", "-f", "eog .*latest.jpg"], capture_output=True).returncode == 0
        if not running:  # eog follows changes to the file, so only open it once
            subprocess.Popen(["eog", os.path.abspath(out)], env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
