#!/usr/bin/env python3
"""Capture one frame with the chest D435i on Thor → pull it to this PC → cut out the object → show it on this PC's screen.

Each run overwrites grasp/runs/latest.jpg, and that window (eog) follows it automatically.
Left = color + the cut-out object (red) + ROI box (light blue); right = depth.

    PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python grasp/tools/thor_snap.py
    # ROI (image rows and distance) can be narrowed:
    ... thor_snap.py --rows 120:350 --zmin 0.40 --zmax 0.75

The arm does not move (it only uses the camera). Can't run while record / stream_chest holds the camera.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from grasp import remote  # noqa: E402
from grasp.capture import depth_to_points  # noqa: E402
from grasp.segment import SegParams, segment_largest_object  # noqa: E402

THOR = remote.HOST  # 既定は東京の Thor。腕とカメラがこの PC にあるなら GRASP_ROBOT_HOST=local（grasp/remote.py）


def self_segments(q_deg):
    """左腕の関節 3〜7・指先の中心・爪の先を結ぶ線分（腕の付け根の座標）。--self-q で腕の点を消すのに使う。"""
    from grasp import plan as P  # pinocchio・爪の寸法（GRASP_GRIPPER_JSON）を使う
    from ik.openarm_left_ik import OpenArmLeftIK
    ik = OpenArmLeftIK()
    pts = P.arm_points(ik, np.deg2rad(q_deg))  # 関節 3〜7・その間の中点・指先の中心・爪の先
    chain = [pts[0], pts[1], pts[2], pts[3], pts[4], pts[-2], pts[-1]]
    return list(zip(chain[:-1], chain[1:]))
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
        if a.target == "wood" and "error" not in rep:
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


def main(argv=None) -> int:
    import cv2

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
    ap.add_argument("--perceive", default=None, help="host:port of grasp/tools/perceive_server.py: pick the aluminium frame from the "
                    "pile by Grounding DINO + SAM 2 + shape (2026-11-21) instead of the largest cluster")
    ap.add_argument("--alu-thr", type=float, default=0.55, help="minimum aluminium score for --perceive")
    ap.add_argument("--gap-half", type=float, default=0.065, help="--perceive: half the jaw opening across the bar [m]; rigid things closer than this crowd the bar")
    ap.add_argument("--crowd-max", type=float, default=0.10, help="--perceive: bars whose jaw corridor is more than this share filled (1 cm cells) go after the clear ones")
    ap.add_argument("--target", choices=["alu", "wood"], default="alu", help="--perceive: what to pick. wood = the wood block "
                    "next to the bars (clear a rigid neighbour first); the bars stay rigid obstacles")
    ap.add_argument("--skip-crowded", action="store_true", help="--perceive: fail (no target) when every bar is crowded, instead of trying the least bad")
    ap.add_argument("--alu-height", type=float, default=0.03, help="assumed bar height [m] when its depth is missing (shiny top)")
    ap.add_argument("--self-q", default=None, help="今の左腕の関節角（lerobot の度・7 つ）。腕のまわりの点を切り出しの前に消す（腕が写るとき）")
    ap.add_argument("--self-r", type=float, default=0.06, help="--self-q で消す、腕の線からの距離 [m]")
    a = ap.parse_args(argv)

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
    depth = np.load(os.path.join(d, "depth.npy"))
    K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
    img = cv2.imread(os.path.join(d, "color.png"))

    v0, v1 = map(int, a.rows.split(":")); u0, u1 = map(int, a.cols.split(":"))
    dm = np.zeros_like(depth); dm[v0:v1, u0:u1] = depth[v0:v1, u0:u1]
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
    if a.self_q and os.path.exists(tbc):
        # ロボット自身の左腕を消す（2026-11-21 北九州）: 机に寝かせた休みの姿勢では左手がカメラに写り、近くに置いた物と
        # ひとかたまりで切り出された。今の関節角から腕の線（関節 3〜7・指先・爪の先）を出し、そこから --self-r 以内の点を消す
        T = np.array(json.load(open(tbc))["T_base_cam"])
        pb = pts @ T[:3, :3].T + T[:3, 3]
        keep = np.ones(len(pts), bool)
        for p0, p1 in self_segments([float(v) for v in a.self_q.split(",")]):
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
    msg = ""
    if a.perceive:
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
        obj, dbg = segment_largest_object(pts, SegParams(zmax=a.zmax, above_plane_max=a.hmax), return_debug=True)
        uv = obj @ K.T; uv = (uv[:, :2] / uv[:, 2:]).astype(int)
        for u, v in uv:
            if 0 <= u < img.shape[1] and 0 <= v < img.shape[0]:
                cv2.circle(img, (u, v), 1, (0, 0, 255), -1)
        ext = obj.max(0) - obj.min(0)
        msg = f"object {len(obj)} pts  size {ext[0]*100:.0f}x{ext[1]*100:.0f}x{ext[2]*100:.0f} cm  dist {obj[:, 2].mean():.2f} m"
        np.save(os.path.join(d, "object.npy"), obj)
      except RuntimeError as e:
        msg = f"segment failed: {e}"
    cv2.rectangle(img, (u0, v0), (u1 - 1, v1 - 1), (255, 200, 0), 1)
    cv2.putText(img, name, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(img, msg, (6, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, 1.5), alpha=170), cv2.COLORMAP_JET)
    out = os.path.join(RUNS, "latest.jpg")
    cv2.imwrite(out, np.hstack([img, dv]))
    cv2.imwrite(os.path.join(d, "overlay.jpg"), np.hstack([img, dv]))
    print(name, "|", msg)
    if not a.no_view:
        running = subprocess.run(["pgrep", "-f", "eog .*latest.jpg"], capture_output=True).returncode == 0
        if not running:  # eog follows changes to the file, so only open it once
            subprocess.Popen(["eog", os.path.abspath(out)], env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":0")},
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
