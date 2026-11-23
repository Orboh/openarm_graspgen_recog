#!/usr/bin/env python3
"""机の上のどこを、左腕・右腕・両方で掴めるかの地図（上から見た図と寸法）。**腕は動かさない・カメラも要らない。**

2026-11-22・作業者: Munenori。両腕で掴むようにしたので、胸のカメラの取り付けを作り直すときに「カメラに写すべき範囲」を決める。

1 つの格子点で「掴める」とするのは、物の中心の高さ（机から --height）に指先の中心（TCP）を置く姿勢のうち、どれか 1 つで
plan.py と同じ決まりが全部通るとき:
  - grasp・pregrasp（近づく向きに 10 cm 手前）・lift（真上に 10 cm）の 3 つの IK が解け、可動域の中
  - pregrasp・lift が grasp から 45° 以内（IK の枝が飛ばない）
  - 爪の先の一番低い角が机から --floor-clear 以上
  - --curobo なら、pregrasp → grasp → lift の途中が本体・机・休んでいるもう片方の腕から 10 mm 以上（server.py を左右とも立てておく）。
    胸のカメラの箱は取り付けを作り直すので既定では見ない（--with-camera で見る）
試す姿勢: 真下からの傾き 0・15・30・45°（--max-tilt まで）× 近づく向き 8 方位 × 指を閉じる向き 6 通り（30° おき・180° 回しは対称）。
低い物は plan.py --raise-to-clear と同じく、爪の角が机に当たる姿勢を近づく向きに沿って最大 --raise-to-clear だけ引いて試す。

    PYTHONPATH=.:~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/tools/reach_map.py --curobo -o grasp/runs/reach_map
    → <out>/reach_map.png（上から見た図）・reach_map.json（格子と寸法）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO)

from grasp.arm import mirror_q  # noqa: E402

URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
PREFER_LEFT = [-17.7, 2.0, 4.7, 28.1, -8.3, -15.5, -28.9]  # plan.py の --prefer の既定
G = {}  # 子のプロセスの持ち物（IK・設定）


def orientations(max_tilt: float):
    """試すグリッパーの向き（3x3・列が Xg 閉じる・Yg 厚み・Zg 近づく）の列。"""
    out = []
    for tilt in [t for t in (0, 15, 30, 45, 60) if t <= max_tilt]:
        for az in ([0] if tilt == 0 else range(0, 360, 45)):
            t, a = np.radians(tilt), np.radians(az)
            z = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])  # 近づく向き（下向き）
            ref = np.array([0.0, 0.0, 1.0]) if tilt else np.array([1.0, 0.0, 0.0])
            x0 = np.cross(ref, z); x0 /= np.linalg.norm(x0)
            y0 = np.cross(z, x0)
            for roll in range(0, 180, 30):
                r = np.radians(roll)
                x = np.cos(r) * x0 + np.sin(r) * y0
                out.append((tilt, np.column_stack([x, np.cross(z, x), z])))
    return out


def init(side, args, gripper):
    from ik.openarm_left_ik import OpenArmIK
    from grasp import plan
    os.environ["GRASP_ARM"] = side
    G.update(side=side, a=args, ik=OpenArmIK(URDF, side=side), T_L7_G=plan.t_l7_g(side), cfg=gripper,
             ori=orientations(args.max_tilt))
    pref = np.deg2rad(PREFER_LEFT if side == "left" else mirror_q(PREFER_LEFT))
    ik = G["ik"]
    rng = np.random.default_rng(0)
    mid = (lambda v: np.asarray(v, float)) if side == "left" else mirror_q
    G["seeds"] = [pref, np.deg2rad(mid([0, 0, 0, 60, 0, 0, 0])), np.deg2rad(mid([-30, 0, 0, 90, 0, 0, 0]))] + \
                 [ik.lo + (ik.hi - ik.lo) * rng.random(7) for _ in range(3)]
    if args.curobo:
        from grasp.curobo import client
        client.set_arm(side)
        G["curobo"] = client


def solve(T, first):
    import pinocchio as pin
    ik = G["ik"]
    Tl7 = T @ np.linalg.inv(G["T_L7_G"])
    best = None
    for sd in ([first] if first is not None else []) + G["seeds"]:
        q, ok, _, _ = ik.solve(pin.SE3(Tl7[:3, :3], Tl7[:3, 3]), sd, position_only=False, max_iter=150)
        if ok and ik.within_limits(q):
            d = 0.0 if first is None else float(np.max(np.abs(q - first)))
            if best is None or d < best[0]:
                best = (d, q)
            if first is None:
                break
    return None if best is None else best[1]


def cell(xy):
    """1 つの格子点 → (掴める, 一番小さい傾き [deg] か None)。"""
    a, cfg = G["a"], G["cfg"]
    tcp = np.array([xy[0], xy[1], a.floor_z + a.height])
    tcp_off = cfg["flange_to_tip_m"] - 0.015
    for tilt, R in G["ori"]:
        Tg = np.eye(4); Tg[:3, :3] = R; Tg[:3, 3] = tcp - R[:, 2] * tcp_off  # 基準（手の付け根の面）の位置
        tip = Tg[:3, 3] + cfg["flange_to_tip_m"] * R[:, 2]
        low = min((tip + sx * 0.058 * R[:, 0] + sy * 0.024 * R[:, 1])[2] for sx in (-1, 1) for sy in (-1, 1))
        if low < a.floor_z + a.floor_clear:  # plan.py の raise_to_clear と同じ: 近づく向きに沿って引く
            back = (a.floor_z + a.floor_clear - low) / max(-R[2, 2], 0.3)
            if back > a.raise_to_clear:
                continue
            Tg[:3, 3] -= back * R[:, 2]
        qg = solve(Tg, None)
        if qg is None:
            continue
        Tp = Tg.copy(); Tp[:3, 3] -= 0.10 * R[:, 2]
        Tl = Tg.copy(); Tl[:3, 3] += [0, 0, 0.10]
        qp, ql = solve(Tp, qg), solve(Tl, qg)
        if qp is None or ql is None:
            continue
        if max(np.max(np.abs(qp - qg)), np.max(np.abs(ql - qg))) > np.radians(45):
            continue
        if "curobo" in G:
            segs = G["curobo"].check_path([np.rad2deg(qp), np.rad2deg(qg), np.rad2deg(ql)], n=15)
            if not a.with_camera:
                segs = [{k: v for k, v in sg.items() if not k.startswith("chest_camera")} for sg in segs]
            if G["curobo"].violations_from(segs):
                continue
        return True, tilt
    return False, None


def run_side(side, pts, args, gripper, procs):
    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(procs, initializer=init, initargs=(side, args, gripper)) as pool:
        return pool.map(cell, pts, chunksize=4)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", default="grasp/runs/reach_map")
    ap.add_argument("--x", default="0.0,0.75", help="前後の範囲 [m]（腕の付け根の座標・前が +）")
    ap.add_argument("--y", default="-0.6,0.6", help="左右の範囲 [m]（左が +）")
    ap.add_argument("--step", type=float, default=0.025)
    ap.add_argument("--height", type=float, default=0.03, help="指先の中心の机からの高さ [m]（物の中心）")
    ap.add_argument("--floor-clear", type=float, default=0.006)
    ap.add_argument("--max-tilt", type=float, default=45.0)
    ap.add_argument("--raise-to-clear", type=float, default=0.03, help="爪の角が机に当たる姿勢を引いてよい量 [m]（試行と同じ）")
    ap.add_argument("--curobo", action="store_true", help="本体・机・もう片方の腕との当たりも見る（server.py を左右とも立てる）")
    ap.add_argument("--with-camera", action="store_true", help="胸のカメラの箱も当たりに入れる")
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 4))
    a = ap.parse_args()
    a.floor_z = float(os.environ.get("GRASP_FLOOR_Z", "0.235"))
    gripper = json.load(open(os.environ.get("GRASP_GRIPPER_JSON", os.path.join(REPO, "grasp", "config", "openarm_left_gripper.json"))))
    x0, x1 = (float(v) for v in a.x.split(",")); y0, y1 = (float(v) for v in a.y.split(","))
    xs = np.round(np.arange(x0, x1 + 1e-9, a.step), 4); ys = np.round(np.arange(y0, y1 + 1e-9, a.step), 4)
    pts = [(x, y) for x in xs for y in ys]
    os.makedirs(a.out, exist_ok=True)
    res = {}
    for side in ("left", "right"):
        t0 = time.time()
        r = run_side(side, pts, a, gripper, a.procs)
        res[side] = np.array([ok for ok, _ in r]).reshape(len(xs), len(ys))
        print(f"[reach] {side}: 掴める点 {int(res[side].sum())} / {len(pts)}（{time.time() - t0:.0f} 秒）", flush=True)
    L, R = res["left"], res["right"]
    both, union = L & R, L | R
    cell_a = a.step ** 2

    def extent(m):
        if not m.any():
            return None
        ix, iy = np.where(m)
        return {"x": [float(xs[ix.min()]), float(xs[ix.max()])], "y": [float(ys[iy.min()]), float(ys[iy.max()])]}

    rows = []
    for i, x in enumerate(xs):
        if union[i].any():
            def span(m):
                j = np.where(m[i])[0]
                return [float(ys[j.min()]), float(ys[j.max()])] if len(j) else None
            rows.append({"x": float(x), "left": span(L), "right": span(R), "both": span(both), "union": span(union)})
    summary = {k: {"area_m2": float(m.sum() * cell_a), "extent": extent(m)} for k, m in
               (("left", L), ("right", R), ("both", both), ("union", union))}
    json.dump({"params": {k: v for k, v in vars(a).items()}, "xs": xs.tolist(), "ys": ys.tolist(),
               "left": L.tolist(), "right": R.tolist(), "summary": summary, "rows": rows},
              open(os.path.join(a.out, "reach_map.json"), "w"), indent=1)

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    code = np.zeros_like(L, int); code[L & ~R] = 1; code[R & ~L] = 2; code[both] = 3
    fig, ax = plt.subplots(figsize=(8, 6.5))
    ax.imshow(code, origin="lower", cmap=ListedColormap(["#f4f4f4", "#4e79a7", "#f28e2b", "#59a14f"]), vmin=0, vmax=3,
              extent=[ys[0] - a.step / 2, ys[-1] + a.step / 2, xs[0] - a.step / 2, xs[-1] + a.step / 2], aspect="equal")
    ax.invert_xaxis()  # 後ろからロボットと同じ向きに見る（左腕が図の左）
    for yy, lab in ((0.0935, "L shoulder"), (-0.0935, "R shoulder")):
        ax.plot([yy], [0.0], "k^", ms=8); ax.annotate(lab, (yy, 0.0), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
    ax.set_xlabel("y [m]  (+ = robot's left)"); ax.set_ylabel("x [m]  (forward)")
    s = summary
    ax.set_title(f"reach at TCP height {a.height * 100:.0f} cm above table, tilt <= {a.max_tilt:.0f} deg"
                 + (" + cuRobo" if a.curobo else "") + "\n"
                 f"left {s['left']['area_m2']:.3f} m2 / right {s['right']['area_m2']:.3f} / both {s['both']['area_m2']:.3f} / union {s['union']['area_m2']:.3f}",
                 fontsize=9)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#4e79a7", label="left only"), Patch(color="#f28e2b", label="right only"),
                       Patch(color="#59a14f", label="both")], loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "reach_map.png"), dpi=110)
    for k in ("union", "both", "left", "right"):
        e = s[k]["extent"]
        print(f"[reach] {k:5s}: 面積 {s[k]['area_m2']:.3f} m²" + (f"・前後 {e['x'][0]:.2f}〜{e['x'][1]:.2f} m・左右 {e['y'][0]:+.2f}〜{e['y'][1]:+.2f} m" if e else "・無し"))
    print(f"[reach] 書いた: {a.out}/reach_map.png・reach_map.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
