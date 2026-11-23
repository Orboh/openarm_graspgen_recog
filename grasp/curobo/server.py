#!/usr/bin/env python3
"""cuRobo の当たり判定のサーバー（port 5558・ZMQ REQ/REP・msgpack）。**腕は動かさない。**
周りの物は箱（机・カメラ）とメッシュ（純正の本体）。

cuRobo は作って温めるのに時間がかかるので常駐させ、plan.py（GraspGenX の venv）と run_grasp.py（lerobot の venv）から
grasp/curobo/client.py で呼ぶ。周りの物は起動時に GRASP_WORLD_YML と GRASP_FLOOR_Z から読む（置き場の env を先に読む）。

    set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
    ~/work/curobo/.venv/bin/python grasp/curobo/server.py

要求:
  {"op": "ping"} → {"world": [箱の名前...]}
  {"op": "check_path", "waypoints_deg": [[j1..j7], ...], "n": 30}
      → {"segments": [{"<箱>": [距離 m, link, s], ...}, ...], "world": [...]}
      隣り合う経由点の間を関節の直線で n 点に分け、周りの物ごとに一番近い球の表面からの距離（負はめり込み）。
      link は "openarm_left_" を外した名前。"<箱>" ごとに、指（umi_finger_*）を除いた値も "<箱>/arm" に入れる
  {"op": "detour", "q_from": [..7], "q_to": [..7], "margin": 0.01, "table_margin": 0.0 か None,
   "floor": {"tcp_min_z", "l7_min_z", "tcp_len"} か None（run_grasp.py の床の確認の線。経由点の間もこれを守る）}
      → {"vias": [[..7], ...]}（始点を除き終点を含む。経由点の間は直線で当たらないと確かめてある）か {"vias": None, "why": ...}
      直線で当たらなければ終点だけを返す。当たるときは cuRobo の MotionPlanner で回り込む（grasp/curobo/planner.py）
  {"op": "set_points", "points": [[x, y, z], ...], "arm_q_deg": [[..7], ...], "arm_margin": 0.03, "cam_pos": [x, y, z], "floor_z": z}
      → {"boxes": [...], "n_in", "n_after_arm"}。カメラで見た点（机より上・掴む物を除いた物）から、撮影時の腕のまわりを消し、
      塊ごとの箱（grasp/curobo/obstacles.py）にして、置き場の箱に足す（前の障害物は入れ替える）
  {"op": "set_boxes", "boxes": [...]} → 障害物の箱をそのまま入れ替える（run_grasp.py が記録の obstacles.json を送り直す）
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from grasp.curobo.client import violations_from  # noqa: E402
from grasp.curobo.model import ArmSpheres, MeshDistance, load_world, obstacle_distances  # noqa: E402


def check_path(arm: ArmSpheres, world, meshes, waypoints_deg, n: int):
    """経由点の区間ごとに、周りの物ごとの一番近い距離（全部の球・指を除いた球）。"""
    W = np.asarray(waypoints_deg, float)
    if W.ndim != 2 or W.shape[1] != 7 or len(W) < 1:
        raise ValueError(f"waypoints_deg は (N>=1, 7)。来たのは {W.shape}")
    if len(W) == 1:
        W = np.vstack([W, W])
    s = np.linspace(0, 1, n)
    Q = (W[:-1, None, :] + (W[1:] - W[:-1])[:, None, :] * s[None, :, None]).reshape(-1, 7)  # (seg*n, 7)
    S = arm.spheres(np.deg2rad(Q))  # (seg*n, S, 4)
    links = np.array([l.replace("openarm_left_", "") for l in arm.sphere_links])
    not_finger = np.array([not l.startswith("umi_finger") for l in links])
    nseg = len(W) - 1
    out = [dict() for _ in range(nseg)]
    for name, D in obstacle_distances(S, world, meshes, arm.sphere_links).items():
        D = D.reshape(nseg, n, -1)
        for key, mask in ((name, np.ones(len(links), bool)), (name + "/arm", not_finger)):
            Dm = np.where(mask[None, None, :], D, np.inf)
            for k in range(nseg):
                i, j = np.unravel_index(np.argmin(Dm[k]), Dm[k].shape)
                out[k][key] = [float(Dm[k, i, j]), str(links[j]), float(s[i])]
    return out


def detour(arm, world, meshes, planner, req):
    """回り込む経由点を作る。cuRobo の経路を、こちらの判定（violations_from）で直線で通れる所までまとめる。"""
    from grasp.curobo.planner import shortcut
    if planner is None:
        return {"vias": None, "why": "サーバーを --no-detour で立てている"}
    q0, q1 = np.asarray(req["q_from"], float), np.asarray(req["q_to"], float)
    margin = float(req.get("margin", 0.01))
    tm = req.get("table_margin")
    tm = None if tm is None else float(tm)

    floor = req.get("floor")  # run_grasp.py の床の確認と同じ線: {"tcp_min_z", "l7_min_z", "tcp_len"}（腕の付け根の座標 [m]）

    def floor_ok(qa, qb):
        if not floor:
            return True
        Q = qa[None] + (qb - qa)[None] * np.linspace(0, 1, 30)[:, None]
        T = arm.link7_pose(np.deg2rad(Q))
        tcp_z = T[:, 2, 3] + T[:, 2, 2] * float(floor["tcp_len"])
        return bool(np.all(T[:, 2, 3] >= float(floor["l7_min_z"])) and np.all(tcp_z >= float(floor["tcp_min_z"])))

    def seg_ok(qa, qb):
        qa, qb = np.asarray(qa, float), np.asarray(qb, float)
        return floor_ok(qa, qb) and not violations_from(check_path(arm, world, meshes, [qa, qb], 30), margin=margin, table_margin=tm)

    for k, q in (("始点", q0), ("終点", q1)):
        if not seg_ok(q, q):
            return {"vias": None, "why": f"{k}の姿勢がもう周りの物に近すぎる"}
    if seg_ok(q0, q1):
        return {"vias": [q1.tolist()]}
    P = planner.dense(q0, q1)
    if P is None:
        return {"vias": None, "why": "cuRobo が経路を作れなかった"}
    P[0], P[-1] = q0, q1  # 端は頼まれた姿勢そのものにする（cuRobo の補間の端の丸め）
    vias = shortcut(P, seg_ok)
    if vias is None:
        return {"vias": None, "why": "cuRobo の経路がこちらの判定で当たる"}
    return {"vias": vias}


def set_points(arm: ArmSpheres, req) -> tuple:
    """カメラの点から障害物の箱を作る。撮影時の腕の姿勢（arm_q_deg・複数可）の球から arm_margin 以内の点は腕自身として消す。"""
    from grasp.curobo.obstacles import points_to_boxes
    P = np.asarray(req["points"], float).reshape(-1, 3)
    n_in = len(P)
    margin = float(req.get("arm_margin", 0.03))
    for q in req.get("arm_q_deg") or []:
        S = arm.spheres(np.deg2rad([q]))[0]
        S = S[S[:, 3] > 0]
        d = np.linalg.norm(P[:, None, :] - S[None, :, :3], axis=-1) - S[None, :, 3]
        P = P[d.min(1) > margin]
    for x, y, z, r in req.get("exclude") or []:  # 持っている物のまわりなど
        P = P[np.linalg.norm(P - [x, y, z], axis=1) > r]
    if req.get("exclude_points") and len(P):  # 持っている物の点（手にくっつけて動かした物）のまわり
        from scipy.spatial import cKDTree
        d, _ = cKDTree(np.asarray(req["exclude_points"], float)).query(P, k=1)
        P = P[d > float(req.get("exclude_r", 0.03))]
    kw = {k: req[k] for k in ("pad", "back", "voxel", "min_voxels", "to_floor") if k in req}
    boxes = points_to_boxes(P, req["cam_pos"], float(req["floor_z"]), **kw)
    return boxes, {"n_in": n_in, "n_after_arm": int(len(P))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=int(os.environ.get("GRASP_CUROBO_PORT", "5558")))
    ap.add_argument("--world", default=None, help="置き場の箱の YAML（無ければ GRASP_WORLD_YML）")
    ap.add_argument("--no-detour", action="store_true", help="回り込む経路（cuRobo の MotionPlanner）を作らない（起動が速い・GPU を使わない）")
    a = ap.parse_args()
    import msgpack
    import zmq
    arm = ArmSpheres()
    world = load_world(a.world)
    base_cuboids = list(world["cuboid"])  # 置き場の箱。カメラで見た障害物（seen_*）はこの後ろに足し、要求ごとに入れ替える
    meshes = [MeshDistance(m) for m in world["mesh"]]
    names = [c["name"] for c in world["cuboid"]] + [m.name for m in meshes]
    check_path(arm, world, meshes, [[0] * 7, [10] * 7], 30)  # 温める（初回の Warp のコンパイル）
    planner, planner_dirty = None, False
    if not a.no_detour:
        from grasp.curobo.planner import DetourPlanner
        t0 = time.time()
        planner = DetourPlanner(world)
        print(f"[curobo] 回り込みの MotionPlanner を作って温めた（{time.time() - t0:.0f} 秒）", flush=True)
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{a.port}")
    print(f"[curobo] 箱 {names}・球 {len(arm.sphere_links)} 個。:{a.port} で待つ", flush=True)
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            t0 = time.time()
            op = req.get("op")
            if op == "ping":
                rep = {"world": names}
            elif op == "check_path":
                segs = check_path(arm, world, meshes, req["waypoints_deg"], int(req.get("n", 30)))
                rep = {"segments": segs, "world": names}
                worst = min(v[0] for sg in segs for v in sg.values())
                print(f"[curobo] check_path {len(segs)} 区間・一番近い {worst*1000:+.0f} mm（{time.time()-t0:.3f} 秒）", flush=True)
            elif op in ("set_points", "set_boxes"):
                if op == "set_points":
                    boxes, info = set_points(arm, req)
                else:
                    boxes, info = list(req.get("boxes") or []), {}
                world["cuboid"] = base_cuboids + boxes
                planner_dirty = True  # 経路計画の障害物は、回り込みを作るときに入れ替える（毎フレームだと重い）
                names = [c["name"] for c in world["cuboid"]] + [m.name for m in meshes]
                rep = {"boxes": boxes, **info}
                if not req.get("quiet"):
                    print(f"[curobo] {op}: 障害物の箱 {len(boxes)} 個 " + " ".join(
                        f"{b['name']}(中心 {np.round(b['pose'][:3], 2).tolist()}・大きさ {np.round(b['dims'], 2).tolist()})" for b in boxes)
                        + (f"（点 {info['n_in']} → 腕を消して {info['n_after_arm']}）" if info else ""), flush=True)
            elif op == "detour":
                if planner is not None and planner_dirty:
                    planner.update_world(world)
                    planner_dirty = False
                rep = detour(arm, world, meshes, planner, req)
                print("[curobo] detour: " + (f"経由点 {len(rep['vias'])} 個" if rep.get("vias") else f"作れない（{rep.get('why')}）")
                      + f"（{time.time()-t0:.2f} 秒）", flush=True)
            else:
                rep = {"error": f"知らない op: {op}"}
        except Exception as e:  # 1 つの悪い要求でサーバーを落とさない
            rep = {"error": f"{type(e).__name__}: {e}"}
            print(f"[curobo] error: {rep['error']}", flush=True)
        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    raise SystemExit(main())
