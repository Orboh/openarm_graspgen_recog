"""cuRobo の当たり判定のサーバー（grasp/curobo/server.py）を呼ぶ。pyzmq と msgpack だけで動く（GraspGenX・lerobot の venv から）。

GRASP_CUROBO（"host:port"）があれば使う。無ければ当たりを調べない（東京の既定）。
右腕（GRASP_ARM=right）は GRASP_CUROBO_RIGHT のサーバー（server.py --arm right・既定 port 5559）に聞く（2026-11-22 Munenori）。
右のサーバーが無いのに右腕で計画・実行するときは止まる（左のサーバーの答えを右腕に使わない）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

def _addr_for(side: str) -> str:
    if side == "right":
        if os.environ.get("GRASP_CUROBO") and not os.environ.get("GRASP_CUROBO_RIGHT"):
            raise SystemExit("[curobo] 右腕なのに GRASP_CUROBO_RIGHT が無い（左のサーバーで右腕の当たりを見ない）。"
                             "server.py --arm right を立てて置き場の env に書く")
        return os.environ.get("GRASP_CUROBO_RIGHT", "")
    return os.environ.get("GRASP_CUROBO", "")


ADDR = _addr_for(os.environ.get("GRASP_ARM", "left"))


def set_arm(side: str) -> None:
    """この後の問い合わせを side の腕のサーバーに向ける（plan.py の腕ごとの子のプロセス・run_grasp.py が呼ぶ）。"""
    global ADDR
    ADDR = _addr_for(side)
# 箱との最小のすき間 [m]。机は指を除いた腕と手の本体で見る（爪の先と机は plan.py の床の確認がもっと細かく見る）
MARGIN = float(os.environ.get("GRASP_CUROBO_MARGIN", "0.01"))
TABLE_MARGIN = float(os.environ.get("GRASP_CUROBO_TABLE_MARGIN", "0.0"))


def enabled() -> bool:
    return bool(ADDR)


_CTX = (None, None)  # (pid, zmq.Context)


def _context():
    """このプロセスの ZMQ の context。fork した子（plan.py --arm both）は親の context を使えないので作り直す。"""
    global _CTX
    import zmq
    if _CTX[0] != os.getpid():
        _CTX = (os.getpid(), zmq.Context())
    return _CTX[1]


def _call(req: Dict, timeout_ms: int = 10000) -> Dict:
    import time
    _t0 = time.time()  # 段ごとの時間（記録だけ・2026-11-25 Shreyan）: 問い合わせ 1 回の往復を op ごとに残す
    import msgpack
    import zmq
    host, _, port = ADDR.partition(":")
    sock = _context().socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, timeout_ms); sock.setsockopt(zmq.LINGER, 0)
    sock.connect(f"tcp://{host or 'localhost'}:{port or '5558'}")
    try:
        sock.send(msgpack.packb(req, use_bin_type=True))
        rep = msgpack.unpackb(sock.recv(), raw=False)
    except Exception as e:  # noqa: BLE001  つながらないときは「調べられない」= 動かさない側に倒す
        raise SystemExit(f"[curobo] サーバー（{ADDR}）に聞けない: {type(e).__name__}: {e}。"
                         "grasp/curobo/server.py を立てるか、GRASP_CUROBO を外す")
    finally:
        sock.close()
        try:
            from grasp import timing
            timing.event(f"curobo rpc/{req.get('op')}", _t0, time.time(), server=ADDR)
        except Exception:  # noqa: BLE001  記録のために問い合わせを止めない
            pass
    if "error" in rep:
        raise SystemExit(f"[curobo] サーバーの誤り: {rep['error']}")
    return rep


def check_path(waypoints_deg: Sequence[Sequence[float]], n: int = 30) -> List[Dict[str, list]]:
    """区間ごとの {箱: [距離, link, s], 箱/arm: [...]}。"""
    return _call({"op": "check_path", "waypoints_deg": [[float(v) for v in q] for q in waypoints_deg], "n": n})["segments"]


def violations(waypoints_deg: Sequence[Sequence[float]], names: Optional[Sequence[str]] = None, n: int = 30,
               margin: float = MARGIN, table_margin: Optional[float] = TABLE_MARGIN) -> List[Tuple[str, str, float, str, float]]:
    """すき間が足りない所の一覧 [(区間名, 物, 距離, link, s)]。table_margin=None なら机を見ない（休みの姿勢の区間など）。"""
    return violations_from(check_path(waypoints_deg, n), names, margin, table_margin)


def violations_from(segs: List[Dict[str, list]], names: Optional[Sequence[str]] = None, margin: float = MARGIN,
                    table_margin: Optional[float] = TABLE_MARGIN) -> List[Tuple[str, str, float, str, float]]:
    """check_path の結果から、すき間が足りない所を拾う（サーバーの回り込みの確かめと同じ決まり）。
    机は指を除いた腕と手の本体で table_margin、ほかの物は全部の球で margin。"""
    names = list(names) if names else [f"{i}→{i + 1}" for i in range(len(segs))]
    bad = []
    for nm, sg in zip(names, segs):
        for key, (d, link, s) in sg.items():
            if key == "table/arm":
                if table_margin is not None and d < table_margin:
                    bad.append((nm, "table", d, link, s))
            elif key.endswith("/arm") or key == "table":
                continue
            elif d < margin:
                bad.append((nm, key, d, link, s))
    return bad


def describe(bad) -> str:
    return "; ".join(f"{nm}: {box} {d * 1000:+.0f} mm（{link}・s={s:.2f}）" for nm, box, d, link, s in bad)


def detour(q_from: Sequence[float], q_to: Sequence[float], table_margin: Optional[float] = TABLE_MARGIN,
           margin: float = MARGIN, floor: Optional[Dict[str, float]] = None) -> Optional[List[List[float]]]:
    """q_from から q_to へ、周りの物をよけて回り込む経由点（[deg]・始点を除き終点を含む）。作れなければ None。
    経由点の間はどれも関節の直線で当たらず、floor（{"tcp_min_z", "l7_min_z", "tcp_len"}）の高さより下がらないことを、
    サーバーが確かめてある。"""
    rep = _call({"op": "detour", "q_from": [float(v) for v in q_from], "q_to": [float(v) for v in q_to],
                 "margin": margin, "table_margin": table_margin, "floor": floor}, timeout_ms=60000)
    return rep.get("vias")


def plan_grasp(q_start_deg: Sequence[float], T_grasp, approach: float = 0.10, lift: float = 0.10,
               target: Optional[Dict] = None) -> Dict:
    """cuRobo の plan_grasp（近づく・差し込む・持ち上げるの細かい軌道 [deg]）。target は拾う物の箱（近づく区間だけよける）。
    返り値 {"success", "status", "approach", "grasp", "lift"}。初回は CUDA graph を作るので時間がかかる。"""
    return _call({"op": "plan_grasp", "q_start_deg": [float(v) for v in q_start_deg],
                  "T_grasp": [[float(v) for v in row] for row in T_grasp], "approach": float(approach), "lift": float(lift),
                  "target": target}, timeout_ms=180000)


def set_obstacle_points(points, arm_q_deg: Sequence[Sequence[float]], cam_pos: Sequence[float], floor_z: float,
                        arm_margin: float = 0.03, exclude: Sequence[Sequence[float]] = (), **box_kw) -> Dict:
    """カメラで見た点（腕の付け根の座標・机より上・掴む物を除いた物）から障害物の箱を作ってサーバーに入れる。
    撮影時の腕の姿勢 arm_q_deg のまわり arm_margin の点は腕自身として消す。exclude の球 [x, y, z, r] の中の点も消す
    （持っている物など）。box_kw は obstacles.points_to_boxes の pad・back・voxel・min_voxels・to_floor と quiet。
    返り値 {"boxes", "n_in", "n_after_arm"}。"""
    return _call({"op": "set_points", "points": [[float(v) for v in p] for p in points],
                  "arm_q_deg": [[float(v) for v in q] for q in arm_q_deg], "arm_margin": arm_margin,
                  "exclude": [[float(v) for v in e] for e in exclude],
                  **({"exclude_points": box_kw.pop("exclude_points"), "exclude_r": box_kw.pop("exclude_r", 0.03)}
                     if "exclude_points" in box_kw else {}),
                  "cam_pos": [float(v) for v in cam_pos], "floor_z": float(floor_z), **box_kw}, timeout_ms=30000)


def set_obstacle_boxes(boxes: List[Dict]) -> None:
    """障害物の箱をそのまま入れ替える（記録の obstacles.json を送り直す・空なら障害物なし）。"""
    _call({"op": "set_boxes", "boxes": boxes}, timeout_ms=30000)


def ik(T_link7, seed_deg=None, return_seeds: int = 4, lm_only: bool = False):
    """link7 の姿勢 (N, 4, 4) を cuRobo の IK でまとめて解く → (関節角 (N, k, 7) [deg], 成功 (N, k))。
    周りの物・カメラで見た障害物・もう片方の腕・自己干渉に当たる解は成功にならない（grasp/curobo/ik.py）。"""
    import numpy as np
    T = np.asarray(T_link7, float).reshape(-1, 4, 4)
    rep = _call({"op": "ik", "T": T.tolist(), "seed_deg": None if seed_deg is None else np.asarray(seed_deg, float).tolist(),
                 "return_seeds": int(return_seeds), **({"lm_only": True} if lm_only else {})}, timeout_ms=60000)
    return np.asarray(rep["q_deg"], float), np.asarray(rep["ok"], bool)


def set_other_arm(q_deg: Optional[Sequence[float]]) -> None:
    """休んでいるもう片方の腕の姿勢（lerobot の度）を入れ替える。None で外す。起動時はもう片方の休みの姿勢が入っている。"""
    _call({"op": "set_other_arm", "q_deg": None if q_deg is None else [float(v) for v in q_deg]}, timeout_ms=30000)
