"""cuRobo の当たり判定のサーバー（grasp/curobo/server.py）を呼ぶ。pyzmq と msgpack だけで動く（GraspGenX・lerobot の venv から）。

GRASP_CUROBO（"host:port"）があれば使う。無ければ当たりを調べない（東京の既定）。
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

ADDR = os.environ.get("GRASP_CUROBO", "")
# 箱との最小のすき間 [m]。机は指を除いた腕と手の本体で見る（爪の先と机は plan.py の床の確認がもっと細かく見る）
MARGIN = float(os.environ.get("GRASP_CUROBO_MARGIN", "0.01"))
TABLE_MARGIN = float(os.environ.get("GRASP_CUROBO_TABLE_MARGIN", "0.0"))


def enabled() -> bool:
    if ADDR and os.environ.get("GRASP_ARM", "left") != "left":
        # the cuRobo arm model (openarm_left_umi) is the left arm only (2026-11-21 Tokyo: right arm via GRASP_ARM=right)
        raise SystemExit("[curobo] GRASP_CUROBO は左腕のモデルだけ。GRASP_ARM=right では外す（unset GRASP_CUROBO）")
    return bool(ADDR)


def _call(req: Dict, timeout_ms: int = 10000) -> Dict:
    import msgpack
    import zmq
    host, _, port = ADDR.partition(":")
    sock = zmq.Context.instance().socket(zmq.REQ)
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
