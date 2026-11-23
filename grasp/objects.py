"""把持の対象を名前で選ぶ（登録した物の色と大きさ・2026-11-22 Munenori）。

登録は grasp/config/objects/<名前>.json（GRASP_OBJECTS_DIR で差し替え）。1 つの物に:
- hsv: 色の範囲（OpenCV の HSV・H は 0〜179）
- min_color_frac: かたまりの点のうち、この色の画素に写る割合の下限
- extent_max_m・extent_min_m: かたまりの外形（カメラの座標・長い辺と短い辺）[m] の範囲。extent_mid_m（あれば）は真ん中の辺
- separate_from: 一緒に置いたときに、切り出しの前に深度から消す相手の物の名前（触れていても別のかたまりにする）

選び方: 机の上のかたまりを全部出し（segment.segment_clusters）、色の割合と大きさが範囲に入るものの中で
点の数がいちばん多いものを対象にする。1 つも入らなければ対象なしにする（いちばん大きいかたまりに戻らない）。
"""
from __future__ import annotations

import json
import os

import numpy as np

OBJECTS_DIR = os.environ.get("GRASP_OBJECTS_DIR",
                             os.path.join(os.path.dirname(os.path.abspath(__file__)), "config", "objects"))


def names() -> list[str]:
    """登録してある物の名前の一覧。"""
    if not os.path.isdir(OBJECTS_DIR):
        return []
    return sorted(f[:-5] for f in os.listdir(OBJECTS_DIR) if f.endswith(".json"))


def load(name: str) -> dict:
    """登録を読む。無い名前なら一覧を付けて止める。"""
    path = os.path.join(OBJECTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        raise SystemExit(f"[objects] 止める。対象 {name!r} は登録に無い（{OBJECTS_DIR}）。登録してある物: {', '.join(names()) or 'なし'}")
    with open(path) as f:
        spec = json.load(f)
    spec.setdefault("name", name)
    spec.setdefault("min_color_frac", 0.6)
    spec.setdefault("separate_from", [])
    return spec


def color_mask(img_bgr: np.ndarray, spec: dict, dilate: int = 0) -> np.ndarray:
    """登録の色に当たる画素 (H,W) bool。小さな点は開いて消し、dilate>0 なら広げる。"""
    import cv2

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    (h0, h1), (s0, s1), (v0, v1) = spec["hsv"]["h"], spec["hsv"]["s"], spec["hsv"]["v"]
    m = ((hsv[..., 0] >= h0) & (hsv[..., 0] <= h1) & (hsv[..., 1] >= s0) & (hsv[..., 1] <= s1)
         & (hsv[..., 2] >= v0) & (hsv[..., 2] <= v1)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    if dilate > 0:
        m = cv2.dilate(m, np.ones((dilate, dilate), np.uint8))
    return m.astype(bool)


def separate_mask(img_bgr: np.ndarray, spec: dict) -> np.ndarray:
    """separate_from の相手の色の画素（広げてある）。深度からこの画素を消すと、触れている相手と分かれる。"""
    out = np.zeros(img_bgr.shape[:2], bool)
    for other in spec.get("separate_from", []):
        out |= color_mask(img_bgr, load(other), dilate=7)
    return out


def _pixels(pts: np.ndarray, K: np.ndarray, shape) -> tuple[np.ndarray, np.ndarray]:
    """カメラの座標の点 → 画素 (u,v) と、画像の中に入るかの bool。"""
    uv = pts @ K.T
    uv = np.round(uv[:, :2] / uv[:, 2:]).astype(int)
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < shape[0])
    return uv, inside


def describe(cluster: np.ndarray, K: np.ndarray, mask: np.ndarray, spec: dict) -> dict:
    """1 つのかたまりの色の割合・外形と、範囲に入るか。"""
    uv, inside = _pixels(cluster, K, mask.shape)
    frac = float(mask[uv[inside, 1], uv[inside, 0]].mean()) if inside.any() else 0.0
    ext = np.sort(np.ptp(cluster, 0))[::-1]
    lo_max, hi_max = spec.get("extent_max_m", [0.0, 1e9])
    lo_min, hi_min = spec.get("extent_min_m", [0.0, 1e9])
    lo_mid, hi_mid = spec.get("extent_mid_m", [0.0, 1e9])  # 真ん中の辺（2026-11-22: 丸太とテープの輪を、太さで分ける）
    size_ok = bool(lo_max <= ext[0] <= hi_max and lo_min <= ext[-1] <= hi_min and lo_mid <= ext[1] <= hi_mid)
    color_ok = frac >= spec["min_color_frac"]
    return {"pts": int(len(cluster)), "color_frac": round(frac, 3), "extent_m": ext.round(3).tolist(),
            "size_ok": size_ok, "color_ok": bool(color_ok), "ok": bool(size_ok and color_ok)}


def split_tall(cluster: np.ndarray, plane, spec: dict) -> np.ndarray | None:
    """立てた細い物（木の丸太）が、背の低い物（同じ色のテープなど）と触れて 1 つのかたまりになったときに、背の高い所だけを取り出す。
    spec["split_tall"] = {"min_height_m": h, "radius_m": r}: 机から h より高い点の、机に沿った向きの中心から r 以内の点（高さは問わない）。
    2026-11-22 北九州: 立てた丸太が同じ色のマスキングテープの輪と触れて 17 × 15 cm のかたまりになり、大きさで外れた。"""
    st = spec.get("split_tall")
    if not st or plane is None:
        return None
    n, d = np.asarray(plane[:3], float), float(plane[3])
    h = cluster @ n + d
    top = cluster[h > float(st["min_height_m"])]
    if len(top) < 20:
        return None
    # 高い所の点を、机に沿った向きで小さなかたまりに分け、太さが一番細い物（丸太の上の方）を選ぶ。
    # テープの輪が立っていると、その縁も高い所に入り、全部の平均だと丸太とテープの間に中心が寄った（2026-11-22 163516）
    from scipy.cluster.hierarchy import fcluster, linkage
    flat = top - np.outer(top @ n + d, n)
    if len(top) > 3:
        lab = fcluster(linkage(flat[:: max(1, len(flat) // 400)], "single"), t=0.012, criterion="distance")
        sub_idx = np.arange(len(flat))[:: max(1, len(flat) // 400)]
        best = None
        for k in np.unique(lab):
            pts_k = flat[sub_idx[lab == k]]
            if len(pts_k) < 10:
                continue
            width = float(np.max(np.ptp(pts_k, 0)))
            score = abs(width - float(st.get("width_m", 0.05)))
            if best is None or score < best[0]:
                best = (score, pts_k.mean(0))
        if best is None:
            return None
        c = best[1] + n * (float(np.median(top @ n + d)) - float(best[1] @ n + d))  # 高さは高い所の中くらいに
    else:
        c = top.mean(0)
    rel = cluster - c
    horiz = rel - np.outer(rel @ n, n)  # 机に沿った向きの距離
    sub = cluster[np.linalg.norm(horiz, axis=1) < float(st["radius_m"])]
    return sub if len(sub) >= 50 else None


def footprint(pts: np.ndarray, up) -> tuple[float, float]:
    """机に沿った外形 (長い辺, 短い辺) [m]。点を机の面（法線 up）に投影し、主成分の向きで 2〜98 パーセンタイルの幅を測る。
    カメラの座標の外形と違い、寝かせた長い物の短い辺が斜めに見えて大きくならない。"""
    P = np.asarray(pts, float)
    n = np.asarray(up, float); n = n / np.linalg.norm(n)
    Q = P - np.outer(P @ n, n)
    Q = Q - Q.mean(0)
    _, _, Vt = np.linalg.svd(Q, full_matrices=False)
    w = [float(np.percentile(Q @ v, 98) - np.percentile(Q @ v, 2)) for v in Vt[:2]]
    return max(w), min(w)


def describe_shape(cluster: np.ndarray, plane, spec: dict, K: np.ndarray | None = None, shape=None,
                   exclude_masks: list | None = None) -> dict:
    """形で選ぶ（spec の "mode": "shape"）ときの測った値: 机からの高さ・机に沿った外形・当てはまるか。
    K と画像の大きさ shape があれば、画面の端（spec の border_px 以内）に掛かるかたまりを外す（画面の外から続くケーブルなど）。"""
    n = np.asarray(plane[:3], float); d = float(plane[3])
    nn = np.linalg.norm(n); n, d = n / nn, d / nn
    h = float(np.percentile(cluster @ n + d, 98))
    long_, short = footprint(cluster, n)
    why = []
    if len(cluster) < spec.get("min_pts", 150):
        why.append("点が少ない")
    if h < spec.get("min_height_m", 0.015):
        why.append(f"低い（{h * 1000:.0f} mm）")
    if h > spec.get("max_height_m", 0.20):
        why.append(f"高い（{h * 1000:.0f} mm）")
    if "max_short_m" in spec and short > spec["max_short_m"]:
        why.append(f"爪に入らない（短い辺 {short * 1000:.0f} mm）")
    for name, mk in exclude_masks or []:  # 置き場（重量計など）の色のかたまりは拾わない
        uv_i, inside = _pixels(cluster, K, mk.shape)
        fr = float(mk[uv_i[inside, 1], uv_i[inside, 0]].mean()) if inside.any() else 0.0
        if fr >= spec.get("exclude_frac", 0.25):
            why.append(f"置き場（{name}・色 {fr:.2f}）")
    if K is not None and shape is not None and spec.get("border_px", 0) > 0:
        uv = cluster @ np.asarray(K).T
        uv = uv[:, :2] / uv[:, 2:]
        b = float(spec["border_px"])
        if np.any((uv[:, 0] < b) | (uv[:, 1] < b) | (uv[:, 0] > shape[1] - 1 - b) | (uv[:, 1] > shape[0] - 1 - b)):
            why.append("画面の端に掛かる（画面の外から続く物）")
    if long_ > spec.get("max_long_m", 0.35):
        why.append(f"長すぎる（{long_ * 1000:.0f} mm）")
    return {"pts": int(len(cluster)), "height_m": round(h, 3), "long_m": round(long_, 3), "short_m": round(short, 3),
            "ok": not why, "why": "・".join(why)}


def pick(clusters: list[np.ndarray], K: np.ndarray, img_bgr: np.ndarray, spec: dict,
         plane=None) -> tuple[int | None, list[dict]]:
    """かたまりの中から対象を 1 つ選ぶ → (番号か None, かたまりごとの測った値)。範囲に入る中で点のいちばん多いもの。
    plane（segment_clusters の dbg["plane"]）と spec の split_tall があると、色は合うが大きすぎるかたまりから背の高い所を
    取り出したかたまりを clusters の後ろに足して候補にする（clusters をその場で書き換える）。
    spec の "mode" が "shape"（config/objects/any.json）なら色を見ず、机からの高さと机に沿った外形で選ぶ（plane が要る）。"""
    if spec.get("mode") == "shape":
        if plane is None:
            raise SystemExit("[objects] 形で選ぶには机の面（plane）が要る")
        ex = [(n, color_mask(img_bgr, load(n))) for n in spec.get("exclude", [])]
        table = [describe_shape(c, plane, spec, K=K, shape=img_bgr.shape, exclude_masks=ex) for c in clusters]
        ok = [i for i, t in enumerate(table) if t["ok"]]
        return (max(ok, key=lambda i: table[i]["pts"]) if ok else None), table
    mask = color_mask(img_bgr, spec)
    table = [describe(c, K, mask, spec) for c in clusters]
    for i in range(len(table)):
        if table[i]["color_ok"] and not table[i]["size_ok"]:
            sub = split_tall(clusters[i], plane, spec)
            if sub is not None:
                clusters.append(sub)
                t = describe(sub, K, mask, spec)
                t["split_from"] = i
                table.append(t)
    ok = [i for i, t in enumerate(table) if t["ok"]]
    return (max(ok, key=lambda i: table[i]["pts"]) if ok else None), table


def not_found(spec: dict, table: list[dict]) -> str:
    """見つからなかったときの 1 行（かたまりごとの色の割合と外形）。"""
    if spec.get("mode") == "shape":
        brief = "; ".join(f"#{i} {t['pts']}pts 高さ {t['height_m'] * 1000:.0f} mm 外形 {t['long_m'] * 1000:.0f}×{t['short_m'] * 1000:.0f} mm"
                          f"（{t['why']}）" for i, t in enumerate(table[:5]))
        return f"target {spec['name']} not found ({len(table)} clusters: {brief or 'none'})"
    brief = "; ".join(f"#{i} {t['pts']}pts color {t['color_frac']:.2f} ext {t['extent_m']}" for i, t in enumerate(table[:5]))
    return f"target {spec['name']} not found ({len(table)} clusters: {brief or 'none'})"
