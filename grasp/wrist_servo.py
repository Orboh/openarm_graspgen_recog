"""手首の D405 で、爪から見た物の位置を測り、掴む直前に直す量を出す（計算だけ・腕もカメラも触らない）。

2026-11-23・作業者: Munenori。北九州の左のグリッパーに D405 を付けた。胸のカメラのハンドアイと腕の運動学を通すと、
掴む所で 1〜3 cm ずれる（2026-11-22 の右腕の試行・11-23 の point_test で指先が 20〜30 mm ずれた）。爪とカメラは一緒に
固定されているので、爪の模型（grasp/tools/wrist_jaw_measure.py が書く）があれば、爪から見た物の位置を直接測れる。

東京の左手首の試し（grasp/tools/run_grasp_servo.py・2026-11-20）から受け継ぐこと:
  - 掴む姿勢の少し手前（近づく向きに 4 cm 戻した所）で測る（掴む所だと物が画像の端で切れる）
  - 直すのは 0.8 倍に弱める（一度に全部直すと行き過ぎた）。1 回で動かすのは max_step まで
  - 下ろすときも 2 cm ごとに測り直す（6 cm 下ろす間に爪が横へ 35 mm 流れた）
変えたこと:
  - 爪を画素の帯で除くのをやめ、爪の模型の幾何（閉じる向きに爪の内側より外・爪の先より手前）で除く
  - 狙いの塊は、計画から出した「ここにあるはず」に一番近い塊（無ければ爪の軸に一番近い塊）。テープの輪・箱と取り違えない
  - 近づく向きの深さは grasp/depth.py と同じく「一番太い所」で測る（その位置が爪の先から depth_target 奥に来るまで下ろす）

座標: 爪の座標は 原点 = 爪の先の中点、+X = 閉じる向き、+Y = 厚み、+Z = 近づく向き（GraspGenX・plan.py の向きと同じ）。
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class JawModel:
    """手首のカメラの座標での爪（wrist_jaw_measure.py の JSON）。"""
    mid: np.ndarray       # 爪の先の中点 [m]
    C: np.ndarray         # 閉じる向き（Xg）
    T: np.ndarray         # 厚み（Yg）
    A: np.ndarray         # 近づく向き（Zg）
    K: np.ndarray         # 内部パラメータ 3x3
    half_open: float      # 開いた爪の内側の間の半分 [m]
    tips: tuple           # (左の先, 右の先) [m]
    tips_px: tuple = None     # (左の先, 右の先) の画素
    edges_px: tuple = None    # (左の縁の 2 画素, 右の縁の 2 画素)（wrist_jaw_measure.py の edge_px）

    @classmethod
    def load(cls, path: str) -> "JawModel":
        j = json.load(open(path))
        for k in ("approach_cam", "closing_cam", "thickness_cam"):
            if k not in j:
                raise SystemExit(f"[wrist] {path} に {k} が無い（wrist_jaw_measure.py で指の縁まで 6 点クリックして作り直す）")
        i = j["intrinsics"]
        fx, fy = i["fx"], i["fy"]
        ppx, ppy = i["ppx"], i["ppy"]
        gap = float(j.get("tip_gap_m", np.linalg.norm(np.subtract(j["right_tip_cam_m"], j["left_tip_cam_m"]))))
        return cls(mid=np.array(j["mid_tips_cam_m"]), C=np.array(j["closing_cam"]), T=np.array(j["thickness_cam"]),
                   A=np.array(j["approach_cam"]), K=np.array([[fx, 0, ppx], [0, fy, ppy], [0, 0, 1.0]]),
                   half_open=gap / 2, tips=(np.array(j["left_tip_cam_m"]), np.array(j["right_tip_cam_m"])),
                   tips_px=(tuple(j["left_tip_px"]), tuple(j["right_tip_px"])) if "left_tip_px" in j else None,
                   edges_px=((tuple(j["edge_px"][0]), tuple(j["edge_px"][1])), (tuple(j["edge_px"][2]), tuple(j["edge_px"][3])))
                   if "edge_px" in j else None)

    @property
    def R(self) -> np.ndarray:
        """爪の座標 → カメラの座標の回転（列が C・T・A）。"""
        return np.column_stack([self.C, self.T, self.A])

    def to_jaw(self, P: np.ndarray) -> np.ndarray:
        """カメラの座標の点 (...,3) → 爪の座標（原点 = 爪の先の中点）。"""
        return (np.asarray(P) - self.mid) @ self.R


def cloud(depth_m: np.ndarray, K: np.ndarray) -> np.ndarray:
    """深度 (H,W) [m] → カメラの座標の点 (H,W,3)。"""
    H, W = depth_m.shape
    v, u = np.mgrid[0:H, 0:W]
    return np.dstack([(u - K[0, 2]) / K[0, 0] * depth_m, (v - K[1, 2]) / K[1, 1] * depth_m, depth_m])


def floor_plane(P: np.ndarray, valid: np.ndarray, seed: int = 0, iters: int = 300, tol: float = 0.006):
    """机の面（遠い側 60% の点に RANSAC）。返り値 (n, d): 机からの高さ = P·n + d、n はカメラの側を向く。"""
    dz = P[..., 2]
    far = P[valid & (dz >= np.percentile(dz[valid], 40))][::7]  # >=: 深さが一定の平らな机でも点が残るように
    if len(far) < 3:
        raise ValueError("机の面を当てはめる点が足りない")
    rng = np.random.default_rng(seed)
    best = (0, None)
    for _ in range(iters):
        s = far[rng.choice(len(far), 3, replace=False)]
        n = np.cross(s[1] - s[0], s[2] - s[0])
        nn = np.linalg.norm(n)
        if nn < 1e-9:
            continue
        n /= nn
        d = -n @ s[0]
        k = int(np.sum(np.abs(far @ n + d) < tol))
        if k > best[0]:
            best = (k, (n, d))
    if best[1] is None:
        raise ValueError("机の面が当てはまらない")
    n, d = best[1]
    if d < 0:  # カメラ（原点）の高さ d が正になる向きに
        n, d = -n, -d
    return n, float(d)


def jaw_pixels(shape, jaw: JawModel, margin_px: float = 10.0) -> np.ndarray:
    """画像の上で爪に当たる画素の印: 左右の指の内側の縁の線より外側で、爪の先の行より下（爪とカメラは一緒に固定なので画素も固定）。
    2026-11-23 北九州の 1 回目: 指の内側の黒いゴムは深さが正しく取れず、幾何の除き方（jaw_points）をすり抜けて物と取り違えた。"""
    H, W = shape
    m = np.zeros(shape, bool)
    if jaw.edges_px is None or jaw.tips_px is None:
        return m
    v, u = np.mgrid[0:H, 0:W]
    cx = (jaw.tips_px[0][0] + jaw.tips_px[1][0]) / 2
    for (p0, p1), tip in zip(jaw.edges_px, jaw.tips_px):
        (u0, v0), (u1, v1) = p0, p1
        nx, ny = v1 - v0, -(u1 - u0)  # 縁の線の法線
        nn = np.hypot(nx, ny)
        nx, ny = nx / nn, ny / nn
        if nx * (cx - u0) + ny * (tip[1] + 50 - v0) > 0:  # 法線を爪の外側（真ん中と逆）に向ける
            nx, ny = -nx, -ny
        outside = (nx * (u - u0) + ny * (v - v0)) > -margin_px
        m |= outside & (v >= tip[1] - margin_px)
    return m


def jaw_points(Q: np.ndarray, jaw: JawModel, margin: float = 0.004) -> np.ndarray:
    """爪そのものに当たる点の印（爪の座標 Q (...,3)）: 閉じる向きに爪の内側より外で、爪の先より手前。"""
    return (np.abs(Q[..., 0]) >= jaw.half_open - margin) & (Q[..., 2] <= margin)


def depth_edges(depth_m: np.ndarray, jump: float = 0.008) -> np.ndarray:
    """隣の画素と深さが jump [m] より違う所の印（接して写る物の境目。東京の記録で木片とアルミの棒が 1 つの塊になった）。"""
    d = depth_m
    e = np.zeros(d.shape, bool)
    dv = np.abs(np.diff(d, axis=0)) > jump
    dh = np.abs(np.diff(d, axis=1)) > jump
    e[1:] |= dv
    e[:-1] |= dv
    e[:, 1:] |= dh
    e[:, :-1] |= dh
    return e


def find_target(depth_m: np.ndarray, jaw: JawModel, hint_jaw: np.ndarray | None = None,
                min_px: int = 600, min_height: float = 0.012, zmax: float = 0.6, jump: float = 0.008,
                keep_radius: float | None = None, max_hint_dist: float | None = None) -> dict | None:
    """手首の深度から狙いの物の塊を切り出す。hint_jaw: 計画から出した物の中心（爪の座標 [m]）。
    keep_radius: hint_jaw があるとき、選んだ塊のうち hint から横（閉じる・厚み）にこの距離 [m] より遠い点を捨てる
    （接して写る別の物と同じ深さで 1 つの塊になったとき。東京の記録で木片とアルミの棒が分かれなかった）。半径は
    腕の誤差（2〜3 cm）と物の大きさの半分を足した値にする。
    max_hint_dist: hint_jaw からこれより遠い塊しか無ければ None（2026-11-23: 木材が指の陰で見えず、111 mm 離れた指のゴムを選んだ）。

    返り値: {"Q": 物の点（爪の座標）, "mask": 画素の印, "bbox": (x, y, w, h), "cut": 画像の端で切れている辺,
            "floor": (n, d), "px": 画素数, "pick": 選んだ理由} か None。"""
    import cv2
    P = cloud(depth_m, jaw.K)
    valid = (depth_m > 0.04) & (depth_m < zmax)
    if valid.sum() < 2000:
        return None
    n, d = floor_plane(P, valid)
    h = P @ n + d
    Q = jaw.to_jaw(P)
    above = valid & (h > min_height) & ~jaw_points(Q, jaw) & ~jaw_pixels(depth_m.shape, jaw)
    num, lab, st, _ = cv2.connectedComponentsWithStats((above & ~depth_edges(depth_m, jump)).astype(np.uint8), 4)
    cands = []
    for k in range(1, num):
        if st[k, 4] < min_px:
            continue
        m = lab == k
        c = np.median(Q[m], axis=0)
        if hint_jaw is not None:
            score = float(np.linalg.norm(c - hint_jaw))
        else:  # 爪の軸（閉じる・厚みが 0、近づく向きの前）からの横の距離
            score = float(np.hypot(c[0], c[1]))
        cands.append((score, k, c))
    if not cands:
        return None
    cands.sort(key=lambda t: t[0])
    if hint_jaw is not None and max_hint_dist is not None and cands[0][0] > max_hint_dist:
        return None
    score, k, c = cands[0]
    m = lab == k
    if hint_jaw is not None and keep_radius is not None:
        far = np.hypot(Q[..., 0] - hint_jaw[0], Q[..., 1] - hint_jaw[1]) > keep_radius
        m &= ~far
        if m.sum() < min_px // 2:
            return None
    ys, xs = np.nonzero(m)
    x, y, w, hh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    H, W = depth_m.shape
    cut = {"left": x <= 1, "top": y <= 1, "right": x + w >= W - 1, "bottom": y + hh >= H - 1}
    return {"Q": Q[m], "mask": m, "bbox": (x, y, w, hh), "cut": cut, "floor": (n, d), "px": int(m.sum()),
            "pick": ("計画の位置に一番近い塊" if hint_jaw is not None else "爪の軸に一番近い塊") + f"（{score * 1000:.0f} mm・候補 {len(cands)}）",
            "height_m": float(np.percentile(h[m], 95))}


def widest_depth(Q: np.ndarray, bin_m: float = 0.005, width_tol: float = 0.003, min_pts: int = 5) -> float | None:
    """物の一番太い所（閉じる向きの幅が最大の所）の近づく向きの位置 [m]（爪の座標・+ は爪の先より前）。grasp/depth.py と同じ考え。"""
    a = Q[:, 2]
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    widths, centres = [], []
    for z0 in np.arange(lo, hi, bin_m):
        b = Q[(a >= z0) & (a < z0 + bin_m), 0]
        if len(b) >= min_pts:
            widths.append(np.percentile(b, 98) - np.percentile(b, 2))
            centres.append(z0 + bin_m / 2)
    if not widths:
        return None
    w, c = np.array(widths), np.array(centres)
    return float(c[w >= w.max() - width_tol].mean())


def offsets(tgt: dict, jaw: JawModel) -> dict:
    """爪の先の中点から見た物の位置（爪の座標）。

    e_c: 閉じる向きのずれ（両側が写っていれば 2〜98 パーセンタイルの真ん中、片側が切れていれば中央値で「当てにならない」印）
    e_t: 厚みのずれ（同じ）、a_wide: 一番太い所の近づく向きの位置（+ は爪の先より前）、tip_h: 爪の先の机からの高さ、
    drop: 近づく向きに 1 m 進むと爪の先が下がる量。"""
    Q = tgt["Q"]
    cut = tgt["cut"]
    sides_c = not (cut["left"] or cut["right"])
    e_c = float((np.percentile(Q[:, 0], 2) + np.percentile(Q[:, 0], 98)) / 2) if sides_c else float(np.median(Q[:, 0]))
    sides_t = not (cut["top"] or cut["bottom"])
    e_t = float((np.percentile(Q[:, 1], 2) + np.percentile(Q[:, 1], 98)) / 2) if sides_t else float(np.median(Q[:, 1]))
    n, d = tgt["floor"]
    tip_h = min(float(p @ n + d) for p in jaw.tips)
    # 切れている辺の向き → 爪の座標で寄せる向き（その辺の外へ物が続いている）。画像の右 = +u、下 = +v の、爪の座標の向き
    du = np.array([jaw.C[0], jaw.T[0]])  # 画像の右へ進むと増える爪の座標（閉じる・厚み）の割合（カメラの x 成分）
    dv = np.array([jaw.C[1], jaw.T[1]])  # 画像の下へ進むと増える割合（カメラの y 成分）
    want = np.zeros(2)
    want += du * (float(cut["right"]) - float(cut["left"]))
    want += dv * (float(cut["bottom"]) - float(cut["top"]))
    c_dir = float(np.sign(want[0])) if (not sides_c and abs(want[0]) > 0.3) else 0.0
    t_dir = float(np.sign(want[1])) if (not sides_t and abs(want[1]) > 0.3) else 0.0
    return {"e_c": e_c, "e_t": e_t, "c_ok": sides_c, "t_ok": sides_t, "c_dir": c_dir, "t_dir": t_dir, "a_wide": widest_depth(Q),
            "tip_h": tip_h, "drop": float(-(jaw.A @ n)), "px": tgt["px"]}


def t_covered(t_vals, bias_t: float = 0.0, half: float = 0.0177, margin: float = 0.003) -> bool:
    """爪の幅（厚みの向き ±half）が全部、物の上に乗っているか（2026-11-23・長い物）。t_vals は手首のカメラで見た物の点の厚みの向きの
    位置（爪の座標）。本当の爪の真ん中はカメラの模型で bias_t の所（--servo-t-shift の目標）。画面の端で切れていても、見えている所が
    両側に届いていれば乗っている（本当の物は見えている所より長い）。乗っていれば厚みの向きは直さなくてよい。"""
    t = np.asarray(t_vals, float)
    if len(t) < 20:
        return False
    return bool(np.percentile(t, 2) <= bias_t - half - margin and np.percentile(t, 98) >= bias_t + half + margin)


def lateral_step(off: dict, gain: float = 0.8, tol: float = 0.005, max_step: float = 0.03,
                 t_tol: float | None = None, peek: float = 0.015) -> tuple[np.ndarray, bool]:
    """横（閉じる・厚み）に直す量（爪の座標 [m]・(3,)）と、もう直さなくてよいか。

    片側が画像の端で切れている向きは、切れた見え方の中心で直さない（東京で、切れた見え方からの厚みの推定が 4〜7 cm 外れた）。
    代わりに、切れている側へ peek だけ寄せて物の全体を写しにいく（off["c_dir"]・off["t_dir"] が寄せる向き）。
    2026-11-23 北九州の 1 回目: 厚みの向きに 25〜29 mm ずれて上の端で切れたまま、直さずに閉じて外した。
    t_tol: 厚みの向きの許し（長い物なら端から 4 cm 内なら良い・既定 tol と同じ）。"""
    t_tol = tol if t_tol is None else t_tol
    ec = off["e_c"] if (off["c_ok"] and abs(off["e_c"]) > tol) else 0.0
    et = off["e_t"] if (off["t_ok"] and abs(off["e_t"]) > t_tol) else 0.0
    pc = 0.0 if off["c_ok"] else peek * off.get("c_dir", 0.0) / gain
    pt = 0.0 if off["t_ok"] else peek * off.get("t_dir", 0.0) / gain
    if ec == 0.0 and et == 0.0 and pc == 0.0 and pt == 0.0:
        return np.zeros(3), True
    step = gain * np.array([ec + pc, et + pt, 0.0])
    s = np.linalg.norm(step)
    if s > max_step:
        step *= max_step / s
    return step, False


def descend_travel(off: dict, depth_target: float = 0.030, floor_clear: float = 0.008,
                   corner_half: float = 0.019) -> tuple[float, str]:
    """近づく向きに下ろす量 [m] と理由。一番太い所が爪の先から depth_target 奥に来るまで。ただし爪の先の角が机から
    floor_clear より下がらない（傾けた爪の角は中心より corner_half·sin だけ低い・東京で角が机に 1 mm めり込んだ）。"""
    drop = max(off["drop"], 1e-3)
    corner = corner_half * float(np.sqrt(max(0.0, 1.0 - off["drop"] ** 2)))
    by_floor = (off["tip_h"] - corner - floor_clear) / drop
    if off["a_wide"] is None:
        return max(0.0, by_floor), "一番太い所が出ない → 机まで"
    want = off["a_wide"] + depth_target
    if want <= by_floor:
        return max(0.0, want), f"一番太い所を爪の先から {depth_target * 1000:.0f} mm 奥へ"
    return max(0.0, by_floor), f"机で止まる（一番太い所は爪の先から {max(0.0, by_floor - off['a_wide']) * 1000:.0f} mm 奥まで）"
