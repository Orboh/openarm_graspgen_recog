"""掴み方の「挟む深さ」を測る（numpy だけ・plan.py と grasp/tools/show_grasp_depth.py が使う）。

2026-11-23・作業者: Munenori。2026-11-22 の右腕の木材で、GraspGenX が爪の先端で浅く挟む掴み方を選び、
持ち上げで抜けた。GraspGenX に渡す爪の箱（指先から 80 mm）の中に物が少しでも入れば正しい候補になるので、
点数の順に選ぶだけでは浅い掴み方が混ざる。ここで深さを測り、plan.py --min-depth で浅い候補を外す。

挟む深さ = 指先から、閉じた爪が物に当たる所までの距離（近づく向き）。爪は物の一番太い所（閉じる向きの幅が最大の所）で
当たるので、爪の面の長さの中を近づく向きに 5 mm ずつ切り、幅が最大に近い区間の真ん中を当たる所とする。
  - 爪の間に横たわる円柱 → 円柱の中心の深さ（太い所が爪の中にあれば）。中心が指先より先なら、当たる所は指先近く → 浅い
  - 縦に長い物を上から → 幅が一定なので、爪の面に入っている範囲の真ん中
胸のカメラは物の片側しか見ないが、一番太い所（輪郭）は見えることが多い。手首寄りの半分しか見えない円柱では
4 mm ほど深く出る。2026-11-22 の 13 回で、抜けた 2 回は 10・17 mm、成功は 17〜37 mm、爪が当たった 5 回は 47〜67 mm
（当たったのは姿勢の癖と横のずれのため。深いほど横の精度は要る）。

爪の座標（GraspGenX と同じ）: 原点 = 手首の先の面、+Z = 近づく向き、+X = 閉じる向き、+Y = 爪の厚み。指先は z = tip。
"""
from __future__ import annotations

import numpy as np

FINGER_HALF_THICK = 0.0177  # 爪の厚みの半分 [m]（so101 のフィンレイ指 35.3 mm）
FACE_LEN = 0.080            # 爪の面のうち物に当たる長さ [m]（指先から付け根の手前まで・sweep の extents の z）


def in_jaw(G: np.ndarray, pts: np.ndarray, half_open: float, tip: float) -> tuple[np.ndarray, np.ndarray]:
    """物の点を爪の座標へ移し、開いた爪の間（厚みの帯・閉じる向きは内側の面の内・指先から面の長さまで）に入る点を返す。

    返り値: (爪の座標の全点 (N,3), 爪の間に入る点の印 (N,))。"""
    q = (np.asarray(pts, float) - G[:3, 3]) @ G[:3, :3]
    m = ((np.abs(q[:, 1]) <= FINGER_HALF_THICK) & (np.abs(q[:, 0]) <= half_open)
         & (q[:, 2] <= tip) & (q[:, 2] >= tip - FACE_LEN))
    return q, m


def grasp_depth(G: np.ndarray, pts: np.ndarray, half_open: float, tip: float,
                bin_m: float = 0.005, width_tol: float = 0.003, min_points: int = 20) -> float:
    """挟む深さ [m]（指先から、閉じた爪が物に当たる所まで）。爪の間に点が min_points 個より少なければ 0（挟めない）。

    bin_m: 近づく向きに切る幅。width_tol: 最大の幅からこれ以内の区間を「当たる所」に入れる。"""
    q, m = in_jaw(G, pts, half_open, tip)
    q = q[m]
    if len(q) < min_points:
        return 0.0
    widths, centres = [], []
    for z0 in np.arange(tip - FACE_LEN, tip - 1e-9, bin_m):
        b = q[(q[:, 2] >= z0) & (q[:, 2] < z0 + bin_m), 0]
        if len(b) >= 5:  # 点の少ない区間は幅が出ない
            widths.append(np.percentile(b, 98) - np.percentile(b, 2))
            centres.append(z0 + bin_m / 2)
    if not widths:
        return 0.0
    w, c = np.array(widths), np.array(centres)
    return float(tip - c[w >= w.max() - width_tol].mean())


def jaw_clearance(q: np.ndarray, half: float, thick: float = FINGER_HALF_THICK + 0.005, face: float = FACE_LEN,
                  tip_off: float = 0.015, finger: float = 0.022, margin: float = 0.005) -> float:
    """開いた爪の内側と物のすき間 [m]（小さい方）。q は指先の中心（tip − tip_off）から見た物の点（爪の座標: 閉じる・厚み・近づく）。
    爪の厚みの帯（±thick）・面の長さ（指先から face）の中で、爪の板の外側（内側の面 + finger + margin）より遠い点は数えない
    （2026-11-23: テープの輪の縁を挟むとき、輪の向こう側の壁まで数えて全部の候補を落としていた）。点が無ければ inf。"""
    q = np.asarray(q, float)
    m = ((np.abs(q[:, 1]) <= thick) & (q[:, 2] >= tip_off - face) & (q[:, 2] <= tip_off)
         & (np.abs(q[:, 0]) <= half + finger + margin))
    if not m.any():
        return float("inf")
    return float(min(half + q[m, 0].min(), half - q[m, 0].max()))


def grasp_depths(Gs: np.ndarray, pts: np.ndarray, half_open: float, tip: float) -> np.ndarray:
    """grasp_depth を候補ごとに（(N,4,4) → (N,)）。"""
    return np.array([grasp_depth(G, pts, half_open, tip) for G in Gs])
