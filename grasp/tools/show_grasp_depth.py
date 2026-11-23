#!/usr/bin/env python3
"""選んだ掴み方で、物が爪のどこに入るかを図にする（実機・サーバーなし・記録だけ読む）。

2026-11-23・作業者: Munenori。2026-11-22 の右腕の木材で、爪の先端で浅く挟んで抜けた回があり、選んだ掴み方の
挟む深さ（grasp/depth.py）を目で確かめるために作った。1 回につき 3 枚:
  左: 胸のカメラの画像に、物の点（赤）と開いた爪（緑の枠）と指先の中心（黄の点）を重ねる
  中: 爪の横から（閉じる向き × 近づく向き）。灰色の帯が爪の面のうち物に当たる 80 mm、青の線が挟む深さ
  右: 爪の前から（厚み × 近づく向き）

    PYTHONPATH=. python3 grasp/tools/show_grasp_depth.py grasp/runs/cap_resident_20261122_181155 ... \\
        --label 181155=抜けた -o grasp/runs/grasp_depth_20261122.png
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from grasp.depth import FACE_LEN, FINGER_HALF_THICK, grasp_depth, in_jaw  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIN_LEN = 0.110     # フィンレイ指の全長 [m]（指先から付け根まで）
FIN_THICK = 0.012   # 図で描く指の太さ（閉じる向き）[m]・目安


def load_run(d: str, T_bc: np.ndarray) -> dict:
    """記録のフォルダから、計画（plan.json）・物の点（腕の付け根の座標）・画像・内部パラメータを読む。"""
    p = json.load(open(os.path.join(d, "plan.json")))
    obj_cam = np.load(os.path.join(d, "object.npy"))
    ob = obj_cam @ T_bc[:3, :3].T + T_bc[:3, 3]
    intr = json.load(open(os.path.join(d, "intrinsics.json")))
    K = np.array(intr["K"]) if "K" in intr else np.array([[intr["fx"], 0, intr["ppx"]], [0, intr["fy"], intr["ppy"]], [0, 0, 1]])
    return dict(plan=p, ob=ob, obj_cam=obj_cam, K=K, color=os.path.join(d, "color.png"))


def jaw_corners(G: np.ndarray, half: float, tip: float) -> list[np.ndarray]:
    """開いた左右の指の箱の 8 頂点（腕の付け根の座標）を、指ごとに返す。"""
    out = []
    for s in (-1, 1):
        xs = (s * half, s * (half + FIN_THICK))
        cs = [G[:3, 3] + G[:3, :3] @ np.array([x, y, z])
              for x in xs for y in (-FINGER_HALF_THICK, FINGER_HALF_THICK) for z in (tip - FIN_LEN, tip)]
        out.append(np.array(cs))
    return out


def project(P_base: np.ndarray, T_bc: np.ndarray, K: np.ndarray) -> np.ndarray:
    """腕の付け根の座標の点を胸のカメラの画素へ。"""
    T_cb = np.linalg.inv(T_bc)
    Pc = P_base @ T_cb[:3, :3].T + T_cb[:3, 3]
    uv = Pc @ K.T
    return uv[:, :2] / uv[:, 2:3]


def draw(ax_img, ax_xz, ax_yz, r: dict, T_bc: np.ndarray, half: float, tip: float, title: str) -> float:
    """1 回分の 3 枚を描き、挟む深さ [m] を返す。"""
    import cv2
    from matplotlib.patches import Rectangle
    G = np.array(r["plan"]["T_grasp_base"])
    depth = grasp_depth(G, r["ob"], half, tip)
    q, m = in_jaw(G, r["ob"], half, tip)

    # 左: 画像に重ねる
    img = cv2.cvtColor(cv2.imread(r["color"]), cv2.COLOR_BGR2RGB)
    uv_o = project(r["ob"], T_bc, r["K"])
    ax_img.imshow(img)
    ax_img.scatter(uv_o[::4, 0], uv_o[::4, 1], s=1, c="red", alpha=0.5)
    edges = [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)]
    for C in jaw_corners(G, half, tip):
        uv = project(C, T_bc, r["K"])
        for i, j in edges:
            ax_img.plot(uv[[i, j], 0], uv[[i, j], 1], c="lime", lw=1)
    tcp = G[:3, 3] + G[:3, 2] * (tip - 0.015)
    ax_img.scatter(*project(tcp[None], T_bc, r["K"])[0], s=30, c="yellow", edgecolors="k", zorder=5)
    c = uv_o.mean(0)
    ax_img.set_xlim(c[0] - 110, c[0] + 110); ax_img.set_ylim(c[1] + 90, c[1] - 90)
    ax_img.set_title(title, fontsize=9); ax_img.axis("off")

    # 中・右: 爪の座標（mm・指先が 0、手首の側が負）
    zz = (q[:, 2] - tip) * 1000
    near = zz > -FIN_LEN * 1000 - 40
    for ax, k, lab, w in ((ax_xz, 0, "閉じる向き [mm]", half * 1000), (ax_yz, 1, "爪の厚み [mm]", FINGER_HALF_THICK * 1000)):
        ax.scatter(q[near & ~m, k] * 1000, zz[near & ~m], s=1, c="lightcoral", alpha=0.4)
        ax.scatter(q[m, k] * 1000, zz[m], s=1, c="red", alpha=0.7)
        ax.axhspan(-FACE_LEN * 1000, 0, color="0.85", zorder=0)
        if k == 0:  # 指を描く
            for s in (-1, 1):
                x0 = s * half * 1000 if s > 0 else -(half + FIN_THICK) * 1000
                ax.add_patch(Rectangle((x0, -FIN_LEN * 1000), FIN_THICK * 1000, FIN_LEN * 1000, color="green", alpha=0.5))
        else:
            ax.axvspan(-w, w, color="green", alpha=0.12, zorder=0)
        ax.axhline(0, c="k", lw=0.8)
        ax.axhline(-depth * 1000, c="tab:blue", lw=1.5)
        ax.text(0.02, 0.03, f"挟む深さ {depth*1000:.0f} mm", transform=ax.transAxes, color="tab:blue", fontsize=9)
        ax.set_xlim(-70, 70); ax.set_ylim(-FIN_LEN * 1000 - 40, 60)
        ax.set_aspect("equal"); ax.set_xlabel(lab, fontsize=8); ax.tick_params(labelsize=7)
    ax_xz.set_ylabel("近づく向き [mm]（0 = 指先）", fontsize=8)
    return depth


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="記録のフォルダ（plan.json・object.npy・color.png がある）")
    ap.add_argument("--label", action="append", default=[], help="<フォルダ名の末尾>=<結果>（図の題に足す）。何度でも")
    ap.add_argument("--t-base-cam", default=os.environ.get("GRASP_T_BASE_CAM",
                    os.path.join(HERE, "config", "sites", "kitakyushu-collect.T_base_cam.json")))
    ap.add_argument("--open-width", type=float, default=0.080, help="開いた爪の内側の間 [m]（北九州の右は 0.080）")
    ap.add_argument("--tip", type=float, default=0.14, help="手首の先の面から指先 [m]")
    ap.add_argument("-o", "--out", required=True, help="書き出す PNG")
    a = ap.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for f in ("Noto Sans CJK JP", "IPAexGothic", "TakaoGothic"):
        if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.family"] = f; break
    labels = dict(s.split("=", 1) for s in a.label)
    T_bc = np.array(json.load(open(a.t_base_cam))["T_base_cam"])
    runs = [d.rstrip("/") for d in a.runs if os.path.exists(os.path.join(d, "plan.json"))]
    fig, axs = plt.subplots(len(runs), 3, figsize=(12, 3.6 * len(runs)), squeeze=False,
                            gridspec_kw=dict(width_ratios=[1.35, 1, 1]))
    for row, d in zip(axs, runs):
        r = load_run(d, T_bc)
        tag = os.path.basename(d).split("_")[-1]
        lab = next((v for k, v in labels.items() if tag.endswith(k)), "")
        p = r["plan"]
        title = f"{tag[:2]}:{tag[2:4]}  {lab}\n傾き {p['tilt_deg']:.0f}°  点数 {p['score']:.2f}  {p.get('arm', 'left')}"
        dep = draw(*row, r, T_bc, a.open_width / 2, a.tip, title)
        print(f"{tag} {lab:12s} 傾き {p['tilt_deg']:3.0f}°  挟む深さ {dep*1000:3.0f} mm")
    plt.tight_layout()
    plt.savefig(a.out, dpi=80)
    print(f"[show] {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
