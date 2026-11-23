#!/usr/bin/env python3
"""手首の D405 から見た爪の先の位置を測り、爪の模型（wrist_jaw_model.json の爪の先の部分）を書く。**腕は動かさない。**

2026-11-23・作業者: Munenori。北九州の左のグリッパーに D405 を付けたときに作った。爪とカメラが一緒に固定されていれば、ここで測る
爪の位置はずっと同じなので、掴む直前に手首のカメラで「爪から見た物の位置」を直接測れる（胸のカメラのハンドアイと腕の運動学を通さない）。
東京の左手首の模型（grasp/config/wrist_jaw_model.json・2026-11-20）は手で測った。ここでは同じ値を画像のクリックで出す。

段:
  1. 爪を掴むときの開く指令の幅に開いておく（左は GRASP_GRIPPER_OPEN=-62。脱力で手で開くなら read_arm.py で -62 前後を見る）。
     爪の先の後ろに物を置かない（机の面か遠い壁が写るように）
  2. D405 の色と深度を撮る（--cap で前に撮ったフォルダも使える）
  3. 窓で 6 点をクリック: ①画像の左の爪の内側の先 ②右の爪の内側の先、③④左の指の内側の面の長い縁（近づく向きに伸びる縁）の上と下、
     ⑤⑥右の指の同じ縁の上と下（r でやり直し・Enter で決める・q で止める）。先の 2 点だけで Enter なら向きは出さない
  4. 各点の深さは 9×9 画素のうち近い側（20 パーセンタイル）を使う（爪の縁の後ろの遠い物を拾わない）
  5. 閉じる向き = 左の先 → 右の先（カメラの座標）
  6. 近づく向き = 左右の指の長い縁の消失点の向き（深さを使わない）。3 次元で平行な 2 本の縁は、それぞれカメラの原点と
     縁の 2 画素を通る面に乗るので、2 枚の面の交わる向きが近づく向き。厚み = 近づく × 閉じる（閉じるは近づくと直交させる）。
     確かめ: 爪の先から出した閉じる向きとの角が 90° に近いこと（2026-11-23 北九州の左で 89.9°・縁を ±3 画素ずらして 0.8°）
     深さで出すのはやめた: D405 は 7 cm より近いと深さが出ず、付け根の側は後ろの床を拾い（東京の記録で 39° 狂った）、
     北九州の指の内側の黒いゴムは深さが抜ける。腕を ±2 cm 動かしてマーカーで出すのも、関節の誤差 2〜3° で指令どおりに動かず使えなかった
  東京の左手首の記録（cap_20261120_191052）で、爪の先は手で測った模型と 1 mm 以内で合った。

    # 窓を出すので、窓の出る OpenCV の python で（lerobot の venv の OpenCV は headless・pyrealsense2 だけ借りる）
    PYTHONPATH=~/.cache/openarm-grasp/rs_only ~/miniconda3/bin/python grasp/tools/wrist_jaw_measure.py --serial 260522271835 \\
        -o grasp/config/sites/kitakyushu-collect.wrist_jaw_left.json
    PYTHONPATH=~/.cache/openarm-grasp/rs_only ~/miniconda3/bin/python grasp/tools/wrist_jaw_measure.py --cap grasp/runs/wristjaw_<時刻> -o ...   # 撮り直さない
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
from grasp.capture import capture, load_capture, save_capture  # noqa: E402

HERE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
NAMES = ["左の爪の先", "右の爪の先", "左の指の縁の上", "左の指の縁の下", "右の指の縁の上", "右の指の縁の下"]


def point_at(depth_m: np.ndarray, K: np.ndarray, u: int, v: int, r: int = 4, pct: float = 20.0) -> np.ndarray:
    """画素 (u, v) のまわり (2r+1)² の有効な深さのうち近い側の値で、カメラの座標の 3 次元の点を返す [m]。"""
    h, w = depth_m.shape
    u, v = int(round(u)), int(round(v))
    win = depth_m[max(0, v - r):min(h, v + r + 1), max(0, u - r):min(w, u + r + 1)]
    z = win[(win > 0.03) & (win < 1.0)]
    if len(z) < 5:
        raise ValueError(f"({u}, {v}) のまわりに深さが無い（爪の縁から少し内側をクリックする）")
    z = float(np.percentile(z, pct))
    return np.array([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z])


def tips_model(lt: np.ndarray, rt: np.ndarray) -> dict:
    """爪の先 2 点から、先の位置・中点・間の距離・閉じる向きを出す（カメラの座標・m）。"""
    cl = (rt - lt) / np.linalg.norm(rt - lt)
    return dict(left_tip_cam_m=lt.tolist(), right_tip_cam_m=rt.tolist(), mid_tips_cam_m=((lt + rt) / 2).tolist(),
                tip_gap_m=float(np.linalg.norm(rt - lt)), closing_cam_from_tips=cl.tolist())


def _ray(K: np.ndarray, u: float, v: float) -> np.ndarray:
    return np.array([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], 1.0])


def approach_from_edges(K: np.ndarray, left_edge, right_edge) -> np.ndarray:
    """左右の指の長い縁（それぞれ 2 画素）の消失点の向き = 近づく向き（カメラの座標・カメラから遠ざかる向きに揃える）。"""
    nL = np.cross(_ray(K, *left_edge[0]), _ray(K, *left_edge[1]))
    nR = np.cross(_ray(K, *right_edge[0]), _ray(K, *right_edge[1]))
    a = np.cross(nL, nR)
    a /= np.linalg.norm(a)
    return a if a[2] > 0 else -a


def axes_model(closing_tips: np.ndarray, approach: np.ndarray) -> dict:
    """近づく向きと、爪の先から出した閉じる向きから、直交する 3 つの向き（Xg 閉じる・Yg 厚み・Zg 近づく）。"""
    cl = closing_tips - (closing_tips @ approach) * approach
    cl /= np.linalg.norm(cl)
    th = np.cross(approach, cl)
    return dict(closing_cam=cl.tolist(), thickness_cam=th.tolist(), approach_cam=approach.tolist(),
                approach_vs_closing_deg=float(np.degrees(np.arccos(np.clip(abs(closing_tips @ approach), 0, 1)))))


def click_points(color: np.ndarray) -> list[tuple[int, int]] | None:
    """窓で 6 点（か先の 2 点だけ）をクリックさせる。q で None。"""
    import cv2
    pts: list[tuple[int, int]] = []
    win = "wrist jaw: 2 tips, then 2 points on each jaw's long inner edge (r=reset, Enter=ok, q=quit)"

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN and len(pts) < 6:
            pts.append((x, y))
            print(f"  {NAMES[len(pts) - 1]}: ({x}, {y})", flush=True)

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = color.copy()
        for i, (x, y) in enumerate(pts):
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
            cv2.putText(img, str(i + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img, f"next: {len(pts) + 1 if len(pts) < 6 else 'ok'}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow(win, img)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord("q"), 27):
            cv2.destroyAllWindows(); return None
        if k == ord("r"):
            pts.clear(); print("  やり直し", flush=True)
        if k in (13, 10) and len(pts) in (2, 6):
            cv2.destroyAllWindows(); return pts


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--serial", default=os.environ.get("GRASP_WRIST_SERIAL", "260522271835"))
    ap.add_argument("--cap", default=None, help="前に撮ったフォルダ（color.png・depth.npy・intrinsics.json）。無ければ撮る")
    ap.add_argument("--px", default=None, help="クリックの代わりに 2 点か 6 点の画素 'u1,v1;u2,v2;...'（窓の無い PC・やり直し用）")
    ap.add_argument("--gripper-deg", type=float, default=None, help="測ったときの爪の角度（lerobot の度）。記録に残すだけ")
    ap.add_argument("-o", "--out", required=True)
    a = ap.parse_args(argv)
    if a.cap:
        color, depth, K = load_capture(a.cap)
        intr = json.load(open(os.path.join(a.cap, "intrinsics.json")))
        cap_dir = a.cap
    else:
        color, depth, K, intr = capture(serial=a.serial)
        cap_dir = os.path.join(HERE, "runs", "wristjaw_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(cap_dir, exist_ok=True)
        save_capture(cap_dir, color, depth, K, intr)
        print(f"[jaw] 撮った: {cap_dir}")
    if a.px:
        px = [tuple(float(v) for v in s.split(",")) for s in a.px.split(";")]
    else:
        px = click_points(color)
        if px is None:
            print("[jaw] 止めた"); return 1
    if len(px) not in (2, 6):
        raise SystemExit("[jaw] 点は 2 つ（左の先・右の先）か 6 つ（続けて左の指の縁 2 つ・右の指の縁 2 つ）")
    m = tips_model(point_at(depth, K, *px[0]), point_at(depth, K, *px[1]))
    if len(px) == 6:
        m.update(axes_model(np.array(m["closing_cam_from_tips"]), approach_from_edges(K, px[2:4], px[4:6])))
        m["edge_px"] = [list(map(float, q)) for q in px[2:]]
    out = dict(_note=f"手首の D405（{intr.get('serial', a.serial)}）から見た爪（grasp/tools/wrist_jaw_measure.py・"
                     f"{datetime.date.today().isoformat()}）。撮った記録: {os.path.relpath(cap_dir, os.path.join(HERE, '..'))}。"
                     + ("近づく向きは指の縁の消失点（edge_px）から" if len(px) == 6 else "爪の先だけ（向きは出していない）"),
               serial=intr.get("serial", a.serial), gripper_deg=a.gripper_deg,
               intrinsics={k: intr[k] for k in ("fx", "fy", "ppx", "ppy", "width", "height", "model", "coeffs") if k in intr},
               left_tip_px=list(map(float, px[0])), right_tip_px=list(map(float, px[1])), **m)
    json.dump(out, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(f"[jaw] 爪の先の間 {m['tip_gap_m']*1000:.0f} mm（開いた幅を定規で測った値と比べる）")
    if "approach_cam" in m:
        print(f"[jaw] 近づく向き {np.round(m['approach_cam'], 3).tolist()}・爪の先から出した閉じる向きとの角 {m['approach_vs_closing_deg']:.1f}°"
              "（90° に近いほど良い。外れたら縁を取り直す）")
    print(f"[jaw] 爪の先の中点（カメラの座標）{np.round(np.array(m['mid_tips_cam_m']) * 1000, 1).tolist()} mm → {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
