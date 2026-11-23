#!/usr/bin/env python3
"""机の面の深度の誤差を測る（平面の残差 RMS と、画素ごとの時間のばらつき・距離ごと）。読むだけで腕は動かさない。

2026-11-22・作業者: Munenori。点群の精度がカメラからの距離でどう変わるかを、今の置き方で数字にする。

測るもの:
- 平面の残差: 机に平面を当てはめ、その面からの距離のばらつき（RMS と、外れに強い σ = 1.4826·MAD）。
  センサーの雑音に加えて、机のたわみ・深度の低周波のゆがみ（ステレオの面の反り）も入る。
- 時間のばらつき: 同じ画素の深さが N フレームでどれだけ揺れるか（--frames 2 以上で撮ったときだけ）。
  センサーの雑音そのもの。平面の残差との差が、ゆがみ・机の凹凸の分。
- 理論値: σz ≈ Z² · subpixel / (fx · baseline)（D435i は baseline 50 mm、subpixel 0.08 を目安）。
- T_base_cam があれば、当てはめた平面の傾きと高さを腕の付け根の座標で出す（ハンドアイの確かめ）。

使い方:
    # 実機のカメラで撮って測る（pyrealsense2 のある lerobot の venv。カメラは 1 プロセスしか開けないので、
    # 収集・推論試験・stream_chest を止めてから。机の上の物はどけておく）
    ~/projects/lerobot/.venv/bin/python grasp/tools/table_rms.py --live
    ~/projects/lerobot/.venv/bin/python grasp/tools/table_rms.py --live --filtered      # capture.py と同じフィルタ後
    # 保存した撮影から測る（capture.py の出力か、--live の出力。カメラ不要）
    python3 grasp/tools/table_rms.py grasp/runs/cap_xxx
    # 机の一部だけ（画素の矩形 u0,v0,u1,v1）
    python3 grasp/tools/table_rms.py grasp/runs/cap_xxx --roi 100,200,540,470

出力: 端末の表と <out>/table_rms.json・residual.png（残差の色地図。cv2 があれば）。
--live の撮影は <out>/depth.npy（中央値）・depth_stack.npy・intrinsics.json にも残すので、後で測り直せる。
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

CHEST_SERIAL = os.environ.get("GRASP_CHEST_SERIAL", "344422070898")
DEFAULT_T = "grasp/config/sites/kitakyushu-collect.T_base_cam.json"
BASELINE_M = {"D435": 0.050, "D405": 0.018}


# ---------- 撮影 ----------

def capture_stack(serial: str, w: int, h: int, fps: int, frames: int, warmup: int, filtered: bool):
    """深度を frames 枚撮る → (stack[N,H,W] m, intr dict)。深度の座標のまま（色に合わせない＝合わせの穴を入れない）。"""
    import pyrealsense2 as rs

    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
    prof = pipe.start(cfg)
    try:
        dev = prof.get_device()
        scale = dev.first_depth_sensor().get_depth_scale()
        chain = []
        if filtered:  # capture.py と同じ順
            chain = [rs.disparity_transform(True), rs.spatial_filter(), rs.temporal_filter(),
                     rs.disparity_transform(False), rs.hole_filling_filter(1)]
        for _ in range(warmup):
            pipe.wait_for_frames()
        stack = []
        for _ in range(frames):
            d = pipe.wait_for_frames().get_depth_frame()
            for f in chain:
                d = f.process(d)
            stack.append(np.asanyarray(d.get_data()).astype(np.float32) * scale)
        intr = d.profile.as_video_stream_profile().intrinsics
        name = dev.get_info(rs.camera_info.name)
    finally:
        pipe.stop()
    K = [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]]
    return np.stack(stack), {"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                             "ppx": intr.ppx, "ppy": intr.ppy, "K": K, "depth_scale": scale,
                             "serial": serial, "camera": name, "filtered": filtered, "frames": frames,
                             "t_unix": time.time()}


# ---------- 計算（numpy だけ。テストから呼ぶ） ----------

def backproject(depth: np.ndarray, K: np.ndarray, zmin: float, zmax: float, roi=None):
    """深度 → (pts[N,3] カメラ座標, 画素の添字 (v,u))。0・範囲外・ROI の外は捨てる。"""
    h, w = depth.shape
    v, u = np.mgrid[0:h, 0:w]
    ok = np.isfinite(depth) & (depth > zmin) & (depth < zmax)
    if roi is not None:
        u0, v0, u1, v1 = roi
        ok &= (u >= u0) & (u < u1) & (v >= v0) & (v < v1)
    z = depth[ok]
    pts = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], 1)
    return pts, (v[ok], u[ok])


def fit_plane(pts: np.ndarray, band: float, up_cam: np.ndarray | None = None, max_tilt_deg: float = 20.0,
              iters: int = 500, ransac_thr: float = 0.01, seed: int = 0):
    """RANSAC で当たりを付け、±band の点で最小二乗を 3 回やり直す → (n, d, inlier mask)。

    n はカメラ側を向く単位法線、面の上の高さ = pts @ n + d。up_cam（カメラ座標の「上」）があれば、
    それから max_tilt_deg 以上傾いた面は取らない（壁や物の側面を机と取り違えないため）。
    """
    rng = np.random.default_rng(seed)
    sub = pts[rng.choice(len(pts), min(len(pts), 20000), replace=False)]
    best, best_cnt = None, -1
    cos_lim = np.cos(np.radians(max_tilt_deg))
    for _ in range(iters):
        p = sub[rng.choice(len(sub), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        if up_cam is not None and abs(n @ up_cam) < cos_lim:
            continue
        cnt = int((np.abs((sub - p[0]) @ n) < ransac_thr).sum())
        if cnt > best_cnt:
            best, best_cnt = (n, -p[0] @ n), cnt
    if best is None:
        raise RuntimeError("平面が見つからない（--max-tilt を緩めるか、ROI を机に合わせる）")
    n, d = best
    for _ in range(3):
        m = np.abs(pts @ n + d) < band
        if m.sum() < 100:
            raise RuntimeError(f"面の近くの点が少ない（{int(m.sum())} 点）")
        c = pts[m].mean(0)
        n = np.linalg.svd(pts[m] - c, full_matrices=False)[2][-1]
        d = -c @ n
    if d < 0:  # カメラ（原点）の側を正に
        n, d = -n, -d
    return n, float(d), np.abs(pts @ n + d) < band


def largest_region(shape, vu, mask: np.ndarray) -> np.ndarray:
    """mask の点のうち、画像の上で 8 近傍につながった一番大きい塊だけを残す（奥の棚など、同じ高さの別の面を除く）。"""
    try:
        import cv2
    except ImportError:
        return mask
    img = np.zeros(shape, np.uint8)
    img[vu[0][mask], vu[1][mask]] = 1
    nlab, lab = cv2.connectedComponents(img, connectivity=8)
    if nlab <= 2:
        return mask
    big = 1 + int(np.argmax(np.bincount(lab.ravel())[1:]))
    return mask & (lab[vu] == big)


def robust_sigma(x: np.ndarray) -> float:
    """外れに強い σ（1.4826 × 中央値からの絶対偏差の中央値）。"""
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


def theory_sigma(z: np.ndarray, fx: float, baseline: float, subpixel: float) -> np.ndarray:
    """ステレオの深度のばらつきの目安 [m]。"""
    return z ** 2 * subpixel / (fx * baseline)


def per_bin(z, res, tstd, edges, fx, baseline, subpixel):
    """距離の区間ごとの表（行の dict のリスト）。"""
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (z >= lo) & (z < hi)
        if m.sum() < 50:
            continue
        zc = float(np.median(z[m]))
        r = {"z_lo": float(lo), "z_hi": float(hi), "z_med": zc, "n": int(m.sum()),
             "rms_mm": float(np.sqrt(np.mean(res[m] ** 2)) * 1e3),
             "sigma_robust_mm": robust_sigma(res[m]) * 1e3,
             "mean_mm": float(res[m].mean() * 1e3),  # 区間ごとの片寄り＝面の反り
             "theory_mm": float(theory_sigma(np.array(zc), fx, baseline, subpixel) * 1e3),
             "spacing_mm": zc / fx * 1e3}
        if tstd is not None:
            r["temporal_mm"] = float(np.nanmedian(tstd[m]) * 1e3)
        rows.append(r)
    return rows


# ---------- 本体 ----------

def load(cap_dir: str):
    """保存した撮影 → (depth[H,W], stack[N,H,W] か None, intr dict)。"""
    depth = np.load(os.path.join(cap_dir, "depth.npy"))
    sp = os.path.join(cap_dir, "depth_stack.npy")
    stack = np.load(sp) if os.path.exists(sp) else None
    with open(os.path.join(cap_dir, "intrinsics.json")) as f:
        intr = json.load(f)
    return depth, stack, intr


def temporal_std(stack: np.ndarray) -> np.ndarray:
    """画素ごとの時間の σ。1 枚でも 0 の（抜けた）フレームがある画素は NaN。"""
    good = (stack > 0).all(0)
    s = stack.std(0)
    s[~good] = np.nan
    return s


def save_heatmap(path, shape, vu, res, clip_mm):
    """残差の色地図（青＝面より奥・赤＝手前、±clip_mm で飽和）。"""
    try:
        import cv2
    except ImportError:
        return False
    img = np.zeros(shape, np.uint8)
    val = np.clip(res * 1e3 / clip_mm, -1, 1)
    img[vu] = ((val + 1) * 127.5).astype(np.uint8).clip(1, 255)
    col = cv2.applyColorMap(img, cv2.COLORMAP_JET)
    col[img == 0] = 0
    cv2.putText(col, f"residual +-{clip_mm:g} mm (red=closer)", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(path, col)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cap_dir", nargs="?", help="保存した撮影（depth.npy と intrinsics.json）。--live なら不要")
    ap.add_argument("--live", action="store_true", help="カメラで撮って測る")
    ap.add_argument("--out", default=None, help="出力先（既定: --live なら grasp/runs/table_rms_<時刻>、ほかは cap_dir）")
    ap.add_argument("--serial", default=CHEST_SERIAL)
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=30, help="撮る枚数（時間のばらつきを出すには 2 以上）")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--filtered", action="store_true", help="capture.py と同じフィルタを掛けてから測る")
    ap.add_argument("--roi", default=None, help="画素の矩形 u0,v0,u1,v1")
    ap.add_argument("--zmin", type=float, default=0.15)
    ap.add_argument("--zmax", type=float, default=2.0)
    ap.add_argument("--band", type=float, default=0.015, help="面からこの距離 [m] までを机の点とする")
    ap.add_argument("--bin", type=float, default=0.1, help="距離の区間の幅 [m]")
    ap.add_argument("--T", default=DEFAULT_T, help="T_base_cam の json（無ければ傾きの制約なし）")
    ap.add_argument("--max-tilt", type=float, default=20.0, help="腕の付け根の z からの面の傾きの上限 [deg]")
    ap.add_argument("--keep-all", action="store_true", help="面から band 以内なら離れた塊も机に入れる")
    ap.add_argument("--baseline", type=float, default=None, help="基線長 [m]（既定: カメラ名から。不明なら D435 の 0.05）")
    ap.add_argument("--subpixel", type=float, default=0.08)
    a = ap.parse_args()

    if a.live:
        out = a.out or f"grasp/runs/table_rms_{time.strftime('%Y%m%d_%H%M%S')}"
        stack, intr = capture_stack(a.serial, a.w, a.h, a.fps, a.frames, a.warmup, a.filtered)
        with np.errstate(all="ignore"), __import__("warnings").catch_warnings():
            __import__("warnings").simplefilter("ignore", RuntimeWarning)
            depth = np.nan_to_num(np.nanmedian(np.where(stack > 0, stack, np.nan), 0)).astype(np.float32)
        os.makedirs(out, exist_ok=True)
        np.save(os.path.join(out, "depth.npy"), depth)
        np.save(os.path.join(out, "depth_stack.npy"), stack.astype(np.float32))  # float16 は 1 m で 0.5 mm 刻みになるので使わない
        with open(os.path.join(out, "intrinsics.json"), "w") as f:
            json.dump(intr, f, indent=2)
        print(f"[table-rms] 撮った: {out}  {a.w}x{a.h} × {a.frames} 枚  フィルタ={'あり' if a.filtered else 'なし'}")
    elif a.cap_dir:
        out = a.out or a.cap_dir
        depth, stack, intr = load(a.cap_dir)
        os.makedirs(out, exist_ok=True)
    else:
        ap.error("cap_dir か --live を渡す")

    K = np.asarray(intr["K"], float)
    fx = float(K[0, 0])
    cam = str(intr.get("camera", ""))
    baseline = a.baseline or next((b for k, b in BASELINE_M.items() if k in cam), BASELINE_M["D435"])
    roi = tuple(int(x) for x in a.roi.split(",")) if a.roi else None

    T = None
    if a.T and os.path.exists(a.T):
        with open(a.T) as f:
            T = np.asarray(json.load(f)["T_base_cam"], float)
    up_cam = T[:3, :3].T @ np.array([0, 0, 1.0]) if T is not None else None

    pts, vu = backproject(depth, K, a.zmin, a.zmax, roi)
    if len(pts) < 1000:
        raise SystemExit(f"[table-rms] 止める: 有効な深度が少ない（{len(pts)} 点）")
    n, d, inl = fit_plane(pts, a.band, up_cam, a.max_tilt)
    if not a.keep_all:  # 一番大きい塊だけで当てはめ直す
        inl = largest_region(depth.shape, vu, inl)
        n, d, _ = fit_plane(pts[inl], a.band, up_cam, a.max_tilt)
        inl = largest_region(depth.shape, vu, np.abs(pts @ n + d) < a.band)
    res = pts[inl] @ n + d
    z = pts[inl, 2]
    vu_in = (vu[0][inl], vu[1][inl])

    tstd = None
    if stack is not None and len(stack) >= 2:
        tstd = temporal_std(stack.astype(np.float32))[vu_in]
    edges = np.arange(np.floor(z.min() / a.bin) * a.bin, z.max() + a.bin, a.bin)
    rows = per_bin(z, res, tstd, edges, fx, baseline, a.subpixel)

    summary = {"n_table": int(inl.sum()), "n_valid": int(len(pts)),
               "rms_mm": float(np.sqrt(np.mean(res ** 2)) * 1e3), "sigma_robust_mm": robust_sigma(res) * 1e3,
               "z_range_m": [float(z.min()), float(z.max())], "plane_cam": [*n.tolist(), d]}
    if tstd is not None:
        summary["temporal_mm"] = float(np.nanmedian(tstd) * 1e3)
    if T is not None:
        n_b = T[:3, :3] @ n
        c_b = T[:3, :3] @ (pts[inl].mean(0)) + T[:3, 3]
        summary["plane_base"] = {"tilt_deg": float(np.degrees(np.arccos(min(1.0, abs(n_b[2]))))),
                                 "height_m_at_center": float(c_b[2])}

    print(f"[table-rms] 机の点 {summary['n_table']} / 有効 {summary['n_valid']}  "
          f"距離 {z.min():.2f}〜{z.max():.2f} m  基線 {baseline * 1e3:.0f} mm・fx {fx:.0f} px")
    print(f"[table-rms] 全体: 残差 RMS {summary['rms_mm']:.2f} mm・σ(MAD) {summary['sigma_robust_mm']:.2f} mm"
          + (f"・時間の σ {summary['temporal_mm']:.2f} mm" if tstd is not None else ""))
    if "plane_base" in summary:
        pb = summary["plane_base"]
        print(f"[table-rms] 付け根の座標: 面の傾き {pb['tilt_deg']:.2f}°・中央の高さ z {pb['height_m_at_center']:.4f} m")
        if pb["tilt_deg"] > 3.0:
            print("[table-rms] 注意: 机が付け根の xy 面から 3° 以上傾いて見える。撮った後にカメラが動いたか、"
                  "--T の T_base_cam がこの撮影のものでない（RMS には効かない）")
    hdr = f"{'距離[m]':>11} {'点数':>7} {'RMS':>6} {'σMAD':>6} {'片寄り':>6} " + \
          (f"{'時間σ':>6} " if tstd is not None else "") + f"{'理論σ':>6} {'点間隔':>6}   (mm)"
    print(hdr)
    for r in rows:
        print(f"{r['z_lo']:.2f}-{r['z_hi']:.2f} {r['n']:>7} {r['rms_mm']:>6.2f} {r['sigma_robust_mm']:>6.2f} "
              f"{r['mean_mm']:>+6.2f} " + (f"{r['temporal_mm']:>6.2f} " if tstd is not None else "")
              + f"{r['theory_mm']:>6.2f} {r['spacing_mm']:>6.2f}")

    rep = {"source": "live" if a.live else a.cap_dir, "intrinsics": {k: intr.get(k) for k in
           ("width", "height", "fx", "camera", "serial", "filtered", "frames")},
           "params": {"band": a.band, "roi": roi, "zmin": a.zmin, "zmax": a.zmax, "baseline": baseline,
                      "subpixel": a.subpixel, "T": a.T if T is not None else None},
           "summary": summary, "bins": rows}
    with open(os.path.join(out, "table_rms.json"), "w") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    hm = save_heatmap(os.path.join(out, "residual.png"), depth.shape, vu_in, res, clip_mm=max(3.0, 3 * summary["sigma_robust_mm"]))
    print(f"[table-rms] 書いた: {out}/table_rms.json" + (f"・{out}/residual.png" if hm else ""))


if __name__ == "__main__":
    main()
