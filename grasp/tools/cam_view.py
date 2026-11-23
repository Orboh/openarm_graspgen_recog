#!/usr/bin/env python3
"""カメラの向きを合わせるための実時間の映像（色・深度を横に並べ、傾きと距離を重ねる）。腕は動かさない。

2026-11-22・作業者: Munenori。胸のカメラの画角を付け替えるときに見る。

画面に出すもの:
- 左: 色。中央の十字と 3×3 の格子
- 右: 深度の色地図（近い=赤・遠い=青。--zmax で上限）
- 上の行: IMU の重力から出したカメラの下向きの角度（pitch）と左右の傾き（roll）、中央の 21×21 画素の深さの中央値、
  深度の取れた割合。D435i・D455 は IMU があるので出る（D405 は無いので出さない）

使い方（pyrealsense2 のある lerobot の venv。カメラは 1 プロセスしか開けないので、収集・推論試験・stream_chest を止めてから）:
    ~/projects/lerobot/.venv/bin/python grasp/tools/cam_view.py              # GRASP_CHEST_SERIAL、無ければ挿さっている 1 台目
    ~/projects/lerobot/.venv/bin/python grasp/tools/cam_view.py --serial 409122272149 --zmax 0.6
キー: q / esc で終わる、s で今の色・深度を grasp/runs/view_<時刻>/ に保存（capture.py と同じ形）
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np

CHEST_SERIAL = os.environ.get("GRASP_CHEST_SERIAL", "344422070898")


def pick_serial(want: str) -> str:
    """want が挿さっていればそれ、無ければ 1 台目（その旨を出す）。1 台も無ければ止める。"""
    import pyrealsense2 as rs

    devs = {d.get_info(rs.camera_info.serial_number): d.get_info(rs.camera_info.name) for d in rs.context().devices}
    if not devs:
        raise SystemExit("[cam-view] 止める: RealSense が 1 台も見えない（USB を挿し直す）")
    if want in devs:
        return want
    s = next(iter(devs))
    print(f"[cam-view] {want} は見えないので {s}（{devs[s]}）を開く。挿さっている: {devs}")
    return s


def tilt_from_accel(a: np.ndarray) -> tuple[float, float]:
    """IMU の加速度（静止時は重力の反力＝上向き）→ (pitch 下向きが正 [deg], roll 右下がりが正 [deg])。

    RealSense の IMU の軸は光学の座標と同じ（+x 右・+y 下・+z 前）。静止時の測定値は上向き＝ -y 寄り。
    """
    g = -a / (np.linalg.norm(a) + 1e-9)  # 重力の向き（下）
    pitch = math.degrees(math.atan2(g[2], g[1]))   # 光軸が水平から下へ向いた角度
    roll = math.degrees(math.atan2(-g[0], g[1]))
    return pitch, roll


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--serial", default=CHEST_SERIAL)
    ap.add_argument("--w", type=int, default=848)
    ap.add_argument("--h", type=int, default=480)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--zmax", type=float, default=1.5, help="深度の色地図の上限 [m]")
    a = ap.parse_args()

    import cv2
    import pyrealsense2 as rs

    serial = pick_serial(a.serial)
    pipe, cfg = rs.pipeline(), rs.config()
    cfg.enable_device(serial)
    cfg.enable_stream(rs.stream.color, a.w, a.h, rs.format.bgr8, a.fps)
    cfg.enable_stream(rs.stream.depth, a.w, a.h, rs.format.z16, a.fps)
    dev = next(d for d in rs.context().devices if d.get_info(rs.camera_info.serial_number) == serial)
    has_imu = any(sen.is_motion_sensor() for sen in dev.query_sensors())
    if has_imu:
        cfg.enable_stream(rs.stream.accel)
    try:
        prof = pipe.start(cfg)
    except RuntimeError as e:
        if not has_imu:
            raise
        # IMU の読み取りに権限が要る（librealsense の udev の規則が無い PC）。映像だけで開き直す
        print(f"[cam-view] IMU を開けないので映像だけにする（傾きは出ない）: {str(e).splitlines()[0][-80:]}")
        has_imu = False
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, a.w, a.h, rs.format.bgr8, a.fps)
        cfg.enable_stream(rs.stream.depth, a.w, a.h, rs.format.z16, a.fps)
        prof = pipe.start(cfg)
    scale = prof.get_device().first_depth_sensor().get_depth_scale()
    align = rs.align(rs.stream.color)
    win = f"cam_view {serial}  (q: quit / s: save)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    acc = None
    fails = 0
    kk = None
    print(f"[cam-view] 開いた: {serial}  {a.w}x{a.h}@{a.fps}  IMU={'あり' if has_imu else 'なし'}。q か esc で終わる")
    try:
        while True:
            try:
                fs = pipe.wait_for_frames(3000)
            except RuntimeError:  # 立ち上がりや USB の取りこぼしで来ないことがある。開き直す（3 回まで）
                fails += 1
                if fails > 3:
                    raise SystemExit("[cam-view] 止める: 映像が来ない（USB を挿し直す。ほかのプロセスがカメラを開いていないか）")
                print(f"[cam-view] 映像が来ないので開き直す（{fails}/3）")
                try:
                    pipe.stop()
                except RuntimeError:
                    pass
                try:
                    prof = pipe.start(cfg)
                except RuntimeError as e:  # カメラが抜けている（マウントの付け替えなど）
                    raise SystemExit(f"[cam-view] 止める: カメラを開き直せない（{e}）。USB を挿し直して打ち直す")
                continue
            if has_imu:
                m = fs.first_or_default(rs.stream.accel)
                if m:
                    d = m.as_motion_frame().get_motion_data()
                    v = np.array([d.x, d.y, d.z])
                    acc = v if acc is None else 0.9 * acc + 0.1 * v  # 揺れを均す
            fs = align.process(fs)
            cf, df = fs.get_color_frame(), fs.get_depth_frame()
            if not cf or not df:
                continue
            color = np.asanyarray(cf.get_data())
            if kk is None:
                ci = cf.profile.as_video_stream_profile().intrinsics
                kk = np.array([[ci.fx, 0, ci.ppx], [0, ci.fy, ci.ppy], [0, 0, 1]])
            depth = np.asanyarray(df.get_data()).astype(np.float32) * scale
            h, w = depth.shape

            dv = cv2.applyColorMap(cv2.convertScaleAbs(np.clip(depth, 0, a.zmax), alpha=255 / a.zmax), cv2.COLORMAP_JET)
            dv[depth == 0] = 0
            view = np.hstack([color.copy(), dv])
            for img_x0 in (0, w):  # 格子と十字を両方に
                for k in (1, 2):
                    cv2.line(view, (img_x0 + w * k // 3, 0), (img_x0 + w * k // 3, h), (200, 200, 200), 1)
                    cv2.line(view, (img_x0, h * k // 3), (img_x0 + w, h * k // 3), (200, 200, 200), 1)
                cv2.drawMarker(view, (img_x0 + w // 2, h // 2), (0, 255, 255), cv2.MARKER_CROSS, 30, 2)

            c = depth[h // 2 - 10:h // 2 + 11, w // 2 - 10:w // 2 + 11]
            zc = float(np.median(c[c > 0])) if (c > 0).any() else float("nan")
            txt = f"center {zc:.3f} m  valid {np.mean(depth > 0):.0%}"
            for name, vr in (("near", h - 15), ("mid", h // 2)):  # その行に写る横幅（両腕の範囲が入るかを見る）
                row = depth[vr]
                ok = row > 0
                if ok.sum() > 10:
                    u = np.where(ok)[0]
                    x = (u - kk[0, 2]) * row[ok] / kk[0, 0]
                    txt += f"  {name}: z {np.median(row[ok]):.2f} m w {x.max() - x.min():.2f} m"
            if acc is not None:
                p, r = tilt_from_accel(acc)
                txt = f"pitch(down) {p:+.1f} deg  roll {r:+.1f} deg  " + txt
            cv2.rectangle(view, (0, 0), (view.shape[1], 30), (0, 0, 0), -1)
            cv2.putText(view, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, view)

            k = cv2.waitKey(1) & 0xFF
            try:  # Qt 版は窓を閉じた後にこの問い合わせで例外を出す
                closed = cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1
            except cv2.error:
                closed = True
            if k in (ord("q"), 27) or closed:
                break
            if k == ord("s"):
                out = f"grasp/runs/view_{time.strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(out, exist_ok=True)
                intr = cf.profile.as_video_stream_profile().intrinsics
                cv2.imwrite(os.path.join(out, "color.png"), color)
                np.save(os.path.join(out, "depth.npy"), depth)
                with open(os.path.join(out, "intrinsics.json"), "w") as f:
                    json.dump({"width": intr.width, "height": intr.height, "fx": intr.fx, "fy": intr.fy,
                               "ppx": intr.ppx, "ppy": intr.ppy, "depth_scale": scale, "serial": serial,
                               "K": [[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]],
                               "imu_accel": None if acc is None else acc.tolist(), "t_unix": time.time()}, f, indent=2)
                print(f"[cam-view] 保存した: {out}")
    finally:
        try:
            pipe.stop()
        except RuntimeError:  # 開き直しに失敗して止まっている
            pass
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
