"""手首の D405 の部品（numpy だけの計算と、マーカーの測り方）。

2026-11-23・作業者: Munenori。北九州の左のグリッパーに D405 を付けた（爪とカメラが一緒に固定）。
- marker_in_cam: 手首のカメラの座標で、机の ArUco マーカーの中心を測る
- fit_axes: 爪を自分の向きに少しずつ動かしたときのマーカーの動きから、カメラの座標での爪の向き（閉じる・厚み・近づく）を出す
  （grasp/tools/point_test.py --jog が使う）

爪の座標（GraspGenX と plan.py と同じ）: +X = 閉じる向き、+Y = 爪の厚み、+Z = 近づく向き。左腕では link7 の (+y, -x, +z)。
"""
from __future__ import annotations

import numpy as np


def fit_axes(deltas_g: np.ndarray, shifts_cam: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """爪を爪の座標で deltas_g (N,3) [m] だけ動かしたとき、止まっているマーカーがカメラの座標で shifts_cam (N,3) [m] 動いた。

    カメラは爪と一緒に動くので shifts_cam ≈ -R deltas_g（R = 爪の座標 → カメラの座標の回転）。
    R を直交行列の範囲で最小二乗に当てはめる（Kabsch）。返り値: (R (3,3)・列が Xg・Yg・Zg のカメラの座標の向き,
    動きの大きさの比 |shift| / |delta| (N,)・1 から離れれば腕が指令どおりに動いていない, 当てはめの残り (N,) [m])。"""
    D = np.asarray(deltas_g, float)
    P = -np.asarray(shifts_cam, float)
    H = D.T @ P
    U, _, Vt = np.linalg.svd(H)
    S = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ S @ U.T
    ratio = np.linalg.norm(P, axis=1) / np.maximum(np.linalg.norm(D, axis=1), 1e-9)
    resid = np.linalg.norm(P - D @ R.T, axis=1)
    return R, ratio, resid


def start_pipeline(cfg, serial: str):
    """RealSense の pipeline を始めて 1 フレーム待つ。来なければカメラを再起動（hardware_reset）して 1 回だけやり直す。

    2026-11-23: 手首の D405 を付けるときに何度も抜き差しした後、D405 も胸の D435i も USB に見えるのにフレームが来なくなった。
    どちらも hardware_reset で直った。返り値: (pipeline, profile)。"""
    import time
    import pyrealsense2 as rs
    pipe = rs.pipeline()
    try:
        prof = pipe.start(cfg)
        pipe.wait_for_frames(5000)
        return pipe, prof
    except RuntimeError as e:
        print(f"[rs] {serial}: フレームが来ない（{e}）。カメラを再起動して 1 回だけやり直す", flush=True)
        try:
            pipe.stop()
        except RuntimeError:
            pass
    next(d for d in rs.context().devices if d.get_info(rs.camera_info.serial_number) == serial).hardware_reset()
    for _ in range(25):
        time.sleep(1)
        if serial in [d.get_info(rs.camera_info.serial_number) for d in rs.context().devices]:
            break
    time.sleep(2)
    pipe = rs.pipeline()
    prof = pipe.start(cfg)
    pipe.wait_for_frames(5000)
    return pipe, prof


class WristMarker:
    """手首の D405 を開いたままにして、ArUco マーカーの中心をカメラの座標で測る（lerobot の venv・pyrealsense2）。"""

    def __init__(self, serial: str, w: int = 640, h: int = 480, fps: int = 30):
        import cv2
        import pyrealsense2 as rs
        self.cv2, self.rs = cv2, rs
        ctx = rs.context()
        if serial not in [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]:
            raise SystemExit(f"[wrist] 手首の D405 {serial} が見えない（USB を挿し直す）")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.pipe, prof = start_pipeline(cfg, serial)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.det = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
        for _ in range(20):  # 露出が落ち着くまで
            self.pipe.wait_for_frames(5000)
        self.last_color = None

    def measure(self, marker: int, n: int = 15) -> np.ndarray | None:
        """マーカーの中心（カメラの座標 [m]・n 枚の中央値）。半分より多くの枚で見えなければ None。"""
        cv2 = self.cv2
        pts = []
        for _ in range(n):
            fs = self.align.process(self.pipe.wait_for_frames(5000))
            col = np.asanyarray(fs.get_color_frame().get_data())
            dep = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.scale
            i = fs.get_color_frame().profile.as_video_stream_profile().intrinsics
            corners, ids, _ = self.det.detectMarkers(cv2.cvtColor(col, cv2.COLOR_BGR2GRAY))
            self.last_color = cv2.aruco.drawDetectedMarkers(col.copy(), corners, ids) if ids is not None else col
            if ids is None or marker not in ids.ravel():
                continue
            c = corners[list(ids.ravel()).index(marker)].reshape(4, 2)
            u, v = c.mean(0)
            # 中心のまわりの内側（角から 1/4 内）だけの深さ（縁の外の物を拾わない）
            q = c.mean(0) + (c - c.mean(0)) * 0.5
            x0, x1, y0, y1 = int(q[:, 0].min()), int(q[:, 0].max()) + 1, int(q[:, 1].min()), int(q[:, 1].max()) + 1
            zs = dep[y0:y1, x0:x1]
            zs = zs[(zs > 0.05) & (zs < 1.0)]
            if len(zs) < 10:
                continue
            z = float(np.median(zs))
            pts.append([(u - i.ppx) * z / i.fx, (v - i.ppy) * z / i.fy, z])
        if len(pts) <= n // 2:
            return None
        return np.median(np.array(pts), axis=0)

    def close(self) -> None:
        try:
            self.pipe.stop()
        except RuntimeError:
            pass


class WristCam:
    """手首の D405 を開いたままにして、色と深度（数枚の中央値 [m]）を撮る（grasp/wrist_approach.py が使う）。"""

    def __init__(self, serial: str, w: int = 640, h: int = 480, fps: int = 30):
        import pyrealsense2 as rs
        ctx = rs.context()
        if serial not in [d.get_info(rs.camera_info.serial_number) for d in ctx.devices]:
            raise SystemExit(f"[wrist] 手首の D405 {serial} が見えない（USB を挿し直す）")
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, w, h, rs.format.bgr8, fps)
        cfg.enable_stream(rs.stream.depth, w, h, rs.format.z16, fps)
        self.pipe, prof = start_pipeline(cfg, serial)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        for _ in range(20):  # 露出が落ち着くまで
            self.pipe.wait_for_frames(5000)

    def grab(self, n: int = 5, settle_s: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
        """止まってから settle_s 待ち、n 枚の深度の中央値と最後の色を返す。古いフレームは捨てる。"""
        import time
        time.sleep(settle_s)
        for _ in range(3):
            self.pipe.wait_for_frames(5000)
        ds, col = [], None
        for _ in range(n):
            fs = self.align.process(self.pipe.wait_for_frames(5000))
            d = np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * self.scale
            ds.append(np.where(d > 0, d, np.nan))
            col = np.asanyarray(fs.get_color_frame().get_data()).copy()
        import warnings
        with warnings.catch_warnings():  # どの枚でも深さの無い画素は All-NaN の警告を出すので黙らせる（0 にする）
            warnings.simplefilter("ignore", RuntimeWarning)
            dm = np.nanmedian(np.stack(ds), axis=0)
        return col, np.nan_to_num(dm, nan=0.0).astype(np.float32)

    def close(self) -> None:
        try:
            self.pipe.stop()
        except RuntimeError:
            pass
