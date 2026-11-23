#!/usr/bin/env python3
"""運ぶ途中に割り込む物（手・箱・棒）を胸のカメラで見ながら、止まって経路を作り直して動く（2026-11-21 Munenori の依頼）。

lerobot の venv で動く（pyrealsense2・numpy・pyzmq）。当たり判定と回り込みは cuRobo のサーバー（grasp/curobo/server.py）に聞く。
- Watcher（スレッド）: 胸の D435i の深度を毎フレーム、腕の付け根の座標の 2 cm の格子にする（机より上・届く範囲）。
  動き出す前の 30 フレームで背景（机の上の物・休みの姿勢の腕・ノイズ）を覚え、背景に無く、直近 3 フレームのうち 2 フレーム以上に
  写った格子だけを割り込んだ物とする（背景を覚える間は何も割り込ませない）
- Guard（スレッド）: 最新の点と今の目標の角度をサーバーに送って障害物の箱にし（腕の球のまわりと、手にくっつけて動かした
  持っている物の点のまわりは消す）、
  ・割り込んだ物が腕の NEAR 以内 → 止める
  ・残りの経路が当たる → 止めて、今の目標の角度から行き先までの回り込みを作り直す（作れたら新しい経路で動き出す）
  ・映像が STALE 秒より古い → 止める
  止めている間は待ち続ける（やめるのは操作する人の Ctrl-C）
- follow(): 毎秒 100 回、経路に沿って目標の角度を進めて送る（MIT の位置の指令・重力の補償つき）。止める合図の間は
  目標を進めない。速さの変わり方に上限（ACCEL）を付けて、止まるのも動き出すのも滑らかにする

腕を動かさない確かめ（カメラと判定だけ・腕はつながない）:
    set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
    ~/projects/lerobot/.venv/bin/python grasp/reactive.py --watch --q <関節 1〜7> --q2 <関節 1〜7>
"""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from typing import Callable, List, Optional, Sequence

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from grasp.curobo import client as curobo  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
J = [f"joint_{i}" for i in range(1, 8)]
NEAR = float(os.environ.get("GRASP_AVOID_NEAR", "0.05"))     # 割り込んだ物が腕のこの距離 [m] に入ったら止まる
STALE = 0.3                 # 映像がこれより古ければ止まる [秒]
ACCEL = 40.0                # 目標の角度の速さの変わり方の上限 [deg/s²]
MAX_HOLD = float(os.environ.get("GRASP_AVOID_MAX_HOLD", "inf"))  # 止まったままこれだけたったら Stuck [秒]。既定は待ち続ける
# （物を持ったまま諦めて戻ると、持った物を机に押し付ける。やめるのは操作する人の Ctrl-C で、run_grasp.py が元の場所に下ろす）
STEP = 0.01                 # 指令の間隔 [秒]
# 割り込む物の箱: 動くので周りに広げ（PAD）、机までは伸ばさない。腕のまわり ARM_R・持っている物のまわり TOOL_R の点は消す
BOX_KW = dict(pad=0.03, back=0.03, voxel=0.02, min_voxels=3, to_floor=False, quiet=True)
ARM_R = 0.05                # 腕の球のまわりのこの距離の点は腕自身として消す [m]
HELD_R = 0.05               # 持っている物の点（手にくっつけて動かす）のまわりのこの距離の点は消す [m]。
# 3 cm では、撮った時に見えた上の面の点だけでは下の面と掴んだずれの分が残り、指の 2 cm 先の「割り込んだ物」に見えて
# 止まる・動き出すを 0.1 秒ごとにくり返した（2026-11-21 18:29 の実機）
CLEAR_FOR = 0.5             # 止めた後、何も無い状態がこれだけ続いたら動き出す [秒]（行ったり来たりしない）
MOTOR_STALE = 0.3           # モーターの返事がこれより古ければ、指令をやめて終わる [秒]（非常停止・電源断に気付く）。
# 0.1 秒では、動き出す前の確かめの間（前の段からの間）だけで 0.11 秒になり、誤って止めた（2026-11-21 18:4x）


GRID = 0.02
_OFF = 1 << 20


def _keys(ix: np.ndarray) -> np.ndarray:
    """整数の格子の番号 (N, 3) を 1 つの int64 に。"""
    ix = ix + _OFF
    return (ix[:, 0] << 42) | (ix[:, 1] << 21) | ix[:, 2]


def _unkeys(k: np.ndarray) -> np.ndarray:
    m = (1 << 21) - 1
    return np.stack([(k >> 42) & m, (k >> 21) & m, k & m], 1).astype(np.float64) - _OFF


_NEIGHBORS = _keys(np.array([[i, j, k] for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])) - _keys(np.zeros((1, 3), np.int64))


class MotorLost(SystemExit):
    """モーターの返事が途切れた（非常停止・電源断・通信の断）。これ以上は指令を送らない。"""


def motor_age(bus, names: Sequence[str]) -> float:
    """names のモーターで、最後に返事が来てからの一番長い時間 [秒]（lerobot の damiao の bus の記録）。"""
    now = time.monotonic()
    return max(now - bus._last_update_ts.get(m, 0.0) for m in names)


class Stuck(SystemExit):
    """止まったまま MAX_HOLD 秒たった・回り込みも作れない。"""


class Watcher(threading.Thread):
    """胸のカメラの深度を、腕の付け根の座標の点（机より上・届く範囲・2 cm 格子）にし続ける。"""

    def __init__(self, serial: str, T_bc: np.ndarray, floor_z: float, decimate: int = 4, bg_frames: int = 30):
        super().__init__(daemon=True)
        self.serial, self.T_bc, self.floor_z, self.dec = serial, T_bc, floor_z, decimate
        self.points: np.ndarray = np.zeros((0, 3))   # 背景に無い・続けて写った格子の中心（割り込んだ物）
        self.stamp = 0.0                             # 背景を覚え終わるまで 0（Guard は止めたまま）
        self.error: Optional[str] = None
        self.bg_frames = bg_frames                   # 動き出す前の、何も割り込んでいない様子を覚えるフレーム数
        self.bg: Optional[np.ndarray] = None
        self._recent: List[np.ndarray] = []
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            import pyrealsense2 as rs
            pipe, cfg = rs.pipeline(), rs.config()
            cfg.enable_device(self.serial)
            cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            prof = pipe.start(cfg)
            dprof = prof.get_stream(rs.stream.depth).as_video_stream_profile()
            cprof = prof.get_stream(rs.stream.color)
            ex = dprof.get_extrinsics_to(cprof)  # 深度のカメラ → カラーのカメラ（T_base_cam はカラーの座標）
            T_cd = np.eye(4); T_cd[:3, :3] = np.array(ex.rotation).reshape(3, 3).T; T_cd[:3, 3] = ex.translation
            T = self.T_bc @ T_cd
            intr = dprof.get_intrinsics()
            scale = prof.get_device().first_depth_sensor().get_depth_scale()
            d = self.dec
            v, u = np.mgrid[0:480:d, 0:640:d]
            rx, ry = (u - intr.ppx) / intr.fx, (v - intr.ppy) / intr.fy
            for _ in range(15):
                pipe.wait_for_frames()
            while not self._stop.is_set():
                fs = pipe.wait_for_frames()
                z = np.asanyarray(fs.get_depth_frame().get_data())[::d, ::d].astype(np.float32) * scale
                ok = (z > 0.1) & (z < 1.2)
                P = np.stack([rx[ok] * z[ok], ry[ok] * z[ok], z[ok]], 1) @ T[:3, :3].T + T[:3, 3]
                h = P[:, 2] - self.floor_z
                P = P[(h > 0.02) & (h < 0.60) & (P[:, 0] > -0.25) & (P[:, 0] < 0.9) & (P[:, 1] > -0.6) & (P[:, 1] < 0.7)]
                keys = _keys(np.floor(P / GRID).astype(np.int64)) if len(P) else np.zeros(0, np.int64)
                if self.bg is None:  # 背景を覚える（机の上の物・腕の休みの姿勢・深度のノイズ）。1 格子ぶん太らせる
                    self._recent.append(np.unique(keys))
                    if len(self._recent) >= self.bg_frames:
                        allk = np.unique(np.concatenate(self._recent))
                        self.bg = np.unique((allk[:, None] + _NEIGHBORS[None, :]).ravel())
                        self._recent = []
                        print(f"[avoid] 背景を覚えた（格子 {len(allk)} 個）", flush=True)
                    continue
                keys = np.unique(keys[~np.isin(keys, self.bg)])
                self._recent = (self._recent + [keys])[-3:]
                if len(self._recent) >= 2:  # 直近 3 フレームのうち 2 フレーム以上に写った格子だけ（ちらつきを抑える）
                    allk, cnt = np.unique(np.concatenate(self._recent), return_counts=True)
                    keys = allk[cnt >= 2]
                self.points, self.stamp = (_unkeys(keys) + 0.5) * GRID, time.time()
            pipe.stop()
        except Exception as e:  # noqa: BLE001  映像が止まれば stamp が古くなり Guard が止める
            self.error = f"{type(e).__name__}: {e}"

    def stop(self) -> None:
        self._stop.set()


class Follower:
    """経路に沿って目標の角度を進め、Guard が止める・経路を入れ替える。"""

    def __init__(self, path: Sequence[Sequence[float]], speed_dps: float):
        self.lock = threading.Lock()
        self.path = [np.asarray(q, float) for q in path]  # path[0] は今の目標
        self.k = 1                                         # 次に向かう経由点
        self.q_ref = self.path[0].copy()
        self.v = 0.0
        self.speed = speed_dps
        self.hold_reason: Optional[str] = None
        self.hold_since: Optional[float] = None
        self.done = False

    def remaining(self) -> List[np.ndarray]:
        with self.lock:
            return [self.q_ref.copy()] + [q.copy() for q in self.path[self.k:]]

    def set_hold(self, reason: Optional[str]) -> None:
        with self.lock:
            if reason and self.hold_reason is None:
                self.hold_since = time.time()
                print(f"[avoid] 止まる: {reason}", flush=True)
            elif not reason and self.hold_reason is not None:
                print(f"[avoid] 動き出す（{time.time() - self.hold_since:.1f} 秒止まった）", flush=True)
                self.hold_since = None
            self.hold_reason = reason

    def replace(self, vias: List[List[float]]) -> None:
        with self.lock:
            self.path = [self.q_ref.copy()] + [np.asarray(v, float) for v in vias]
            self.k = 1

    def step(self, dt: float) -> np.ndarray:
        """目標の角度を dt 秒ぶん進める（L∞ の距離で一定の速さ・加速度の上限つき）。"""
        with self.lock:
            target_v = 0.0 if (self.hold_reason or self.done) else self.speed
            if self.k < len(self.path):  # 残りの距離で止まれる速さまでに抑える
                left = float(np.max(np.abs(self.path[-1] - self.q_ref))) if self.k == len(self.path) - 1 else np.inf
                target_v = min(target_v, np.sqrt(2 * ACCEL * max(left, 0.0)) + 0.5)
            dv = np.clip(target_v - self.v, -ACCEL * dt, ACCEL * dt)
            self.v = max(0.0, self.v + dv)
            move = self.v * dt
            while move > 0 and self.k < len(self.path):
                d = self.path[self.k] - self.q_ref
                n = float(np.max(np.abs(d)))
                if n <= move:
                    self.q_ref = self.path[self.k].copy(); self.k += 1; move -= n
                else:
                    self.q_ref = self.q_ref + d * (move / n); move = 0
            if self.k >= len(self.path):
                self.done = True
            return self.q_ref.copy()


class Guard(threading.Thread):
    """カメラの点で障害物を入れ替え、近すぎ・経路が当たる・映像が古いときに止め、回り込みを作り直す。"""

    def __init__(self, fol: Follower, watcher: Watcher, cam_pos, floor_z: float,
                 floor_for: Optional[Callable] = None, period: float = 0.07, held_l7: Optional[np.ndarray] = None):
        super().__init__(daemon=True)
        self.fol, self.w, self.cam_pos, self.floor_z, self.floor_for, self.period = fol, watcher, cam_pos, floor_z, floor_for, period
        self.held_l7 = held_l7  # 持っている物の点（link7 の座標）。手と一緒に動かして、そのまわりの点を消す
        self._stop = threading.Event()
        self.clear_since: Optional[float] = None
        self.n_boxes = 0
        self.error: Optional[str] = None

    def run(self) -> None:
        try:
            while not self._stop.is_set() and not self.fol.done:
                t0 = time.time()
                self.once()
                time.sleep(max(0.0, self.period - (time.time() - t0)))
        except Exception as e:  # noqa: BLE001  判定が死んだら止めたままにする
            self.error = f"{type(e).__name__}: {e}"
            self.fol.set_hold(f"判定の誤り（{self.error}）")

    def once(self) -> None:
        if time.time() - self.w.stamp > STALE:
            self.fol.set_hold("背景を覚えている途中" if self.w.bg is None and self.w.error is None else
                              f"映像が {time.time() - self.w.stamp:.1f} 秒古い" + (f"（{self.w.error}）" if self.w.error else ""))
            return
        rem = self.fol.remaining()
        q = rem[0]
        kw = {}
        if self.held_l7 is not None and len(self.held_l7):
            kw = dict(exclude_points=held_points(q, self.held_l7).tolist(), exclude_r=HELD_R)
        r = curobo.set_obstacle_points(self.w.points, [q], self.cam_pos, self.floor_z, arm_margin=ARM_R, **kw, **BOX_KW)
        self.n_boxes = len(r["boxes"])
        near = [b for b in curobo.violations([q, q], margin=NEAR, table_margin=None) if b[1].startswith("seen_")]
        if near:
            self.clear_since = None
            self.fol.set_hold(f"割り込んだ物が腕の {NEAR * 100:.0f} cm 以内（{curobo.describe(near[:1])}）")
            return
        bad = [b for b in curobo.violations(rem, table_margin=None) if b[1].startswith("seen_")] if len(rem) >= 2 else []
        if not bad:
            self.clear_since = self.clear_since or time.time()
            if self.fol.hold_reason is None or time.time() - self.clear_since >= CLEAR_FOR:
                self.fol.set_hold(None)
            return
        self.clear_since = None
        self.fol.set_hold(f"先の経路に割り込んだ物（{curobo.describe(bad[:1])}）")
        goal = rem[-1]
        fl = self.floor_for(q, goal) if self.floor_for else None
        t0 = time.time()
        vias = curobo.detour(q, goal, floor=fl)
        if vias:
            self.fol.replace(vias)
            print(f"[avoid] 回り込みを作り直した（経由点 {len(vias)} 個・{time.time() - t0:.2f} 秒）", flush=True)
            self.fol.set_hold(None)

    def stop(self) -> None:
        self._stop.set()


def link7_pose(q_deg) -> np.ndarray:
    """link7 の 4x4（腕の付け根の座標）。Pinocchio を使わず URDF の小さな FK（grasp/urdf_fk.py）で。"""
    from grasp.urdf_fk import LeftArmFK
    global _FK
    if _FK is None:
        _FK = LeftArmFK()
    return _FK.link_poses(np.deg2rad(q_deg))["openarm_left_link7"]


def held_in_link7(obj_base: np.ndarray, q_grasp_deg) -> np.ndarray:
    """撮った物の点（腕の付け根の座標）を、掴んだ姿勢の link7 の座標にして 2 cm に間引く（手にくっつけて動かす用）。"""
    L7 = link7_pose(q_grasp_deg)
    P = (obj_base - L7[:3, 3]) @ L7[:3, :3]
    return np.unique(np.round(P / 0.02), axis=0) * 0.02


def held_points(q_deg, held_l7: np.ndarray) -> np.ndarray:
    """今の姿勢で、持っている物の点（腕の付け根の座標）。"""
    L7 = link7_pose(q_deg)
    return held_l7 @ L7[:3, :3].T + L7[:3, 3]


_FK = None


def follow(bus, grav: Callable, path: Sequence[Sequence[float]], speed_dps: float, gripper_deg: float,
           watcher: Watcher, cam_pos, floor_z: float, floor_for: Optional[Callable] = None, phase: str = "",
           held_l7: Optional[np.ndarray] = None) -> List[float]:
    """path（[今の目標, 経由点..., 行き先]・lerobot の度）を、割り込む物をよけながら動く。行き先に着いたらその角度を返す。
    止まったまま MAX_HOLD 秒で Stuck。"""
    from lerobot.motors.damiao.damiao_alignment import OPENARM_ALIGN_KD, OPENARM_ALIGN_KP
    fol = Follower(path, speed_dps)
    guard = Guard(fol, watcher, cam_pos, floor_z, floor_for, held_l7=held_l7)
    guard.once()  # 動き出す前に 1 度確かめる
    guard.start()
    print(f"[avoid] {phase}: 見ながら動く（{speed_dps:.0f}°/秒・経由点 {len(path) - 1} 個）", flush=True)
    t_next = time.time()
    bus._mit_control_batch({m: (OPENARM_ALIGN_KP[m], OPENARM_ALIGN_KD[m], v, 0.0, float(grav(dict(zip(J, fol.q_ref.tolist()))).get(m, 0.0)))
                            for m, v in list(zip(J, fol.q_ref.tolist())) + [("gripper", float(gripper_deg))]})  # 返事の時刻を今にそろえる
    try:
        while not fol.done:
            q = fol.step(STEP)
            target = {j: float(q[i]) for i, j in enumerate(J)}
            target["gripper"] = float(gripper_deg)
            ff = grav(target)
            bus._mit_control_batch({m: (OPENARM_ALIGN_KP[m], OPENARM_ALIGN_KD[m], target[m], 0.0, float(ff.get(m, 0.0)))
                                    for m in target})
            age = motor_age(bus, J)
            if age > MOTOR_STALE:  # 返事が無い: 指令が届いていない。これ以上送らない
                raise MotorLost(f"[avoid] モーターの返事が {age:.2f} 秒来ない（非常停止・電源・通信）。指令をやめる")
            if fol.hold_since and time.time() - fol.hold_since > MAX_HOLD:
                raise Stuck(f"{MAX_HOLD:.0f} 秒止まったまま（{fol.hold_reason}）")
            t_next += STEP
            time.sleep(max(0.0, t_next - time.time()))
    finally:
        guard.stop()
    return fol.q_ref.tolist()


def _watch_main() -> int:
    """腕を動かさずに、カメラと判定だけを回す（Follower の目標の角度は動かさない）。"""
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--watch", action="store_true")
    ap.add_argument("--q", required=True, help="腕の今の角度（lerobot の度・7 つ）")
    ap.add_argument("--q2", required=True, help="仮の行き先（lerobot の度・7 つ）。この直線の経路で判定する")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--cap", default=None, help="記録のフォルダ: その物を持っている（掴んだ姿勢の手にくっついている）として消す")
    a = ap.parse_args()
    T_bc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
    fz = float(os.environ["GRASP_FLOOR_Z"])
    w = Watcher(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"), T_bc, fz)
    w.start()
    q1, q2 = [float(v) for v in a.q.split(",")], [float(v) for v in a.q2.split(",")]
    fol = Follower([q1, q2], 0.0)
    held = None
    if a.cap:
        pl = json.load(open(os.path.join(a.cap, "plan.json")))
        ob = np.load(os.path.join(a.cap, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3]
        held = held_in_link7(ob, pl["q_grasp_deg"])
    g = Guard(fol, w, T_bc[:3, 3], fz, held_l7=held)
    t_end = time.time() + a.seconds
    last = ""
    while time.time() < t_end:
        t0 = time.time()
        g.once()
        s = f"点 {len(w.points)}・箱 {g.n_boxes}・{'止める: ' + fol.hold_reason if fol.hold_reason else '進める'}"
        if s != last:
            print(f"[watch] {time.strftime('%H:%M:%S')} {s}（判定 {time.time() - t0:.3f} 秒）", flush=True)
            last = s
        if len(fol.path) > 2:  # 回り込みを作れた: 仮の経路を元に戻して見続ける
            fol.replace([q2])
        time.sleep(0.1)
    w.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(_watch_main())
