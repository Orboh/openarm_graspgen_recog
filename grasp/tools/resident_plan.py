#!/usr/bin/env python3
"""常駐して、合図ごとに「撮る → 切り出す → GraspGenX → 絞り込みと IK」を回し、段ごとの時間を出す。腕は動かさない。

最初に一度だけ: 読み込み（open3d・pinocchio・GraspGenX の接続の部品＝torch）、胸のカメラの起動と明るさの安定、GraspGenX の温め。
合図ごと: 動き続けているカメラから数枚を重ねて撮る → thor_snap.py と同じ切り出し → plan.py（図は描かない・同じプロセスの中で呼ぶ）。
毎回 Python を起動すると、読み込みだけで GraspGenX の接続の部品 1.4 秒・open3d 0.6 秒・カメラの起動と明るさ待ち 2〜3 秒かかる
（2026-11-21 北九州で測った）。その分を最初に払っておく。作業者 Munenori（Kota の PC の Claude Code から）。

GraspGenX の環境で、置き場の設定を読み込んでから打つ（pyrealsense2 が要る）:
    PYTHONPATH=$PWD:~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/tools/resident_plan.py --q <関節 1〜7> \
        --snap-args "--band 0.18,0.45,-0.10,0.45 --zmin 0.22 --zmax 0.70 --hmax 0.15" \
        --plan-args "--max-tilt 45 --rounds 15 --min-score 0.5 --max-try 400 --floor-clear 0.006 --raise-to-clear 0.03 --jaw-clear 0.012"
    # Enter で 1 回まわす・q で終わる。--repeat N で合図なしに N 回まわして終わる
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
RUNS = os.path.join(REPO, "grasp", "runs")
sys.path.insert(0, REPO)


class Camera:
    """胸の D435i を開いたままにして、合図のたびに数枚を重ねた色と深度を返す（capture.py と同じフィルタの順）。"""

    def __init__(self, serial: str, warmup: int = 30):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        prof = self.pipe.start(cfg)
        self.scale = prof.get_device().first_depth_sensor().get_depth_scale()
        self.align = rs.align(rs.stream.color)
        self.serial = serial
        for _ in range(warmup):  # 自動露出が落ち着くまで捨てる（最初に一度だけ）
            self.pipe.wait_for_frames()

    def grab(self, frames: int = 4):
        """色 (H,W,3)・深度 [m]・K・内部パラメータの辞書。時間方向のフィルタは毎回作り直す（前の撮影を混ぜない）。"""
        rs = self.rs
        to_disp, to_depth = rs.disparity_transform(True), rs.disparity_transform(False)
        spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)
        self.pipe.wait_for_frames()  # 合図の前にたまった古い 1 組を捨てる
        depth_f = color_f = None
        for _ in range(frames):
            fs = self.align.process(self.pipe.wait_for_frames())
            color_f = fs.get_color_frame()
            d = to_disp.process(fs.get_depth_frame())
            depth_f = holes.process(to_depth.process(temporal.process(spatial.process(d))))
        i = color_f.profile.as_video_stream_profile().intrinsics
        K = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]], dtype=np.float64)
        intr = {"width": i.width, "height": i.height, "fx": i.fx, "fy": i.fy, "ppx": i.ppx, "ppy": i.ppy,
                "model": str(i.model), "coeffs": list(i.coeffs), "depth_scale": self.scale, "serial": self.serial}
        return (np.asanyarray(color_f.get_data()).copy(),
                np.asanyarray(depth_f.get_data()).astype(np.float32) * self.scale, K, intr)

    def close(self):
        self.pipe.stop()


def call_main(mod, argv):
    """mod.main(argv) を同じプロセスで呼び、終了コードと標準出力を返す（読み込みを使い回す）。"""
    buf = io.StringIO()
    rc = 0
    with contextlib.redirect_stdout(buf):
        try:
            rc = mod.main(argv) or 0
        except SystemExit as e:
            rc = e.code if isinstance(e.code, int) else 1
            print(e.code)
    return rc, buf.getvalue()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", required=True, help="計画の始まりの関節角（lerobot の度・7 つ）")
    ap.add_argument("--serial", default=os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
    ap.add_argument("--frames", type=int, default=4, help="1 回の撮影で重ねる枚数（capture.py の既定は 10・--fast は 4）")
    ap.add_argument("--snap-args", default="--band 0.18,0.45,-0.10,0.45 --zmin 0.22 --zmax 0.70 --hmax 0.15")
    ap.add_argument("--plan-args", default="--max-tilt 45 --rounds 15 --min-score 0.5 --max-try 400 "
                                           "--floor-clear 0.006 --raise-to-clear 0.03 --jaw-clear 0.012")
    ap.add_argument("--repeat", type=int, default=0, help=">0 なら合図なしにこの回数まわして終わる")
    ap.add_argument("--trigger-dir", default=None,
                    help="合図をファイルで受ける（端末の Enter の代わり）。DIR/go＝撮って計画・DIR/snap＝撮るだけ・DIR/quit＝終わる。"
                         "結果は DIR/last.json（撮った回のフォルダ・計画が立ったか・段ごとの時間）")
    a = ap.parse_args()

    # ---- 最初に一度だけ
    T = {}
    t = time.time()
    import cv2  # noqa: F401
    import open3d  # noqa: F401
    from grasp import capture, plan  # noqa: F401  (pinocchio・IK)
    from grasp.infer import infer_grasps, load_sweep
    from grasp.tools import thor_snap
    import graspgenx.serving  # noqa: F401  （torch・trimesh も入る。1 回目の推論が遅い原因）
    T["読み込み"] = time.time() - t
    t = time.time()
    cam = Camera(a.serial)
    T["カメラの起動と明るさの安定"] = time.time() - t
    t = time.time()
    rng = np.random.default_rng(0)
    dummy = (rng.random((2000, 3)) * [0.05, 0.05, 0.05] + [0, 0, 0.4]).astype(np.float32)  # 5 cm の箱の点群
    infer_grasps(dummy, sweep=load_sweep(plan.SWEEP_JSON))
    T["GraspGenX の温め（1 回推論）"] = time.time() - t
    print("[resident] 準備: " + "・".join(f"{k} {v:.2f} 秒" for k, v in T.items()), flush=True)

    def snap(prefix="cap_resident"):
        """撮って保存する。(回のフォルダ名, 段ごとの時間) を返す。"""
        tt = {}
        t = time.time()
        color, depth, K, intr = cam.grab(a.frames)
        tt["撮る"] = time.time() - t
        name = f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}"
        d = os.path.join(RUNS, name)
        t = time.time()
        os.makedirs(d, exist_ok=True)
        cv2.imwrite(os.path.join(d, "color.png"), color)
        np.save(os.path.join(d, "depth.npy"), depth)
        json.dump({**intr, "K": K.tolist(), "t_unix": time.time()}, open(os.path.join(d, "intrinsics.json"), "w"), indent=2)
        tt["保存"] = time.time() - t
        return name, tt

    def cycle(k, extra_snap=""):
        t0 = time.time()
        name, tt = snap()
        d = os.path.join(RUNS, name)
        t = time.time()
        rc, out = call_main(thor_snap, ["--cap", name, "--no-view", *a.snap_args.split(), *extra_snap.split()])
        tt["切り出し"] = time.time() - t
        if rc or not os.path.exists(os.path.join(d, "object.npy")):
            print(f"[resident] #{k} 切り出しに失敗: {out.strip()[-300:]}", flush=True)
            return {"cap": d, "planned": False, "error": out.strip()[-300:], "times": tt}
        t = time.time()
        rc, out = call_main(plan, [d, f"--q={a.q}", "--reinfer", "--no-render", *a.plan_args.split()])
        tp = time.time() - t
        inf = [float(x) for x in re.findall(r"\[infer\] \d+ grasps in ([\d.]+)s", out)]
        tt["GraspGenX"] = sum(inf)
        tt["絞り込みと IK"] = tp - sum(inf)
        total = time.time() - t0
        chosen = [l.strip() for l in out.splitlines() if "chosen" in l or "no candidate" in l]
        print(f"[resident] #{k} {name}: " + "・".join(f"{kk} {v:.2f}" for kk, v in tt.items())
              + f" 秒 → 合計 {total:.2f} 秒（GraspGenX {len(inf)} 回）| {chosen[-1] if chosen else out.strip()[-200:]}", flush=True)
        return {"cap": d, "planned": os.path.exists(os.path.join(d, "plan.json")) and any("chosen" in c for c in chosen),
                "summary": chosen[-1] if chosen else out.strip()[-300:], "times": tt, "total_s": total}

    try:
        if a.trigger_dir:
            td = os.path.expanduser(a.trigger_dir)
            os.makedirs(td, exist_ok=True)
            for f in ("go", "snap", "quit"):
                if os.path.exists(os.path.join(td, f)):
                    os.remove(os.path.join(td, f))
            print(f"[resident] 合図を待つ: {td}/go（撮って計画）・snap（撮るだけ）・quit（終わる）", flush=True)
            k = 0
            while not os.path.exists(os.path.join(td, "quit")):
                if os.path.exists(os.path.join(td, "go")):
                    extra = open(os.path.join(td, "go")).read().strip()  # 中身は切り出しに足す引数（例 --self-q=<今の腕の角度>）
                    os.remove(os.path.join(td, "go"))
                    k += 1
                    res = {"kind": "plan", "snap_extra": extra, **cycle(k, extra)}
                elif os.path.exists(os.path.join(td, "snap")):
                    os.remove(os.path.join(td, "snap"))
                    name, tt = snap("cap_snap")
                    res = {"kind": "snap", "cap": os.path.join(RUNS, name), "times": tt}
                    print(f"[resident] snap {name}", flush=True)
                else:
                    time.sleep(0.05)
                    continue
                json.dump({**res, "t_unix": time.time()}, open(os.path.join(td, "last.json.tmp"), "w"), ensure_ascii=False, indent=2)
                os.replace(os.path.join(td, "last.json.tmp"), os.path.join(td, "last.json"))
            os.remove(os.path.join(td, "quit"))
        elif a.repeat > 0:
            for k in range(1, a.repeat + 1):
                cycle(k)
        else:
            k = 0
            while True:
                s = input("[resident] Enter で撮って計画する（q で終わる）: ")
                if s.strip().lower() == "q":
                    break
                k += 1
                cycle(k)
    finally:
        cam.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
