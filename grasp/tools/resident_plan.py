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
    # --place-on tape を足すと、同じ 1 枚からテープの輪の中心も出し、plan_place.py --at で置き先まで解く（<回>/place_target.json）
    # 速い設定（2026-11-23・撮ってから置き先まで約 1 秒）: --plan-args に "--rounds 1 --per-call 600" を入れ、
    # cuRobo のサーバーを GRASP_CUROBO_IK_ITERS=20・GRASP_CUROBO_IK_SEEDS=16（置き場の env）で立てる
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
        import threading
        self.lock = threading.Lock()  # 撮影（grab）と録画のスレッドが同じ pipeline を読むので、1 度に 1 つだけ
        self.rec = None

    def start_record(self, path: str, fps: int = 15):
        """胸のカメラの色を path（mp4）に録り始める（2026-11-23 Shreyan: 選別の回を丸ごと録る）。撮影の間だけ途切れる。"""
        import threading
        import cv2
        self.stop_record()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (640, 480))
        st = {"path": path, "stop": False, "n": 0, "t0": time.time()}

        def loop():
            k = 0
            while not st["stop"]:
                with self.lock:
                    fs = self.pipe.wait_for_frames()
                k += 1
                if k % max(1, round(30 / fps)) == 0:
                    vw.write(np.asanyarray(fs.get_color_frame().get_data()))
                    st["n"] += 1
            vw.release()

        st["thread"] = threading.Thread(target=loop, daemon=True)
        st["thread"].start()
        self.rec = st
        return {"recording": path}

    def stop_record(self):
        """録画を止めて mp4 を閉じる。録っていなければ何もしない。"""
        st, self.rec = self.rec, None
        if not st:
            return {"recording": None}
        st["stop"] = True
        st["thread"].join(timeout=5)
        return {"recorded": st["path"], "frames": st["n"], "seconds": round(time.time() - st["t0"], 1)}

    def grab(self, frames: int = 4):
        """色 (H,W,3)・深度 [m]・K・内部パラメータの辞書。時間方向のフィルタは毎回作り直す（前の撮影を混ぜない）。"""
        rs = self.rs
        to_disp, to_depth = rs.disparity_transform(True), rs.disparity_transform(False)
        spatial, temporal, holes = rs.spatial_filter(), rs.temporal_filter(), rs.hole_filling_filter(1)
        with self.lock:
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
        self.stop_record()
        self.pipe.stop()


def surface_spots(P: np.ndarray, near_xy, inset: float = 0.04, step: float = 0.03) -> list:
    """置き場の上面の点 P（付け根の座標・(N,3)）から、置き先の候補 (x, y, z) を、near_xy（拾う物）に近い順に返す。
    上面の向き（主成分）に沿った長方形の、縁から inset 内側を step ごとに並べ、すぐ近く（1.5 cm）に上面の点がある所だけ残す。
    z は上面の高さ（中央値）。2026-11-23: 青い重量計の上に置く。"""
    P = np.asarray(P, float)
    c = P[:, :2].mean(0)
    _, _, Vt = np.linalg.svd(P[:, :2] - c, full_matrices=False)
    L = (P[:, :2] - c) @ Vt.T
    lo, hi = np.percentile(L, 2, axis=0) + inset, np.percentile(L, 98, axis=0) - inset
    if np.any(hi < lo):
        return []
    g0 = np.arange(lo[0], hi[0] + 1e-9, step); g1 = np.arange(lo[1], hi[1] + 1e-9, step)
    G = np.array([(a, b) for a in g0 for b in g1]) @ Vt + c
    sub = P[:: max(1, len(P) // 1500), :2]
    keep = np.min(np.linalg.norm(G[:, None, :] - sub[None, :, :], axis=2), axis=1) <= 0.015
    G = G[keep]
    G = G[np.argsort(np.hypot(G[:, 0] - near_xy[0], G[:, 1] - near_xy[1]), kind="stable")]
    z = float(np.median(P[:, 2]))
    return [(float(x), float(y), z) for x, y in G]


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


def serve(addr, snap, process, cam=None):
    """REP の口で頼みごとを 1 つずつ受ける（json）。
    {"op": "capture"} → 撮って保存 → {"cap": <回のフォルダ>, "name", "times"}
    {"op": "plan", "name": <回の名前>, "q": "j1,...,j7"} → 切り出し・計画・置き先 → {"cap", "planned", "place", "summary", "times", "total_s"}
    {"op": "record_start", "path": <mp4>, "fps": 15} → 胸のカメラを録り始める・{"op": "record_stop"} → 止めて閉じる
    {"op": "ping"} → {"ok": true}。失敗は {"error": "..."}（腕を動かす側は計画が立たなかったとみなす）"""
    import zmq
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(addr)
    print(f"[resident] {addr} で待つ（capture・plan・ping）", flush=True)
    k = 0
    while True:
        req = sock.recv_json()
        try:
            op = req.get("op")
            if op == "ping":
                rep = {"ok": True}
            elif op == "capture":
                t0 = time.time()
                name, tt = snap()
                rep = {"cap": os.path.join(RUNS, name), "name": name, "times": tt, "t0": t0}
                print(f"[resident] capture {name}（{time.time() - t0:.2f} 秒）", flush=True)
            elif op == "record_start" and cam is not None:
                rep = cam.start_record(req["path"], int(req.get("fps", 15)))
                print(f"[resident] 録画を始めた: {req['path']}", flush=True)
            elif op == "record_stop" and cam is not None:
                rep = cam.stop_record()
                print(f"[resident] 録画を止めた: {rep}", flush=True)
            elif op == "plan":
                k += 1
                rep = process(k, req["name"], {}, time.time(), q=req.get("q"), q_right=req.get("q_right"))
            else:
                rep = {"error": f"知らない op: {op!r}"}
        except Exception as e:  # noqa: BLE001  計画の失敗で口を止めない
            rep = {"error": f"{type(e).__name__}: {e}"}
            print(f"[resident] 失敗: {rep['error']}", flush=True)
        sock.send_json(rep, default=str)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=None, help="計画の始まりの関節角（lerobot の度・7 つ）。--serve では頼みごとに受け取る")
    ap.add_argument("--serial", default=os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
    ap.add_argument("--frames", type=int, default=4, help="1 回の撮影で重ねる枚数（capture.py の既定は 10・--fast は 4）")
    ap.add_argument("--snap-args", default="--band 0.18,0.45,-0.10,0.45 --zmin 0.22 --zmax 0.70 --hmax 0.15")
    ap.add_argument("--plan-args", default="--max-tilt 45 --rounds 15 --min-score 0.5 --max-try 400 "
                                           "--floor-clear 0.006 --raise-to-clear 0.03 --jaw-clear 0.012")
    ap.add_argument("--repeat", type=int, default=0, help=">0 なら合図なしにこの回数まわして終わる")
    ap.add_argument("--trigger-dir", default=None,
                    help="合図をファイルで受ける（端末の Enter の代わり）。DIR/go＝撮って計画・DIR/snap＝撮るだけ・DIR/quit＝終わる。"
                         "結果は DIR/last.json（撮った回のフォルダ・計画が立ったか・段ごとの時間）")
    ap.add_argument("--place-on", default=None,
                    help="同じ 1 枚から置き先の物（例 tape）も切り出し、輪の中心を当てはめて plan_place.py --at で置き先まで解く"
                         "（2026-11-23 Munenori・テープの輪と木材を別々に撮って起動を 2 回払っていた）。結果は <回>/place_target.json")
    ap.add_argument("--place-to", default=None,
                    help="置き先を、拾う物の今の位置からのずらし量 dx,dy [m]（腕の付け根の座標・右は y が負）で決める。"
                         "\"0,-0.04;0,-0.03;0,-0.05\" のように ; で並べると順に試す（2026-11-23: 木材を右へ 3〜5 cm 置き直す）")
    ap.add_argument("--self-filter", action="store_true",
                    help="切り出しの前に両腕の点を消し、消し残した腕のかけらのかたまりを外す（左は計画の --q・右は頼みごとの q_right か "
                         "GRASP_REST_RIGHT・2026-11-23: --target any で自分の爪を拾わない）")
    ap.add_argument("--place-on-surface", default=None,
                    help="置き場の物の名前（例 blue_scale）。その色の上面を見つけ、縁から 4 cm 内側を拾う物に近い順に置き先として試す"
                         "（置く高さは上面に合わせる・2026-11-23）")
    ap.add_argument("--place-by-material", action="store_true",
                    help="置き先を、thor_snap.py --yoloe が <回>/pick.json に書いた材質から plan_place.py --by-material で決める"
                         "（grasp/classification.py の ROUTE_OFFSET・--snap-args に --yoloe が要る・2026-11-23 Shreyan）")
    ap.add_argument("--place-args", default="", help="--place-on・--place-to のとき plan_place.py に足す引数")
    ap.add_argument("--retry-plan-args", default="--rounds 3",
                    help="計画が立たなかったとき、GraspGenX を出し直して 1 度だけ計画し直す。そのとき --plan-args の後ろに足す引数"
                         "（2026-11-23: --rounds 1 --per-call 600 で 0.4 秒にしたので、候補が足りないときの保険。空なら計画し直さない）")
    ap.add_argument("--serve", default=None,
                    help="ZMQ の口（例 tcp://*:5560）で「撮る」と「計画する」を別々に受ける（grasp/tools/grasp_daemon.py が使う・2026-11-23）。"
                         "撮った直後に腕を上げ始め、その間に計画するため")
    a = ap.parse_args()
    if a.place_by_material and "--yoloe" not in a.snap_args.split():
        ap.error("--place-by-material には --snap-args の --yoloe が要る（材質は pick.json から読む）")
    if not a.serve and not a.q:
        ap.error("--q が要る（--serve のときだけ省ける）")

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
        return process(k, name, tt, t0, extra_snap)

    def process(k, name, tt, t0, extra_snap="", q=None, q_right=None):
        """撮った回を切り出し → plan.py →（--place-on なら）置き先。q は計画の始まりの関節角の文字列（無ければ --q）。
        q_right は右腕の今の角度の文字列（無ければ GRASP_REST_RIGHT）。"""
        q = q or a.q
        q_right = q_right or os.environ.get("GRASP_REST_RIGHT")
        if a.self_filter:
            extra_snap = " ".join([extra_snap, f"--self-q={q}", "--self-r 0.03", "--self-drop 0.06"]
                                  + ([f"--self-q-right={q_right}"] if q_right else []))
        d = os.path.join(RUNS, name)
        t = time.time()
        rc, out = call_main(thor_snap, ["--cap", name, "--no-view", *a.snap_args.split(), *extra_snap.split()])
        tt["切り出し"] = time.time() - t
        if rc or not os.path.exists(os.path.join(d, "object.npy")):
            print(f"[resident] #{k} 切り出しに失敗: {out.strip()[-300:]}", flush=True)
            return {"cap": d, "planned": False, "error": out.strip()[-300:], "times": tt}
        t = time.time()
        qr = [f"--q-right={q_right}"] if q_right else []
        rc, out = call_main(plan, [d, f"--q={q}", *qr, "--reinfer", "--no-render", *a.plan_args.split()])
        tp = time.time() - t
        inf = [float(x) for x in re.findall(r"\[infer\] \d+ grasps in ([\d.]+)s", out)]
        tt["GraspGenX"] = sum(inf)
        tt["絞り込みと IK"] = tp - sum(inf)
        chosen = [l.strip() for l in out.splitlines() if "chosen" in l or "no candidate" in l]
        planned = os.path.exists(os.path.join(d, "plan.json")) and any("chosen" in c for c in chosen)
        if not planned and a.retry_plan_args.strip():
            t = time.time()
            rc, out2 = call_main(plan, [d, f"--q={q}", *qr, "--reinfer", "--no-render", *a.plan_args.split(), *a.retry_plan_args.split()])
            inf2 = [float(x) for x in re.findall(r"\[infer\] \d+ grasps in ([\d.]+)s", out2)]
            tt["計画し直し"] = time.time() - t
            inf += inf2
            chosen = [l.strip() for l in out2.splitlines() if "chosen" in l or "no candidate" in l]
            planned = os.path.exists(os.path.join(d, "plan.json")) and any("chosen" in c for c in chosen)
        place = None
        if a.place_on and planned:
            t = time.time()
            place = place_step(name, extra_snap)
            tt["置き先"] = time.time() - t
        elif a.place_on_surface and planned:
            t = time.time()
            place = surface_step(name)
            tt["置き先"] = time.time() - t
        elif a.place_to and planned:
            t = time.time()
            place = shift_step(name)
            tt["置き先"] = time.time() - t
        elif a.place_by_material and planned:
            t = time.time()
            place = material_step(name)
            tt["置き先"] = time.time() - t
        total = time.time() - t0
        print(f"[resident] #{k} {name}: " + "・".join(f"{kk} {v:.2f}" for kk, v in tt.items())
              + f" 秒 → 合計 {total:.2f} 秒（GraspGenX {len(inf)} 回）| {chosen[-1] if chosen else out.strip()[-200:]}", flush=True)
        if place is not None:
            print(f"[resident] 置き先: {place['summary']}", flush=True)
        return {"cap": d, "planned": planned, "place": place,
                "summary": chosen[-1] if chosen else out.strip()[-300:], "times": tt, "total_s": total}

    def surface_step(name, max_try: int = 15):
        """--place-on-surface: 置き場の色の上面を見つけ、置き先の候補を順に plan_place.py --at --surface-z で解く。"""
        import cv2
        from grasp import objects as OBJ
        from grasp.capture import depth_to_points  # noqa: F401  （同じ venv にあることの確かめ）
        from grasp.tools import plan_place
        d = os.path.join(RUNS, name)
        img = cv2.imread(os.path.join(d, "color.png"))
        dep = np.load(os.path.join(d, "depth.npy"))
        K = np.array(json.load(open(os.path.join(d, "intrinsics.json")))["K"])
        m = OBJ.color_mask(img, OBJ.load(a.place_on_surface)) & (dep > 0)
        vs, us = np.nonzero(m)
        res = {"ok": False, "surface": a.place_on_surface}
        if len(vs) < 300:
            res["summary"] = f"{a.place_on_surface} が見えない（色の画素 {len(vs)}）"
        else:
            z = dep[vs, us]
            Pc = np.c_[(us - K[0, 2]) * z / K[0, 0], (vs - K[1, 2]) * z / K[1, 1], z]
            Tbc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
            P = Pc @ Tbc[:3, :3].T + Tbc[:3, 3]
            ob = np.load(os.path.join(d, "object.npy")) @ Tbc[:3, :3].T + Tbc[:3, 3]
            spots = surface_spots(P, ob[:, :2].mean(0))
            res["surface_box"] = [round(float(v), 3) for v in (np.percentile(P[:, 0], 2), np.percentile(P[:, 0], 98),
                                                                np.percentile(P[:, 1], 2), np.percentile(P[:, 1], 98),
                                                                np.median(P[:, 2]))]  # run_grasp.py が戻りで障害物にする
            tried = []
            for x, y, zt in spots[:max_try]:
                rc, out = call_main(plan_place, [d, "--at", f"{x:.3f},{y:.3f}", "--surface-z", f"{zt:.3f}", "--no-full-retry",
                                                 *a.place_args.split()])
                line = [l.strip() for l in out.splitlines() if l.startswith("[place]")]
                tried.append(f"({x:.3f},{y:.3f}): {line[-1][:60] if line else ''}")
                if rc == 0:
                    res.update(ok=True, at=[round(x, 3), round(y, 3)], surface_z=round(zt, 3))
                    break
            res["summary"] = (f"{a.place_on_surface} の上面 z {np.median(P[:, 2]):.3f}・候補 {len(spots)} 個 → "
                              + (f"({res['at'][0]:.3f},{res['at'][1]:.3f}) に置く（{len(tried)} 個目）" if res["ok"]
                                 else f"{len(tried)} 個とも解けない: " + " / ".join(tried[:3])))
        with open(os.path.join(d, "place_target.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return res

    def material_step(name):
        """--place-by-material: pick.json の材質から plan_place.py --by-material で置き先を解く。結果は <回>/place_target.json。"""
        from grasp.tools import plan_place
        d = os.path.join(RUNS, name)
        rc, out = call_main(plan_place, [d, "--by-material", *a.place_args.split()])
        line = [l.strip() for l in out.splitlines() if l.startswith("[place]")]
        mat = None
        if os.path.exists(os.path.join(d, "pick.json")):
            mat = json.load(open(os.path.join(d, "pick.json")))["chosen"].get("material")
        res = {"ok": rc == 0, "material": mat, "summary": " / ".join(line[-2:]) if line else out.strip()[-200:]}
        with open(os.path.join(d, "place_target.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return res

    def shift_step(name):
        """--place-to のずらし量を順に試し、最初に解けたものを <回>/plan.json に足す。結果は <回>/place_target.json。"""
        from grasp.tools import plan_place
        d = os.path.join(RUNS, name)
        tried = []
        for to in [v.strip() for v in a.place_to.split(";") if v.strip()]:
            rc, out = call_main(plan_place, [d, "--to", to, *a.place_args.split()])
            line = [l.strip() for l in out.splitlines() if l.startswith("[place]")]
            tried.append(f"{to}: {line[-1] if line else out.strip()[-120:]}")
            if rc == 0:
                break
        res = {"ok": rc == 0, "to": to if rc == 0 else None, "summary": " / ".join(tried)}
        with open(os.path.join(d, "place_target.json"), "w") as f:
            json.dump(res, f, ensure_ascii=False, indent=2)
        return res

    def place_step(name, extra_snap=""):
        """同じ撮影の写し（<回>_place）から --place-on の物を切り出し、輪の中心を当てはめて、<回>/plan.json に置き先を足す。"""
        from scipy.optimize import least_squares
        from grasp.tools import plan_place
        d, dp = os.path.join(RUNS, name), os.path.join(RUNS, name + "_place")
        os.makedirs(dp, exist_ok=True)
        for f in ("color.png", "depth.npy", "intrinsics.json"):  # 切り出しは object.npy などを書くので、拾う物の回とは別のフォルダで
            if not os.path.exists(os.path.join(dp, f)):
                os.link(os.path.join(d, f), os.path.join(dp, f))
        rc, out = call_main(thor_snap, ["--cap", name + "_place", "--no-view", *a.snap_args.split(), *extra_snap.split(),
                                        "--target", a.place_on])
        of = os.path.join(dp, "object.npy")
        if rc or not os.path.exists(of):
            return {"ok": False, "summary": f"{a.place_on} を切り出せない: {out.strip()[-200:]}"}
        Tbc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
        P = (np.load(of) @ Tbc[:3, :3].T + Tbc[:3, 3])[:, :2]
        c0 = (P.min(0) + P.max(0)) / 2
        fit = least_squares(lambda x: np.hypot(P[:, 0] - x[0], P[:, 1] - x[1]) - x[2], [c0[0], c0[1], 0.05])
        cx, cy, r = (float(v) for v in fit.x)
        rc, out = call_main(plan_place, [d, "--at", f"{cx:.3f},{cy:.3f}", *a.place_args.split()])
        line = [l.strip() for l in out.splitlines() if l.startswith("[place]")]
        res = {"ok": rc == 0, "object": a.place_on, "centre_xy": [round(cx, 3), round(cy, 3)], "r_m": round(r, 3),
               "cap": dp, "summary": f"{a.place_on} の中心 ({cx:.3f}, {cy:.3f})・半径 {r:.3f} m | " + (line[-1] if line else out.strip()[-200:])}
        json.dump(res, open(os.path.join(d, "place_target.json"), "w"), ensure_ascii=False, indent=2)
        return res

    try:
        if a.serve:
            serve(a.serve, snap, process, cam)
        elif a.trigger_dir:
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
