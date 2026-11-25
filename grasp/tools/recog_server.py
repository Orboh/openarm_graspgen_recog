#!/usr/bin/env python3
"""切り出し（YOLOE-26）と材質（SigLIP2）を読み込んだまま常駐し、thor_snap.py の main を同じプロセスで回す。腕もカメラも使わない。

2026-11-25・作業者: Shreyan。run_pipeline.sh は回ごとに thor_snap.py を新しいプロセスで起動していて、SigLIP2 の読み込み（約 11 秒）と
YOLOE の読み込み（約 1 秒）を毎回払っていた（分類そのものは 0.1 秒）。ここで 1 度だけ読み込んで温めておく。
対象の選び方は thor_snap.py の main をそのまま呼ぶので変わらない。

切り出しの venv（~/work/recog/.venv）で、置き場の env を読んでから立てる（run_pipeline.sh が立っていなければ立てる）:
    PYTHONPATH=$PWD ~/work/recog/.venv/bin/python grasp/tools/recog_server.py            # tcp://*:5561
要求（json）:
    {"op": "ping"} → {"ok": true}
    {"op": "snap", "argv": [thor_snap.py の引数...], "env": {"GRASP_TIMING_ROUND": "2", ...}}
        → {"rc": 終了コード, "stdout": 標準出力}。env は呼ぶ前に os.environ に入れる（段ごとの時間の記録の回の番号など）
"""
from __future__ import annotations

import argparse
import contextlib
import io
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)


def call_main(mod, argv):
    """mod.main(argv) を同じプロセスで呼び、(終了コード, 標準出力) を返す（resident_plan.py の call_main と同じ）。"""
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
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--bind", default="tcp://*:5561")
    a = ap.parse_args()
    t0 = time.time()
    import numpy as np
    from grasp import classification, segment
    from grasp.tools import thor_snap
    model = segment._yoloe()  # YOLOE を読み込む
    model.predict(source=np.zeros((480, 640, 3), np.uint8), save=False, verbose=False,
                  device=os.environ.get("GRASP_RECOG_DEVICE", "cuda"))  # 1 回目の推論の CUDA の準備を済ませる
    classification.classify_material(np.full((64, 64, 3), 128, np.uint8))  # SigLIP2 を読み込んで温める
    print(f"[recog] YOLOE と SigLIP2 を読み込んで温めた（{time.time() - t0:.1f} 秒）。{a.bind} で待つ", flush=True)

    import zmq
    sock = zmq.Context.instance().socket(zmq.REP)
    sock.bind(a.bind)
    while True:
        req = sock.recv_json()
        try:
            if req.get("op") == "ping":
                rep = {"ok": True}
            elif req.get("op") == "snap":
                for k, v in (req.get("env") or {}).items():
                    os.environ[str(k)] = str(v)
                t = time.time()
                rc, out = call_main(thor_snap, [str(x) for x in req["argv"]])
                rep = {"rc": rc, "stdout": out}
                print(f"[recog] snap rc={rc}（{time.time() - t:.2f} 秒）", flush=True)
            else:
                rep = {"error": f"知らない op: {req.get('op')!r}"}
        except Exception as e:  # noqa: BLE001  1 つの失敗で口を止めない
            rep = {"error": f"{type(e).__name__}: {e}"}
            print(f"[recog] 失敗: {rep['error']}", flush=True)
        sock.send_json(rep)


if __name__ == "__main__":
    raise SystemExit(main())
