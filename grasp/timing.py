"""段ごとの時間を JSONL に残す（記録だけ・標準ライブラリだけ・どの venv からも読む）。2026-11-25・作業者: Shreyan。

選別の 1 回（撮る → 切り出す → 計画 → 置き先 → 実機）のどこに時間がかかっているかを調べるため。
**流れの動きは何も変えない**: 標準出力・標準エラーには何も出さず、GRASP_TIMING_LOG（ファイルのパス）があるときだけ 1 段 1 行の
JSON を追記する。書けない・値が変でも例外を出さない（記録を捨てるだけ）。

    from grasp import timing
    timing.startup()                       # main の最初: 起動（Python の立ち上げ・import）の時間を残す
    t0 = time.time(); ...; timing.event("snap/yoloe predict", t0, time.time(), masks=9)
    with timing.stage("plan/graspgenx infer"):
        ...

1 行の中身: run（GRASP_TIMING_RUN）・round（GRASP_TIMING_ROUND）・proc（script の名前）・pid・thread・stage・
t_start・t_end（UNIX 秒）・dur_s・そのほか渡した値。まとめは grasp/tools/timing_report.py。
GRASP_T_LAUNCH（呼ぶ側が起動の直前に入れた UNIX 秒）があれば、startup() が「起動から import まで」を残す。
"""
from __future__ import annotations

import atexit
import contextlib
import json
import os
import sys
import threading
import time

T_IMPORT = time.time()  # この module を読んだ時刻
_LOCK = threading.Lock()
_STARTED = False


def event(stage: str, t_start: float, t_end: float, /, **kw) -> None:
    """区間 [t_start, t_end] を 1 行残す。GRASP_TIMING_LOG が無ければ何もしない。どんな失敗も外に出さない。"""
    try:
        path = os.environ.get("GRASP_TIMING_LOG")
        if not path:
            return
        rec = {"run": os.environ.get("GRASP_TIMING_RUN", ""), "round": int(os.environ.get("GRASP_TIMING_ROUND", "0") or 0),
               "proc": os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else "python", "pid": os.getpid(),
               "thread": threading.current_thread().name, "stage": stage,
               "t_start": round(float(t_start), 4), "t_end": round(float(t_end), 4), "dur_s": round(float(t_end) - float(t_start), 4)}
        rec.update({k: v for k, v in kw.items() if v is not None})
        line = (json.dumps(rec, ensure_ascii=False, default=str) + "\n").encode()
        with _LOCK:  # O_APPEND の 1 回の write は、ほかのプロセス（plan.py --arm both の子）の行と混ざらない
            fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, line)
            finally:
                os.close(fd)
    except Exception:  # noqa: BLE001  記録のために流れを止めない
        pass


@contextlib.contextmanager
def stage(stage_name: str, /, **kw):
    """with の中の時間を残す。中の例外はそのまま外へ出す（記録に ok=False を付けるだけ）。"""
    t0 = time.time()
    ok = True
    try:
        yield
    except BaseException:
        ok = False
        raise
    finally:
        event(stage_name, t0, time.time(), **kw, **({} if ok else {"ok": False}))


def startup() -> None:
    """main の最初に 1 度呼ぶ。起動（GRASP_T_LAUNCH から、この module の import まで）と import（そこから今まで）を残し、
    終わるときにプロセス全体（process/total）を残す。"""
    global _STARTED
    try:
        if _STARTED or not os.environ.get("GRASP_TIMING_LOG"):
            return
        _STARTED = True
        now = time.time()
        t0 = T_IMPORT
        launch = os.environ.get("GRASP_T_LAUNCH")
        if launch:
            t0 = float(launch)
            event("process/launch→first import", t0, T_IMPORT)
        event("process/imports", T_IMPORT, now)
        atexit.register(lambda: event("process/total", t0, time.time()))
    except Exception:  # noqa: BLE001
        pass
