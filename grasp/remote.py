"""腕とカメラを動かす PC（ロボット側）へコマンドを打ち、ファイルをやり取りする。

東京（2026-11-20〜21）は、計算の PC から ssh で Thor に打っていた（既定のまま）。
腕・カメラ・GPU が 1 台にそろう PC（北九州の収集機など）では GRASP_ROBOT_HOST=local にすると、
同じコマンドをこの PC の bash でホームから打ち、scp の代わりに cp で写す。
ロボット側の置き場（~/grasp_work・~/grasp_runs）とコマンドの中身はどちらも同じ。
local では ~/grasp_work をこの repo へのリンクにする（ln -s ~/projects/openarm-grasp ~/grasp_work）。
カメラを開く Python は GRASP_CAMERA_PY（Thor は /usr/bin/python3・収集機は lerobot の venv）。
"""
from __future__ import annotations

import glob
import os
import shlex
import shutil
import subprocess

HOST = os.environ.get("GRASP_ROBOT_HOST", "orbou-thor@orboh-thor")
LOCAL = HOST == "local"
HOME = os.path.expanduser("~")
CAMERA_PY = os.environ.get("GRASP_CAMERA_PY", "/usr/bin/python3")


def env_assign() -> str:
    """この PC で設定した GRASP_*（腕の CAN・較正・カメラ・床の高さなど）を `A=1 B=2 ` の形で返す。

    ssh 先にも、すでに動いている tmux の中にも環境変数は届かないので、ロボット側のコマンドの頭に付けて渡す。
    """
    return "".join(f"{k}={shlex.quote(v)} " for k, v in sorted(os.environ.items())
                   if k.startswith("GRASP_") and k != "GRASP_ROBOT_HOST")


def run(cmd: str, timeout: float | None = 60, connect_timeout: int = 8) -> subprocess.CompletedProcess:
    """ロボット側で cmd を打つ（ssh か、この PC の bash）。標準出力・標準エラーは文字列で返す。"""
    exports = env_assign()
    if exports:
        cmd = f"export {exports.strip()}; {cmd}"
    if LOCAL:
        argv = ["bash", "-c", cmd]
    else:
        argv = ["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={connect_timeout}", HOST, cmd]
    return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, cwd=HOME)


def reachable() -> bool:
    """ロボット側に打てるか（ssh が通るか。local なら常に True）。"""
    return LOCAL or run("true", timeout=15, connect_timeout=6).returncode == 0


def put(src: str, dst_rel: str) -> None:
    """この PC の src を、ロボット側のホームから見た dst_rel（フォルダ）へ写す。"""
    if LOCAL:
        dst = os.path.join(HOME, dst_rel)
        os.makedirs(dst, exist_ok=True)
        if os.path.realpath(os.path.dirname(os.path.abspath(src))) != os.path.realpath(dst):  # リンク先が同じなら写さない
            shutil.copy2(src, dst)
        return
    subprocess.check_call(["scp", "-q", src, f"{HOST}:{dst_rel}"])


def get(src_rel: str, dst: str, check: bool = True) -> None:
    """ロボット側のホームから見た src_rel（ファイル・フォルダ・* を含む型）を、この PC の dst へ写す。"""
    if LOCAL:
        srcs = glob.glob(os.path.join(HOME, src_rel))
        if not srcs and check:
            raise FileNotFoundError(f"[remote] none: ~/{src_rel}")
        os.makedirs(dst, exist_ok=True)
        for s in srcs:
            if os.path.isdir(s):
                shutil.copytree(s, os.path.join(dst, os.path.basename(s)), dirs_exist_ok=True)
            else:
                shutil.copy2(s, dst)
        return
    r = subprocess.run(["scp", "-q", "-r", f"{HOST}:{src_rel}", dst], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise subprocess.CalledProcessError(r.returncode, r.args, r.stdout, r.stderr)
