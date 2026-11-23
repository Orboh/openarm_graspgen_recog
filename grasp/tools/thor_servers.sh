#!/bin/bash
# Thor で、識別（:5558）・GraspGenX（:5556）・cuRobo（:5559）を tmux のセッション grasp-servers に常駐させる（2026-11-21）。
# 使い方: bash grasp/tools/thor_servers.sh start|stop|status   ログ: ~/grasp_runs/logs/server_*.log（/tmp は再起動で消えるので使わない）
set -u
R=$HOME/projects/openarm-grasp; L=$HOME/grasp_runs/logs; mkdir -p $L
# パネルとダッシュボードは Tailscale の口だけで開く（同じ LAN の他の機械から腕のボタンを押せないように）
TS=$(tailscale ip -4 2>/dev/null | head -1); TS=${TS:-127.0.0.1}
S=grasp-servers
case ${1:-status} in
start)
  [ -e $HOME/work/GraspGenX/ext/graspgenx_checkpoints ] || { mkdir -p $HOME/work/GraspGenX/ext; ln -sfn $HOME/work/GraspGenX_ext/* $HOME/work/GraspGenX/ext/; }
  tmux has-session -t $S 2>/dev/null && { echo "already running"; exit 0; }
  tmux new -d -s $S -n perceive "cd $R && $HOME/work/perceive/.venv/bin/python grasp/tools/perceive_server.py --port 5558 > $L/server_perceive.log 2>&1"
  tmux new-window -t $S -n graspgenx "cd $HOME/work/GraspGenX && .venv/bin/python client-server/graspgenx_server.py --config ext/graspgenx_checkpoints/release --assets_dir ext/gripper_descriptions/gripper_descriptions/assets --port 5556 > $L/server_graspgenx.log 2>&1"
  tmux new-window -t $S -n curobo "cd $R && set -a && . grasp/config/sites/tokyo-thor-local.env && set +a && PYTHONPATH=$R $HOME/work/curobo/.venv/bin/python grasp/curobo/server.py --port 5559 > $L/server_curobo.log 2>&1"
  # 画像言語モデル（Qwen3-VL-4B・Apache-2.0）: 何から取るかの相談（grasp/tools/decide.py）。読み込み 約 15 秒・1 回 約 9 秒
  tmux new-window -t $S -n vlm "cd $R && $HOME/work/perceive/.venv/bin/python grasp/tools/vlm_server.py --port 5560 > $L/server_vlm.log 2>&1"
  # 2026-11-23 (Sota: 実行の結果は毎回 S3 に): safety net every 10 min, in addition to the upload after every run
  tmux new-window -t $S -n s3sync "cd $R && while true; do bash grasp/tools/s3_sync.sh >> $L/s3_sync_loop.log 2>&1; sleep 600; done"
  # 記録を見るダッシュボード（PC のブラウザから Tailscale で http://<Thor の Tailscale IP>:8765 を開く）
  tmux new-window -t $S -n dashboard "cd $R && python3 grasp/enpire/dashboard.py --host $TS --port 8765 > $L/server_dashboard.log 2>&1"
  # 操作パネル（準備・GO・目視の結果・締め）。PC のブラウザで http://$TS:8770
  tmux new-window -t $S -n panel "cd $R && set -a && . grasp/config/sites/tokyo-thor-local.env && set +a && PYTHONPATH=$R:$HOME/work/GraspGenX $HOME/work/GraspGenX/.venv/bin/python grasp/enpire/panel.py --host $TS --port 8770 > $L/server_panel.log 2>&1"
  echo "started (cuRobo needs about 2 min to warm up). panel: http://$TS:8770  dashboard: http://$TS:8765";;
stop) tmux kill-session -t $S 2>/dev/null; echo stopped;;
status)
  tmux list-windows -t $S 2>/dev/null || echo "not running"
  for p in 5558 5556 5559 5560 8765 8770; do ss -ltn | grep -q ":$p " && echo "port $p listening" || echo "port $p down"; done;;
esac
