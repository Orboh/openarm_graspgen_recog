#!/bin/bash
# PC から Thor の ENPIRE（grasp/enpire/live.py）を打つだけの入口（2026-11-21）。計算・撮影・腕はすべて Thor の中。
# 例: grasp/tools/pc_thor.sh prepare --iter iter_054 --snap-args "--perceive localhost:5558 --band 0.25,0.56,-0.20,0.35 --hmax 0.10"
#     grasp/tools/pc_thor.sh servers start|stop|status
#     grasp/tools/pc_thor.sh pull            # 記録（grasp/runs）を見る用に PC の grasp/runs_thor/ へ引く
set -u
H=${GRASP_THOR:-orbou-thor@orboh-thor}
case ${1:-} in
servers) shift; exec ssh $H "bash ~/projects/openarm-grasp/grasp/tools/thor_servers.sh ${1:-status}";;
pull) exec rsync -a $H:projects/openarm-grasp/grasp/runs/ "$(dirname "$0")/../runs_thor/";;
esac
printf -v ARGS '%q ' "$@"
exec ssh $H "cd ~/projects/openarm-grasp && set -a && . grasp/config/sites/tokyo-thor-local.env && set +a && PYTHONPATH=\$HOME/projects/openarm-grasp:\$HOME/work/GraspGenX \$HOME/work/GraspGenX/.venv/bin/python -m grasp.enpire.live $ARGS"
