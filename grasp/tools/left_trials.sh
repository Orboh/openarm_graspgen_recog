#!/usr/bin/env bash
# Left-arm aluminium-bar trials (runs ON Thor; 2026-11-22 Sota: 左腕だけ 10 回). Every trial: chest capture -> decide
# (left, tip -5 mm below the floor plane, tilt 20 then 50 deg) -> dry run -> run_grasp --go with the wrist-camera
# centring (--wrist-center), pressing 6 mm while closing (--close-press), put back where it was, hold at home v5.
# Stops on: ~/grasp_runs/STOP, a hit / lost motors / traceback, a failed dry run, 2 no-plans in a row, or $2 grasps.
#   bash grasp/tools/left_trials.sh [N=10] [STOP_AFTER_GRASPS=0 (never)]
cd ~/projects/openarm-grasp
N=${1:-10}; STOP_AFTER=${2:-0}
LOG=~/grasp_runs/left_trials_$(date +%Y%m%d_%H%M%S).csv
echo "n,time,cap,tilt,tcp,result,grip_deg,wrist_centre,start_vs_home_deg,rest_raises" > $LOG; echo "[left] log $LOG"
aws sts get-caller-identity --profile ${GRASP_S3_PROFILE:-kota0} >/dev/null 2>&1 || echo "[s3] WARNING: S3 の鍵が切れている。記録は手元に残し、鍵を入れ直したら上げる（aws sso login --profile ${GRASP_S3_PROFILE:-kota0} --use-device-code）"
set -a; . grasp/config/sites/tokyo-thor-local.env; set +a
export PYTHONPATH=$PWD:$HOME/work/GraspGenX
HOME_YAML=grasp/config/home_left_cart-sym-v5.yaml
Q=$(python3 -c "import yaml;h=yaml.safe_load(open('$HOME_YAML'));print(','.join(str(h[f'joint_{i}']) for i in range(1,8)))")
PA="--bar-grasps --bar-pitch 30 --raise-to-clear 0.015 --ik-seeds 40 --avoid-obstacles --soft-below=-1 --clear-gap --gap-soft-below=-0.01 --open-mm 60 --tip-min -0.005 --floor-clear=-0.012 --max-branch-deg 60 --center-tol 0.15 --min-score 0.3 --max-try 5000"
RG="--home $HOME_YAML --standby grasp/config/standby_left_cart-v1.yaml --transit-dps 50 --near-dps 20 --hold-home --grip-kp 20 --grip-close 0 --max-track-err 20 --close-wait 1.0 --retries 0 --wrist-snap --wrist-center --close-press 0.006 ${GRASP_BLEND---blend}"
np_=0; held=0
for n in $(seq 1 $N); do
  [ -f ~/grasp_runs/STOP ] && { echo "[left] STOP file"; break; }
  timeout 120 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/thor_snap.py --hmax 0.10 --no-view >/dev/null 2>&1
  c=$(basename $(ls -td grasp/runs/cap_2* | head -1)); t=$(date +%H:%M:%S)
  for MT in 20 50; do
    rm -f grasp/runs/$c/plan.json
    timeout 900 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/decide.py grasp/runs/$c --q=$Q --no-vlm --plan-args="$PA --max-tilt $MT" > grasp/runs/$c/decide.log 2>&1
    [ -f grasp/runs/$c/plan.json ] && break
  done
  if [ ! -f grasp/runs/$c/plan.json ]; then
    np_=$((np_+1)); echo "$n,$t,$c,,,noplan,,,," >> $LOG; echo "[left] $n $c no plan ($np_)"
    [ $np_ -ge 2 ] && { echo "[left] 2 no-plans in a row: stop"; break; }; continue
  fi
  np_=0
  info=$(python3 -c "import json;p=json.load(open('grasp/runs/$c/plan.json'));print(round(p['tilt_deg']),'\"'+' '.join(f'{v:.3f}' for v in p['tcp_base'])+'\"',sep=',')")
  if ! timeout 900 ~/projects/lerobot/.venv/bin/python -u grasp/run_grasp.py grasp/runs/$c $RG --start=$Q 2>&1 | grep -q "DRY RUN"; then
    echo "$n,$t,$c,$info,dryfail,,,," >> $LOG; echo "[left] $n dry run failed: stop"; break; fi
  [ -f ~/grasp_runs/STOP ] && { echo "[left] STOP file"; break; }
  timeout 900 ~/projects/lerobot/.venv/bin/python -u grasp/run_grasp.py grasp/runs/$c $RG --go 2>&1 | grep -v -i warn > grasp/runs/$c/run.log
  L=grasp/runs/$c/run.log
  res=$(grep -o "held check 1/1:.*" $L | head -1)
  g=$(echo "$res" | grep -o "gripper [-+0-9.]*" | cut -d' ' -f2); r=$(echo "$res" | grep -o "[A-Z]*$")
  wc=$(grep -o "wrist centre:.*" $L | head -1 | tr ',' ';'); sv=$(grep -o "start vs home max diff [0-9.]*" $L | grep -o "[0-9.]*$"); rr=$(grep -c "raise from rest" $L)
  echo "$n,$t,$c,$info,${r:-?},$g,$wc,$sv,$rr" >> $LOG
  echo "[left] $n $c tilt ${info%%,*} -> ${r:-?} | $wc | start vs home ${sv:-?} deg"
  echo "$(date +%F\ %T) left_trials $c -> ${r:-?} ($wc)" >> ~/rollouts/RUNLOG.md
  (bash grasp/tools/s3_sync.sh $LOG >/dev/null 2>&1 &)  # the CSV after every round (run_grasp.py uploads the capture itself)
  grep -qE "Blocked|Traceback|MotorLost|TrackingLost|モーターの返事|max joint error [0-9.]+° >" $L && { echo "[left] stopped by a hit / lost motors / error: see $L"; break; }  # 22:31: the motor-loss message is Japanese
  [ "$r" = "HELD" ] && held=$((held+1))
  [ $STOP_AFTER -gt 0 ] && [ $held -ge $STOP_AFTER ] && { echo "[left] $held grasps: done"; break; }
done
echo "[left] done: $held held. log $LOG"
