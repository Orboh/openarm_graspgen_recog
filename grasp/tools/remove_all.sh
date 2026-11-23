#!/usr/bin/env bash
# Take everything off the work area, one object per round (runs ON Thor; 2026-11-22 Sota: 映ってるやつ 3 つ全部取り除いて).
# Each round: chest capture -> decide (right arm first, then left; tilt 20 then 50 deg; aluminium first, then the
# rigid things touching it) -> plan_drop: carry it aside and open there (right arm: front-right on the carpet,
# left arm: front-left) -> dry run -> run_grasp --go. Left arm on an aluminium bar: wrist-camera centring + 6 mm press.
# Stops when nothing is left to take (2 rounds without a plan), on ~/grasp_runs/STOP, a hit / lost motors / error,
# a failed dry run, or after N rounds.
#   bash grasp/tools/remove_all.sh [N=6]
cd ~/projects/openarm-grasp
N=${1:-6}
LOG=~/grasp_runs/remove_all_$(date +%Y%m%d_%H%M%S).csv
echo "n,time,cap,arm,object,kind,tilt,result,grip_deg,note" > $LOG; echo "[remove] log $LOG"
aws sts get-caller-identity --profile ${GRASP_S3_PROFILE:-kota0} >/dev/null 2>&1 || echo "[s3] WARNING: S3 の鍵が切れている。記録は手元に残し、鍵を入れ直したら上げる（aws sso login --profile ${GRASP_S3_PROFILE:-kota0} --use-device-code）"
PA="--bar-grasps --bar-pitch 30 --raise-to-clear 0.015 --ik-seeds 40 --avoid-obstacles --soft-below=-1 --clear-gap --gap-soft-below=-0.01 --open-mm 60 --tip-min -0.005 --floor-clear=-0.012 --max-branch-deg 60 --center-tol 0.15 --min-score 0.3 --max-try 5000"
np_=0
for n in $(seq 1 $N); do
  [ -f ~/grasp_runs/STOP ] && { echo "[remove] STOP file"; break; }
  (set -a; . grasp/config/sites/tokyo-thor-local.env; set +a; export PYTHONPATH=$PWD:$HOME/work/GraspGenX
   timeout 120 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/thor_snap.py --hmax 0.10 --no-view >/dev/null 2>&1)
  c=$(basename $(ls -td grasp/runs/cap_2* | head -1)); t=$(date +%H:%M:%S); got=""
  for arm in right left; do
    for MT in 20 50; do
      rm -f grasp/runs/$c/plan.json
      ( set -a; . grasp/config/sites/tokyo-thor-local.env; [ $arm = right ] && . grasp/config/sites/tokyo-thor-local-right.env; set +a
        export PYTHONPATH=$PWD:$HOME/work/GraspGenX
        [ $arm = right ] && export PLAN_SEED_Q=49.2,-8.5,-46.0,40.1,19.8,36.6,-58.7
        Q=$(python3 -c "import yaml;h=yaml.safe_load(open('grasp/config/home_${arm}_cart-sym-v5.yaml'));print(','.join(str(h[f'joint_{i}']) for i in range(1,8)))")
        timeout 900 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/decide.py grasp/runs/$c --q=$Q --no-vlm --clear-all --plan-args="$PA --max-tilt $MT" > grasp/runs/$c/decide_$arm.log 2>&1 )
      [ -f grasp/runs/$c/plan.json ] && { got=$arm; break 2; }
    done
  done
  if [ -z "$got" ]; then
    np_=$((np_+1)); echo "$n,$t,$c,,,,,noplan,,$(grep -o '今は取れる物がない.*' grasp/runs/$c/decide_left.log | head -1 | cut -c1-120 | tr ',' ';')" >> $LOG
    echo "[remove] $n $c nothing to take ($np_)"; [ $np_ -ge 2 ] && { echo "[remove] nothing left that can be taken: stop"; break; }; continue
  fi
  np_=0
  obj=$(python3 -c "import json;d=json.load(open('grasp/runs/$c/decision.json'));ch=d.get('chosen') or {};o=[x for x in d['objects'] if x['id']==ch.get('id')];print((o[0]['label'][:30] if o else '?').replace(',',' '),'|',o[0]['kind'] if o else '?')" 2>/dev/null)
  kind=$(echo "$obj" | awk -F'|' '{print $2}' | tr -d ' ')
  tilt=$(python3 -c "import json;print(round(json.load(open('grasp/runs/$c/plan.json'))['tilt_deg']))")
  ( set -a; . grasp/config/sites/tokyo-thor-local.env; [ $got = right ] && . grasp/config/sites/tokyo-thor-local-right.env; set +a
    export PYTHONPATH=$PWD:$HOME/work/GraspGenX
    AT=$([ $got = right ] && echo 0.15,-0.42,0.42 || echo 0.15,0.42,0.42)  # beside the shoulders, outside the decide band (x 0.25.., y -0.20..0.35): 21:40 the bar dropped at 0.30,-0.36 landed at y -0.18 and was picked again
    $HOME/work/GraspGenX/.venv/bin/python grasp/tools/plan_drop.py grasp/runs/$c --at $AT --home grasp/config/home_${got}_cart-sym-v5.yaml > grasp/runs/$c/drop.log 2>&1 )
  if ! grep -q "drop at" grasp/runs/$c/drop.log; then echo "$n,$t,$c,$got,$obj,$tilt,nodrop,,no drop pose" >> $LOG; echo "[remove] $n no drop pose: stop"; break; fi
  EXTRA=""; [ $got = left ] && [ "$kind" = aluminium ] && EXTRA="--wrist-snap --wrist-center --close-press 0.006"
  [ $got = right ] && EXTRA="--held-below -2"
  run() { ( set -a; . grasp/config/sites/tokyo-thor-local.env; [ $got = right ] && . grasp/config/sites/tokyo-thor-local-right.env; set +a
            export PYTHONPATH=$PWD:$HOME/work/GraspGenX
            timeout 900 ~/projects/lerobot/.venv/bin/python -u grasp/run_grasp.py grasp/runs/$c --home grasp/config/home_${got}_cart-sym-v5.yaml \
              --standby grasp/config/standby_${got}_cart-v1.yaml --transit-dps 50 --near-dps 20 --hold-home --grip-kp 20 --grip-close 0 \
              --max-track-err 20 --close-wait 1.0 --retries 0 ${GRASP_BLEND---blend} $EXTRA "$@" 2>&1 | grep -v -i warn ); }
  S=$(python3 -c "import yaml;h=yaml.safe_load(open('grasp/config/home_${got}_cart-sym-v5.yaml'));print(','.join(str(h[f'joint_{i}']) for i in range(1,8)))")
  if ! run --start=$S | grep -q "DRY RUN"; then echo "$n,$t,$c,$got,$obj,$tilt,dryfail,,dry run failed" >> $LOG; echo "[remove] $n dry run failed: stop"; break; fi
  [ -f ~/grasp_runs/STOP ] && { echo "[remove] STOP file"; break; }
  run --go > grasp/runs/$c/run.log
  L=grasp/runs/$c/run.log; res=$(grep -o "held check 1/1:.*" $L | head -1)
  g=$(echo "$res" | grep -o "gripper [-+0-9.]*" | cut -d' ' -f2); r=$(echo "$res" | grep -o "[A-Z]*$")
  echo "$n,$t,$c,$got,$obj,$tilt,${r:-?},$g,$(grep -o 'wrist centre:.*' $L | head -1 | tr ',' ';')" >> $LOG
  echo "[remove] $n $c $got arm: $obj tilt $tilt -> ${r:-?}"
  echo "$(date +%F\ %T) remove_all $c $got $obj -> ${r:-?}" >> ~/rollouts/RUNLOG.md
  (bash grasp/tools/s3_sync.sh $LOG >/dev/null 2>&1 &)  # the CSV after every round (run_grasp.py uploads the capture itself)
  grep -qE "Blocked|Traceback|MotorLost|TrackingLost|モーターの返事|max joint error [0-9.]+° >" $L && { echo "[remove] stopped by a hit / lost motors / error: see $L"; break; }  # 22:31: the motor-loss message is Japanese; the loop went on
done
echo "[remove] done. log $LOG"
