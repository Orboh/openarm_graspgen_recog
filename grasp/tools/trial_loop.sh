#!/usr/bin/env bash
# 2026-11-22 Sota: 100 回ほど自動で試す（毎回の確認なし・宮嶋の許可）。撮影 → 計画（右腕→左腕・傾き 20°→50°）→ 空打ち → 本番 → 記録。
# 掴めたら落とさず元の場所に置く（plan_drop を使わない）。止める: ~/grasp_runs/STOP、モーター途絶・当たりで止まった、計画が 5 回続けて立たない。
cd ~/projects/openarm-grasp
N=${1:-100}; LOG=~/grasp_runs/trial_loop_$(date +%Y%m%d_%H%M%S).csv
echo "n,time,cap,arm,tilt,tcp,tips_mm,result,grip_deg,camera,note" > $LOG; echo "[loop] log $LOG"
aws sts get-caller-identity --profile ${GRASP_S3_PROFILE:-kota0} >/dev/null 2>&1 || echo "[s3] WARNING: S3 の鍵が切れている。記録は手元に残し、鍵を入れ直したら上げる（aws sso login --profile ${GRASP_S3_PROFILE:-kota0} --use-device-code）"
PA="--bar-grasps --bar-pitch 30 --raise-to-clear 0.015 --ik-seeds 40 --avoid-obstacles --soft-below=-1 --clear-gap --gap-soft-below=-0.01 --open-mm 60 --tip-min 0.006 --floor-clear=-0.006 --max-branch-deg 60 --center-tol 0.15 --min-score 0.3 --max-try 5000"
noplan=0
for n in $(seq 1 $N); do
  [ -f ~/grasp_runs/STOP ] && { echo "[loop] STOP file"; break; }
  (set -a; . grasp/config/sites/tokyo-thor-local.env; set +a; export PYTHONPATH=$PWD:$HOME/work/GraspGenX; timeout 120 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/thor_snap.py --hmax 0.10 --no-view >/dev/null 2>&1)
  c=$(basename $(ls -td grasp/runs/cap_2* | head -1))
  got=""
  for arm in ${TRIAL_ARMS:-right left}; do  # TRIAL_ARMS=left: left arm only (2026-11-22)
    for MT in 20 50; do
      rm -f grasp/runs/$c/plan.json
      ( set -a; . grasp/config/sites/tokyo-thor-local.env; [ $arm = right ] && . grasp/config/sites/tokyo-thor-local-right.env; set +a
        export PYTHONPATH=$PWD:$HOME/work/GraspGenX
        # plan from the pose the arm really starts at (home v5; it was still the old home here until 11/22 16:40)
        Q=$(python3 -c "import yaml;h=yaml.safe_load(open('grasp/config/home_${arm}_cart-sym-v5.yaml'));print(','.join(str(h[f'joint_{i}']) for i in range(1,8)))")
        [ $arm = right ] && export PLAN_SEED_Q=49.2,-8.5,-46.0,40.1,19.8,36.6,-58.7
        timeout 900 $HOME/work/GraspGenX/.venv/bin/python grasp/tools/decide.py grasp/runs/$c --q=$Q --no-vlm --plan-args="$PA --max-tilt $MT" >/dev/null 2>&1 )
      if [ -f grasp/runs/$c/plan.json ]; then got=$arm; break 2; fi
    done
  done
  t=$(date +%H:%M:%S)
  if [ -z "$got" ]; then noplan=$((noplan+1)); echo "$n,$t,$c,,,,,noplan,,,consecutive $noplan" >> $LOG; echo "[loop] $n no plan ($noplan)"; [ $noplan -ge 5 ] && { echo "[loop] 5 no-plan in a row: stop"; break; }; continue; fi
  noplan=0
  info=$(python3 -c "
import json,numpy as np;p=json.load(open('grasp/runs/$c/plan.json'));G=np.array(p['T_grasp_base']);f=json.load(open('grasp/config/floor_plane.json'));n=np.array(f['n'])
print(round(p['tilt_deg']),'\"'+' '.join(f'{v:.3f}' for v in p['tcp_base'])+'\"','\"'+' '.join(str(round((n@(G[:3,3]+G[:3,2]*0.14+G[:3,0]*s*0.03)+f['d'])*1000)) for s in (-1,1))+'\"',sep=',')")
  HOME_YAML=grasp/config/home_${got}_cart-sym-v5.yaml
  run() { ( set -a; . grasp/config/sites/tokyo-thor-local.env; [ $got = right ] && . grasp/config/sites/tokyo-thor-local-right.env; set +a; export PYTHONPATH=$PWD:$HOME/work/GraspGenX
            timeout 900 ~/projects/lerobot/.venv/bin/python -u grasp/run_grasp.py grasp/runs/$c --home $HOME_YAML --standby grasp/config/standby_${got}_cart-v1.yaml ${GRASP_BLEND---blend} --transit-dps 50 --near-dps 20 --hold-home --grip-kp 20 --grip-close 0 --max-track-err 20 --close-wait 1.0 --retries 0 $([ $got = right ] && echo --held-below -2) "$@" 2>&1 | grep -v -i warn ); }
  if ! run --start=$(python3 -c "import yaml;h=yaml.safe_load(open('$HOME_YAML'));print(','.join(str(h[f'joint_{i}']) for i in range(1,8)))") | grep -q "DRY RUN"; then
    echo "$n,$t,$c,$got,$info,dryfail,,,dry run failed" >> $LOG; echo "[loop] $n dry run failed"; continue; fi
  [ -f ~/grasp_runs/STOP ] && { echo "[loop] STOP file"; break; }
  run --go > grasp/runs/$c/run.log
  res=$(grep -o "held check 1/1: gripper [-+0-9.]* deg ([a-z]* by angle), camera: [a-z/]* -> [A-Z]*" grasp/runs/$c/run.log | head -1)
  g=$(echo "$res" | grep -o "gripper [-+0-9.]*" | cut -d' ' -f2); cam=$(echo "$res" | grep -o "camera: [a-z/]*" | cut -d' ' -f2); r=$(echo "$res" | grep -o "[A-Z]*$")
  echo "$n,$t,$c,$got,$info,${r:-?},$g,$cam," >> $LOG; echo "[loop] $n $got tilt ${info%%,*} -> ${r:-?} ($g deg, cam $cam)"
  (bash grasp/tools/s3_sync.sh $LOG >/dev/null 2>&1 &)  # the CSV after every round (run_grasp.py uploads the capture itself)
  if grep -v "直接は運べない" grasp/runs/$c/run.log | grep -q "モーターの返事\|Blocked\|Stopping\|Traceback"; then echo "[loop] motor/blocked/error in run.log: stop"; grep "モーターの返事\|Blocked\|Stopping\|Error" grasp/runs/$c/run.log | tail -3; break; fi
done
echo "[loop] done"; tail -5 $LOG
