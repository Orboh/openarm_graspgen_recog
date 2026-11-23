#!/usr/bin/env bash
# 撮る → YOLOE-26 で全部切り出す → 規則で拾う物を決める → SigLIP2 で材質 → GraspGenX と IK → 材質で置き先 → 実機、を 1 本で通す。
# 2026-11-23・作業者: Shreyan（Kota の PC の Claude Code から）。
#
#   # 撮った回で、腕を動かさずに確かめる（最後は乾式）
#   bash grasp/tools/run_pipeline.sh <cap_dir>
#   # 胸のカメラで撮って、計画まで（腕は動かさない）
#   bash grasp/tools/run_pipeline.sh --live
#   # 撮って実機で掴んで置く。既定で、画面の中の物が無くなるまで（拾える物を全部置くまで）くり返す。--n N で最大 N 回にする
#   # **腕が動く**: 人がロボットの前で非常停止を持っているときだけ
#   bash grasp/tools/run_pipeline.sh --live --go [--n 5]
#   止める: Ctrl-C か touch ~/grasp_runs/STOP（次の回の前に止まる）
#
#   ほかの引数: [--site kitakyushu-collect] [--no-yoloe] [--shared-config] [--q 関節 1〜7]
#       [--snap-args "..."] [--plan-args "..."] [--place-args "..."] [--run-args "..."]
#
# 先に立っているもの: GraspGenX のサーバー（:5556）と cuRobo のサーバー（置き場の env の GRASP_CUROBO）
# 撮るのは、計画の常駐（resident_plan.py --serve・:5560）が立っていればそれに頼む（カメラを持っているため）。無ければ capture.py
# Python は 3 つ: 切り出し（YOLOE・SigLIP2）は ~/work/recog/.venv（RECOG_PY）、計画・置き先は GraspGenX の venv（PLAN_PY）、
#   腕は lerobot の venv（ARM_PY）。GraspGenX の venv に ultralytics を入れると torch が上がって壊れるので分けた
# 置き場の env は ~/projects/openarm-grasp の設定を指すので、既定ではこの checkout の grasp/config に読み替える（--shared-config で読み替えない）
set -euo pipefail
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${RECOG_PY:-$HOME/work/recog/.venv/bin/python}
GGX=${GRASPGENX_DIR:-$HOME/work/GraspGenX}
PLAN_PY=${PLAN_PY:-$GGX/.venv/bin/python}
ARM_PY=${ARM_PY:-$HOME/projects/lerobot/.venv/bin/python}
RESIDENT=${GRASP_RESIDENT:-tcp://localhost:5560}
SITE=kitakyushu-collect; YOLOE=1; SHARED=0; Q=""; LIVE=0; GO=0; N=0; RECORD=0
# --isolation-gap: 開いた爪は物の両側へ約 2.6 cm 出るので、孤立は 5 cm 以上（3 cm だと隣と 3 cm のバーを先に選び、掴めなかった）
SNAP_ARGS="--band 0.12,0.50,-0.50,0.50 --zmin 0.22 --zmax 0.70 --hmax 0.15 --isolation-gap 0.05"
# 既定は北九州で動いている常駐（resident_plan.py --serve）の --plan-args と同じ（2026-11-23）
PLAN_ARGS="--obstacle-objects --max-tilt 45 --rounds 1 --per-call 600 --min-score 0.5 --max-try 400 --floor-clear 0.006 --raise-to-clear 0.03 --jaw-clear 0.012 --min-depth 0"
PLACE_ARGS="--yaw-auto --carry-h 0.06"
# 材質ごとに使う腕（置き先の側の腕・2026-11-23: 左腕は中央より右に届かない）。both は左右を計算して点数の良い方
ARM_FOR="${ARM_FOR:-metal:right,wood:left,composite:both}"
# 腕の速さは grasp_daemon.py の北九州の設定（2026-11-23 Munenori が実機で使った値）。run_grasp.py の既定（15・6 deg/s）だと 1 回 約 90 秒
RUN_ARGS="--transit-dps 75 --near-dps 12 --gripper-dps 45 --hold-s 0.5"
CAP=""
while [ $# -gt 0 ]; do
  case $1 in
    --site) SITE=$2; shift 2;;
    --no-yoloe) YOLOE=0; shift;;
    --shared-config) SHARED=1; shift;;
    --q) Q=$2; shift 2;;
    --live) LIVE=1; shift;;
    --go) GO=1; shift;;
    --record) RECORD=1; shift;;
    --n) N=$2; shift 2;;
    --snap-args) SNAP_ARGS=$2; shift 2;;
    --plan-args) PLAN_ARGS=$2; shift 2;;
    --place-args) PLACE_ARGS=$2; shift 2;;
    --run-args) RUN_ARGS=$2; shift 2;;
    -h|--help) sed -n 2,20p "$0"; exit 0;;
    *) CAP=$1; shift;;
  esac
done
[ $LIVE = 1 ] || [ -n "$CAP" ] || { echo "[pipeline] 使い方: run_pipeline.sh <cap_dir> か --live [--go]" >&2; exit 2; }
[ $GO = 0 ] || [ $LIVE = 1 ] || { echo "[pipeline] --go は --live と一緒に使う（古い写真で腕を動かさない）" >&2; exit 2; }
[ -x "$PY" ] || { echo "[pipeline] $PY が無い（~/work/recog/.venv を作る）" >&2; exit 2; }
# N=0 は「物が無くなるまで」。止まらない場合の上限は MAX_ROUNDS（既定 30）。撮った回を 1 つ渡したときは 1 回だけ
if [ $LIVE = 0 ] || [ $GO = 0 ]; then N=1; elif [ "$N" = 0 ]; then N=${MAX_ROUNDS:-30}; fi

set -a; . "$REPO/grasp/config/sites/$SITE.env"; set +a
if [ $SHARED = 0 ]; then  # ~/projects/openarm-grasp を指す値を、この checkout に読み替える
  for v in $(env | grep -oE '^GRASP_[A-Z_]+' ); do
    val=${!v}; [[ $val == "$HOME/projects/openarm-grasp"* ]] && export "$v=${val/#$HOME\/projects\/openarm-grasp/$REPO}"
  done
fi
export PYTHONPATH=$REPO:$GGX
Q=${Q:-$GRASP_REST}
cd "$REPO"
mkdir -p grasp/runs ~/grasp_runs
rm -f ~/grasp_runs/STOP
LOG=grasp/runs/pipeline_$(date +%Y%m%d_%H%M%S).csv
echo "n,cap,material,conf,isolated,plan,place,result,seconds" > "$LOG"

el() { printf '%.1f' "$(echo "$(date +%s.%N) - $1" | bc)"; }
step() { echo "=== [pipeline] $1"; }

capture() {  # 撮って、回のフォルダ（絶対パス）を出す
  local cap
  cap=$($PY - "$RESIDENT" <<'EOF' 2>/dev/null || true
import sys, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 20000); s.setsockopt(zmq.LINGER, 0)
s.connect(sys.argv[1]); s.send_json({"op": "capture"}); r = s.recv_json()
print(r["cap"])
EOF
)
  if [ -z "$cap" ]; then  # 常駐が無い: このプロセスでカメラを開いて撮る
    cap=$REPO/grasp/runs/cap_$(date +%Y%m%d_%H%M%S)
    $GRASP_CAMERA_PY grasp/capture.py --out "$cap" --warmup 8 --temporal 4 >/dev/null
  fi
  echo "$cap"
}

skip_here() {  # 掴めない・置けない物は覚えて、次の回からは次の物を選ぶ（2026-11-23: 届かない物で止まり、残りの物に進まなかった）
  PLACED="${PLACED:+$PLACED;}$($PY -c "import json,sys;c=json.load(open(sys.argv[1]))['chosen']['centroid'];print('%.3f,%.3f,0.03'%tuple(c))" "$1/pick.json")"
  echo "[pipeline] この物は次の回から選ばない"
}

one() {  # 1 回。0 = 置けた（か乾式が通った）・1 = 拾う物が無い・2 = 計画か置き先が立たない・3 = 実機で失敗
  local k=$1 src=$2 t0 NAME D mat="" conf="" iso="" res
  t0=$(date +%s.%N)
  NAME=$(basename "$src")
  if [ "$(cd "$(dirname "$src")" && pwd)" != "$REPO/grasp/runs" ]; then  # thor_snap.py --cap は grasp/runs/<名前> しか読まない
    [ -e "$REPO/grasp/runs/$NAME" ] || cp -r "$src" "$REPO/grasp/runs/$NAME"
  fi
  D=$REPO/grasp/runs/$NAME
  rm -f "$D"/{object.npy,pick.json,mask.png,plan.json,plan_left.json,plan_right.json,place_target.json,grasps_cam.npz}

  step "#$k 1/4 切り出し・拾う物・材質（$NAME）"
  $PY grasp/tools/thor_snap.py --cap "$NAME" --no-view $SNAP_ARGS $([ $YOLOE = 1 ] && echo --yoloe) \
      --self-q="$Q" --self-r 0.03 --self-drop 0.04 ${GRASP_REST_RIGHT:+--self-q-right=$GRASP_REST_RIGHT} \
      ${PLACED:+--exclude-xy=$PLACED} --exclude-r 0.10 2>&1 \
      | grep -E "^\[snap\]|$NAME \|" || true
  if [ ! -f "$D/object.npy" ]; then echo "$k,$NAME,,,,,,nothing,$(el $t0)" >> "$LOG"; return 1; fi
  if [ $YOLOE = 1 ]; then
    read -r mat conf iso < <($PY -c "import json,sys;c=json.load(open(sys.argv[1]))['chosen'];print(c['material'],round(c['mat_conf'],2),c['isolated'])" "$D/pick.json")
    echo "[pipeline] 材質 $mat ($conf)・$([ "$iso" = True ] && echo 孤立 || echo 積みの上)"
  fi

  local arm=left
  if [ $YOLOE = 1 ]; then arm=$(echo "$ARM_FOR" | tr ',' '\n' | awk -F: -v m="$mat" '$1==m{print $2}'); arm=${arm:-both}; fi
  step "#$k 2/4 掴み方と IK（腕: $arm）"
  # GraspGenX は毎回違う候補を出す。細い物では上からの候補が 600 個中 1〜4 個しか残らず、1 回では立たないことが多い
  # （2026-11-23 のバーで 4 回中 1 回）。常駐の --retry-plan-args と同じく、立つまで PLAN_TRIES 回まで出し直す
  # 決めた腕で 3 回立たなければ、残り（PLAN_TRIES まで）は両腕で計算して良い方にする（置き先の向きは材質のまま・届く距離まで縮める）
  local tries=${PLAN_TRIES:-5} a2=$arm
  for i in $(seq 1 "$tries"); do
    [ "$i" -gt 3 ] && [ "$arm" != both ] && a2=both
    $PLAN_PY grasp/plan.py "$D" --q="$Q" --arm "$a2" --reinfer --no-render $PLAN_ARGS 2>&1 | grep -E "chosen|Error" || true
    [ -f "$D/plan.json" ] && break
    echo "[pipeline] 計画 $i 回目（腕: $a2）: 立たなかった"
  done
  if [ ! -f "$D/plan.json" ]; then
    echo "$k,$NAME,$mat,$conf,$iso,fail,,noplan,$(el $t0)" >> "$LOG"
    skip_here "$D"; return 2
  fi

  step "#$k 3/4 置き先"
  if [ $YOLOE = 1 ]; then
    # 材質の向き（grasp/classification.py の ROUTE_OFFSET）は変えずに、届かなければ距離を 100 → 75 → 50 % と縮めて解き直す
    # （2026-11-23: 同じ「前へ 15 cm」でも、物が手前なら届き、奥なら届かなかった）
    for f in 1.0 0.75 0.5; do
      to=$($PY -c "import sys;from grasp.classification import ROUTE_OFFSET as R;dx,dy=R.get(sys.argv[1],R['composite']);f=float(sys.argv[2]);print(f'{dx*f:.3f},{dy*f:.3f}')" "$mat" "$f")
      $PLAN_PY grasp/tools/plan_place.py "$D" --to "$to" --no-full-retry $PLACE_ARGS 2>&1 | grep "^\[place\]" | grep -E "from \[|no reachable" | tail -1 || true
      grep -q q_place_deg "$D/plan.json" && { echo "[pipeline] 置き先: $mat → $to（ROUTE_OFFSET の $f 倍）"; break; }
    done
  else
    $PLAN_PY grasp/tools/plan_place.py "$D" $PLACE_ARGS 2>&1 | grep "^\[place\]" | tail -1 || true
  fi
  if ! grep -q q_place_deg "$D/plan.json"; then
    echo "$k,$NAME,$mat,$conf,$iso,ok,fail,noplace,$(el $t0)" >> "$LOG"
    skip_here "$D"; return 2
  fi

  if [ $GO = 0 ]; then
    step "#$k 4/4 乾式の確かめ（--go なし・腕は動かない）"
    $ARM_PY grasp/run_grasp.py "$D" --start="$Q" $RUN_ARGS 2>&1 | grep -E "DRY RUN|blocked|Blocked|止め|Stopping|Error" | cut -c1-200 || true
    echo "$k,$NAME,$mat,$conf,$iso,ok,ok,dry,$(el $t0)" >> "$LOG"
    return 0
  fi
  [ -f ~/grasp_runs/STOP ] && { echo "[pipeline] STOP があるので腕を動かさない"; return 3; }
  step "#$k 4/4 実機（腕が動く）"
  # run_grasp.py --go は、腕を動かす前に全区間の当たりと可動域を確かめる（乾式と同じ確かめ）ので、別に乾式を打たない
  # 右腕の cuRobo は回り込み（MotionPlanner）を作らずに立てる（8 GB の GPU に左右の MotionPlanner が載らない・2026-11-23）。
  # 右腕は plan_grasp を使わず、関節の直線で近づく（--legacy-approach。区間ごとに当たりを確かめ、当たれば動く前に止まる）
  local extra=""
  [ "$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('arm','left'))" "$D/plan.json")" = right ] && extra="--legacy-approach"
  if $ARM_PY -u grasp/run_grasp.py "$D" --go $RUN_ARGS $extra > "$D/run.log" 2>&1; then res=placed; else res=error; fi
  grep -E "done \(|back at start|Blocked|Traceback|モーター" "$D/run.log" | grep -v "check ok" | tail -4 || true
  grep -qE "Blocked|Traceback|MotorLost|TrackingLost|モーターの返事" "$D/run.log" && res=error
  echo "$k,$NAME,$mat,$conf,$iso,ok,ok,$res,$(el $t0)" >> "$LOG"
  echo "[pipeline] #$k $res（$(el $t0) 秒）"
  # 置いた所を覚え、次の回からそこにある物を拾わない（置き先は画面の端で、外へは出ないため）
  [ $res = placed ] && PLACED="${PLACED:+$PLACED;}$($PY -c "import json,sys;p=json.load(open(sys.argv[1]));print('%.3f,%.3f'%tuple(p['place_target_xy']))" "$D/plan.json")"
  [ $res = placed ] && return 0 || return 3
}

# 腕の cuRobo のサーバー（左 5558・右 5559）: 立っていなければ立て、回が全部終わったら自分で止める（2026-11-23 Shreyan）
srv() {  # $1 = left|right
  local port=5558 args="" log=$HOME/grasp_runs/logs/server_curobo_$1.log
  [ "$1" = right ] && port=5559 args="--arm right --no-detour"
  ss -ltn | grep -q ":$port " && return 0
  mkdir -p "$HOME/grasp_runs/logs"
  echo "[pipeline] cuRobo（$1）を立てる"
  GRASP_CUROBO_IK_BATCH=8 setsid nohup "$HOME/work/curobo/.venv/bin/python" grasp/curobo/server.py $args > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do ss -ltn | grep -q ":$port " && return 0; sleep 1; done
  echo "[pipeline] cuRobo（$1）が立たない: $log" >&2; return 1
}
stop_srv() {  # 5558・5559 を持っているプロセスだけを止める（名前で探すと自分の行に当たる）
  local p
  for p in $(ss -ltnp 2>/dev/null | grep -E ':(5558|5559) ' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -TERM "$p"; done
  echo "[pipeline] cuRobo のサーバーを止めた"
}
zreq() {  # 計画の常駐（カメラ）に json を 1 つ送り、返事を出す
  $PY - "$RESIDENT" "$1" <<'PYEOF'
import sys, json, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.RCVTIMEO, 20000); s.setsockopt(zmq.LINGER, 0)
s.connect(sys.argv[1]); s.send_json(json.loads(sys.argv[2])); print(json.dumps(s.recv_json(), ensure_ascii=False))
PYEOF
}
VIDEO=""
finish() {
  [ -n "$VIDEO" ] && echo "[pipeline] 録画: $(zreq '{"op":"record_stop"}' 2>/dev/null)"
  [ $GO = 1 ] && stop_srv
  return 0
}
# --record: 胸のカメラ（計画の常駐が持っている）で、回の全部を grasp/runs/pipeline_<日時>.mp4 に録る（2026-11-23 Shreyan）
if [ $RECORD = 1 ]; then
  VIDEO=$REPO/${LOG%.csv}.mp4
  zreq "{\"op\":\"record_start\",\"path\":\"$VIDEO\",\"fps\":15}" >/dev/null && echo "[pipeline] 録画を始めた: $VIDEO" \
    || { echo "[pipeline] 録画を始められない（計画の常駐 $RESIDENT が要る）" >&2; VIDEO=""; }
fi
if [ $GO = 1 ]; then srv left; srv right; fi
trap finish EXIT
T0=$(date +%s.%N); fails=0; PLACED=""
echo "[pipeline] 最大 $N 回・物が無くなる（拾える物が残らない）か、続けて 3 回掴めない／置けないか、STOP で終わる"
for k in $(seq 1 "$N"); do
  [ -f ~/grasp_runs/STOP ] && { echo "[pipeline] STOP"; break; }
  src=$([ $LIVE = 1 ] && capture || echo "$CAP")
  rc=0; one "$k" "$src" || rc=$?
  case $rc in
    1) echo "[pipeline] 拾う物が無い: 終わる"; break;;
    3) echo "[pipeline] 実機で失敗: 止める（$src/run.log）"; break;;
    2) fails=$((fails+1)); [ $fails -ge 3 ] && { echo "[pipeline] 続けて 3 回、掴めないか置けない: 止める"; break; };;
    0) fails=0;;
  esac
done
echo; column -s, -t "$LOG"; echo "[pipeline] 合計 $(el $T0) 秒・記録 $LOG"
