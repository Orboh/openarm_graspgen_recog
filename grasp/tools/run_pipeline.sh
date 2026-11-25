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
RECOG=${GRASP_RECOG:-tcp://localhost:5561}  # 切り出しと材質の常駐（grasp/tools/recog_server.py・YOLOE と SigLIP2 を読み込んだまま）
SITE=kitakyushu-collect; YOLOE=1; SHARED=0; Q=""; LIVE=0; GO=0; N=0; RECORD=0; DUAL=1; OVERLAP=1; ARM_PIDS=""
NEXT_D=""; NEXT_FRAME=""; LAST_FRAME=""  # 前の回で用意した次の物・次の回で使う撮った回・いちばん新しい 1 枚（2026-11-25 Shreyan）
# --isolation-gap: 開いた爪は物の両側へ約 2.6 cm 出るので、孤立は 5 cm 以上（3 cm だと隣と 3 cm のバーを先に選び、掴めなかった）
SNAP_ARGS="--band 0.12,0.50,-0.50,0.50 --zmin 0.22 --zmax 0.70 --hmax 0.15 --isolation-gap 0.05"
# 既定は北九州で動いている常駐（resident_plan.py --serve）の --plan-args と同じ（2026-11-23）
# --check-descent（2026-11-25 Shreyan）: 差し込み・持ち上げを手先の直線でたどれる掴み方だけ選ぶ。--jaw-clear 0.008（12 mm は絞りでいちばん候補を落とした）・--short-fallback（手前 10 cm が届かない掴み方は 6・5 cm で解き直す）・--per-call 500（2026-11-25 Shreyan）
PLAN_ARGS="--check-descent --obstacle-objects --max-tilt 45 --rounds 1 --per-call 500 --min-score 0.5 --max-try 400 --floor-clear 0.006 --raise-to-clear 0.03 --jaw-clear 0.008 --min-depth 0 --short-fallback 0.06,0.05"
PLACE_ARGS="--yaw-auto --carry-h 0.06"
# どちらの腕で掴むかは arm_for（ロボットの真ん中の面の左右と材質・2026-11-25 Shreyan）。前の材質だけの決め方（ARM_FOR）はやめた
# 腕の速さは grasp_daemon.py の北九州の設定（2026-11-23 Munenori が実機で使った値）。run_grasp.py の既定（15・6 deg/s）だと 1 回 約 90 秒
# 2026-11-25 Shreyan（速くする・段ごとの時間で 1 回 81 秒のうち腕が 52 秒・その半分は 12 deg/s の動き）:
#   物に近づく・持ち上げ・置く・置いた後に上がるは物の近くなので 20 deg/s（README の決まりの上限）。運ぶは 75 deg/s（移動の上限・前は 12）。爪の開閉 120 deg/s（2026-11-25 Shreyan）。
#   持ち上げた後に止めない（--hold-s 0）・閉じた後の待ち 0.2 秒・位置の補正 1 回 0.3 秒・移動の区間の補正は 3° を超えたときだけ・
#   拾う物の箱は YOLOE の mask の点だけで作る（--target-grow 0・色で切り出した点は物の一部のことがあったので 4 cm 広げていたが、mask は物の全体で、
#   広げると隣の物や机の点まで入り、24 cm のバーの箱が 30 × 18 cm になって右腕の近づく区間が止まった・2026-11-25 Shreyan）。
#   爪は raise で止まらずに pregrasp へ動く間に開く。plan_grasp で近づけなければ前の動き（区間ごとに確かめる）で近づく（--approach-fallback）
RUN_ARGS="--transit-dps 75 --near-dps 20 --lift-dps 20 --place-dps 20 --carry-dps 75 --gripper-dps 120 --hold-s 0 --close-wait 0.2 --fix-s 0.3 --transit-tol 3 --open-during-approach --approach-fallback --target-grow 0"
CAP=""
while [ $# -gt 0 ]; do
  case $1 in
    --site) SITE=$2; shift 2;;
    --no-yoloe) YOLOE=0; shift;;
    --no-dual) DUAL=0; shift;;  # 両腕を同時に動かさない（1 つずつ）
    --no-overlap) OVERLAP=0; shift;;  # 腕が動いている間に次の物を用意しない
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
QR=${GRASP_REST_RIGHT:-}  # 右腕の今の角度（--go では常駐の腕から回ごとに読む・2026-11-25 Shreyan）
cd "$REPO"
mkdir -p grasp/runs ~/grasp_runs
rm -f ~/grasp_runs/STOP
LOG=grasp/runs/pipeline_$(date +%Y%m%d_%H%M%S).csv
echo "n,cap,material,conf,isolated,plan,place,result,seconds" > "$LOG"

el() { printf '%.1f' "$(echo "$(date +%s.%N) - $1" | bc)"; }
step() { echo "=== [pipeline] $1"; }
# 段ごとの時間（記録だけ・流れは変えない・2026-11-25 Shreyan）: この script と呼ぶ Python（grasp/timing.py）が同じ JSONL に
# 1 段 1 行で書く。まとめは最後に grasp/tools/timing_report.py。GRASP_T_LAUNCH は Python を起動する直前の時刻（起動と import の時間）
export GRASP_TIMING_LOG=$REPO/${LOG%.csv}.timing.jsonl GRASP_TIMING_RUN=$(basename "${LOG%.csv}") GRASP_TIMING_ROUND=0
now() { date +%s.%N; }
inproc() {  # inproc <口> <op> <script> <引数...>: 常駐に頼み、その標準出力を出す（2026-11-25 Shreyan・毎回の起動と読み込みを払わない）。
  # 常駐が立っていなければ 99 を返す（呼ぶ側が今までどおり新しいプロセスで打つ）。計画が立たないなど、script の終了コードが 0 でなければ 1
  local addr=$1 op=$2 script=$3; shift 3
  $PY - "$addr" "$op" "$script" "$@" <<'PYEOF'
import json, os, sys, zmq
addr, op, script, argv = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
env = {k: v for k, v in os.environ.items() if k.startswith("GRASP_TIMING_") or k == "GRASP_SNAP_DEBUG"}
env["GRASP_PLAN_SEQUENTIAL"] = "1"  # 常駐の中では plan.py --arm both も fork せず左右を順に計算する
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 2000)
s.connect(addr); s.send_json({"op": "ping"})
try:
    s.recv_json()
except zmq.Again:
    sys.exit(99)
s.setsockopt(zmq.RCVTIMEO, 300000)
s.send_json({"op": op, "script": script, "argv": argv, "env": env})
try:
    r = s.recv_json()
except zmq.Again:
    print(f"[pipeline] 常駐（{addr}）の返事が来ない", file=sys.stderr); sys.exit(97)
if "error" in r:
    print(f"[pipeline] 常駐（{addr}）の誤り: {r['error']}"); sys.exit(98)
sys.stdout.write(r.get("stdout", ""))
sys.exit(0 if not r.get("rc") else 1)
PYEOF
}
tev() {  # tev <段の名前> <始まりの時刻> [JSON に足す "key":値,...]: 始まりから今までを 1 行残す。失敗しても流れを止めない
  {
    local t1; t1=$(now)
    printf '{"run":"%s","round":%s,"proc":"run_pipeline.sh","pid":%d,"thread":"bash","stage":"%s","t_start":%s,"t_end":%s,"dur_s":%s%s}\n' \
      "$GRASP_TIMING_RUN" "${GRASP_TIMING_ROUND:-0}" $$ "$1" "$2" "$t1" "$(awk -v a="$t1" -v b="$2" 'BEGIN{printf "%.4f", a-b}')" \
      "${3:+,$3}" >> "$GRASP_TIMING_LOG"
  } 2>/dev/null || true
}

capture() {  # 撮って、回のフォルダ（絶対パス）を出す
  local cap tc; tc=$(now)
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
  tev "1 capture" "$tc"
  echo "$cap"
}

skip_here() {  # 掴めない・置けない物は覚えて、次の回からは次の物を選ぶ（2026-11-23: 届かない物で止まり、残りの物に進まなかった）
  PLACED="${PLACED:+$PLACED;}$($PY -c "import json,sys;c=json.load(open(sys.argv[1]))['chosen']['centroid'];print('%.3f,%.3f,0.03'%tuple(c))" "$1/pick.json")"
  echo "[pipeline] この物は次の回から選ばない"
}

arm_for() {  # arm_for <材質> <物の中心の y [m]>: どちらの腕で掴むか（2026-11-25 Shreyan の決め方）
  # 両肩を結ぶ線に垂直で、ロボットの真ん中を通る面（腕の付け根の座標の y = 0・肩は y = ±0.0935）で分ける。
  # 面より右（y < 0）は右腕、ただし木は左腕（木の置き先は左）。面より左（y >= 0）は左腕、ただし金属は右腕（金属の置き先は右）
  awk -v m="$1" -v y="$2" 'BEGIN{ if (y < 0) print (m == "wood" ? "left" : "right"); else print (m == "metal" ? "right" : "left") }'
}

prep_obj() {  # prep_obj <回> <回のフォルダ> <外す所 "x,y,r;..."> <腕を決めるなら left|right> [<先に見る 1>] [<動いている腕>] [<切り出し済み 1>]:
  # 切り出し・材質・掴み方・置き先。0 = 計画と置き先が立った（P_D・P_NAME・P_MAT・P_CONF・P_ISO・P_T0・P_ARM に入れる）・
  # 1 = 拾う物が無い・2 = 計画か置き先が立たない・4 = 決めた腕に合わない物（両腕を同時に動かすための 2 つ目の物を探すとき）・
  # 5 = 動いている腕の物なので、切り出しと腕を決めるだけで止めた（その腕の cuRobo のサーバーの周りの物を、動いている回の途中で
  #     入れ替えないため。計画はその腕が終わってから「切り出し済み」で呼ぶ）
  # 先に見る（前の物を腕が動かしている間に次の物を用意する・2026-11-25 Shreyan）ときは、立たなくても記録（CSV）と「選ばない」を残さない
  local k=$1 src=$2 excl=$3 force=$4 pre=${5:-} busy=${6:-} done_snap=${7:-} override=${8:-} t0 NAME D mat="" conf="" iso="" cy=0
  t0=$(date +%s.%N)
  NAME=$(basename "$src")
  if [ "$(cd "$(dirname "$src")" && pwd)" != "$REPO/grasp/runs" ]; then  # thor_snap.py --cap は grasp/runs/<名前> しか読まない
    [ -e "$REPO/grasp/runs/$NAME" ] || cp -r "$src" "$REPO/grasp/runs/$NAME"
  fi
  D=$REPO/grasp/runs/$NAME
  local ts
  if [ -z "$done_snap" ]; then
  rm -f "$D"/{object.npy,pick.json,mask.png,plan.json,plan_left.json,plan_right.json,place_target.json,grasps_cam.npz}

  step "#$k 1/4 切り出し・拾う物・材質（$NAME）"
  ts=$(now)
  local ex="${PLACED:+$PLACED}${excl:+${PLACED:+;}$excl}"
  local sa=(--cap "$NAME" --no-view $SNAP_ARGS $([ $YOLOE = 1 ] && echo --yoloe) \
      --self-q="$Q" --self-r 0.03 --self-drop 0.04 ${QR:+--self-q-right=$QR} \
      ${ex:+--exclude-xy=$ex} --exclude-r 0.10)
  { inproc "$RECOG" snap thor_snap "${sa[@]}" || [ $? -ne 99 ] || GRASP_T_LAUNCH=$(now) $PY grasp/tools/thor_snap.py "${sa[@]}"; } 2>&1 \
      | grep -E "^\[snap\]|$NAME \|" || true
  tev "2 segment+pick+material (thor_snap.py)" "$ts"
  if [ ! -f "$D/object.npy" ]; then [ -z "$force$pre" ] && echo "$k,$NAME,,,,,,nothing,$(el $t0)" >> "$LOG"; return 1; fi
  fi
  if [ $YOLOE = 1 ]; then
    read -r mat conf iso cy < <($PY -c "import json,sys;c=json.load(open(sys.argv[1]))['chosen'];print(c['material'],round(c['mat_conf'],2),c['isolated'],round(c['centroid'][1],3))" "$D/pick.json")
    echo "[pipeline] 材質 $mat ($conf)・$([ "$iso" = True ] && echo 孤立 || echo 積みの上)・中心の y $cy m"
  fi

  local arm=left
  if [ $YOLOE = 1 ]; then arm=$(arm_for "$mat" "$cy"); fi
  # 動いている腕の物でも、空いている腕で掴めるなら空いている腕で（両腕を使う・2026-11-25 Shreyan）。立たなければ呼ぶ側が元の腕に戻す
  [ -n "$override" ] && arm=$override
  if [ -n "$force" ] && [ "$arm" != "$force" ]; then
    echo "[pipeline] 2 つ目の物（$mat・y $cy）は $arm 腕の物。$force 腕では掴まない（次の回で選ぶ）"; return 4
  fi
  if [ -n "$busy" ] && [ "$arm" = "$busy" ]; then
    echo "[pipeline] 次の物（$mat・y $cy）は今動いている $busy 腕の物: 切り出しだけ済ませ、計画はその腕が終わってから"
    P_D=$D P_NAME=$NAME P_MAT=$mat P_CONF=$conf P_ISO=$iso P_T0=$t0 P_ARM=$arm
    return 5
  fi
  step "#$k 2/4 掴み方と IK（腕: $arm）"
  # GraspGenX は毎回違う候補を出す。細い物では上からの候補が 600 個中 1〜4 個しか残らず、1 回では立たないことが多い
  # （2026-11-23 のバーで 4 回中 1 回）。常駐の --retry-plan-args と同じく、立つまで PLAN_TRIES 回まで出し直す
  # 決めた腕で 3 回立たなければ、残り（PLAN_TRIES まで）は両腕で計算して良い方にする（同時に動かす 2 つ目の物は腕を変えない）
  local tries=${PLAN_TRIES:-5} a2=$arm
  local tp; tp=$(now)
  for i in $(seq 1 "$tries"); do
    [ "$i" -gt 3 ] && [ -z "$force" ] && [ -z "$busy" ] && [ -z "$override" ] && a2=both  # 動いている腕のサーバーには計画を頼まない
    ts=$(now)
    local pa=("$D" --q="$Q" ${QR:+--q-right=$QR} --arm "$a2" --reinfer --no-render $PLAN_ARGS)
    { inproc "$RESIDENT" run plan "${pa[@]}" || [ $? -ne 99 ] || GRASP_T_LAUNCH=$(now) $PLAN_PY grasp/plan.py "${pa[@]}"; } 2>&1 | grep -E "chosen|Error" || true
    tev "3a plan.py (one try)" "$ts" "\"try\":$i,\"arm\":\"$a2\""
    [ -f "$D/plan.json" ] && break
    echo "[pipeline] 計画 $i 回目（腕: $a2）: 立たなかった"
  done
  tev "3 grasp plan (all tries)" "$tp"
  if [ ! -f "$D/plan.json" ]; then
    [ -z "$force$pre" ] && { echo "$k,$NAME,$mat,$conf,$iso,fail,,noplan,$(el $t0)" >> "$LOG"; skip_here "$D"; }
    return 2
  fi

  step "#$k 3/4 置き先"
  tp=$(now)
  if [ $YOLOE = 1 ]; then
    # 材質の向き（grasp/classification.py の ROUTE_OFFSET）は変えずに、届かなければ距離を 100 → 75 → 50 % と縮めて解き直す
    # （2026-11-23: 同じ「前へ 15 cm」でも、物が手前なら届き、奥なら届かなかった）
    for f in 1.0 0.75 0.5; do
      to=$($PY -c "import sys;from grasp.classification import ROUTE_OFFSET as R;dx,dy=R.get(sys.argv[1],R['composite']);f=float(sys.argv[2]);print(f'{dx*f:.3f},{dy*f:.3f}')" "$mat" "$f")
      ts=$(now)
      local pl=("$D" --to "$to" --no-full-retry $PLACE_ARGS)
      { inproc "$RESIDENT" run place "${pl[@]}" || [ $? -ne 99 ] || GRASP_T_LAUNCH=$(now) $PLAN_PY grasp/tools/plan_place.py "${pl[@]}"; } 2>&1 \
        | grep "^\[place\]" | grep -E "from \[|no reachable" | tail -1 || true
      tev "4a plan_place.py (one try)" "$ts" "\"scale\":$f"
      grep -q q_place_deg "$D/plan.json" && { echo "[pipeline] 置き先: $mat → $to（ROUTE_OFFSET の $f 倍）"; break; }
    done
  else
    $PLAN_PY grasp/tools/plan_place.py "$D" $PLACE_ARGS 2>&1 | grep "^\[place\]" | tail -1 || true
  fi
  tev "4 place plan (all tries)" "$tp"
  if ! grep -q q_place_deg "$D/plan.json"; then
    [ -z "$force$pre" ] && { echo "$k,$NAME,$mat,$conf,$iso,ok,fail,noplace,$(el $t0)" >> "$LOG"; skip_here "$D"; }
    return 2
  fi
  P_D=$D P_NAME=$NAME P_MAT=$mat P_CONF=$conf P_ISO=$iso P_T0=$t0
  P_ARM=$($PY -c "import json,sys;print(json.load(open(sys.argv[1])).get('arm','left'))" "$D/plan.json")
  return 0
}

armrun() {  # armrun <腕> <回のフォルダ> <引数...>: 常駐の腕（grasp/tools/arm_server.py）に 1 回を頼み、出力を <回>/run.log に書く。
  # 常駐が無ければ 99（呼ぶ側が今までどおり run_grasp.py を新しいプロセスで打つ）
  local side=$1 d=$2; shift 2
  local port=5562; [ "$side" = right ] && port=5563
  $PY - "tcp://localhost:$port" "$d" "$@" > "$d/run.log" 2>&1 <<'PYEOF'
import json, os, sys, zmq
addr, argv = sys.argv[1], sys.argv[2:]
env = {k: v for k, v in os.environ.items() if k.startswith("GRASP_TIMING_")}
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 2000)
s.connect(addr); s.send_json({"op": "ping"})
try:
    s.recv_json()
except zmq.Again:
    sys.exit(99)
s.setsockopt(zmq.RCVTIMEO, -1)  # 回が終わるまで待つ（腕が動いている間に返事を待つのをやめない）
s.send_json({"op": "run", "argv": argv, "env": env})
r = s.recv_json()
if "error" in r:
    print(f"[pipeline] 常駐の腕の誤り: {r['error']}"); sys.exit(98)
sys.stdout.write(r.get("stdout", ""))
sys.exit(0 if not r.get("rc") else 1)
PYEOF
}

exec_obj() {  # exec_obj <回> <回のフォルダ> <名前> <材質> <確からしさ> <孤立> <始まりの時刻> <腕>: 実機（--go）か乾式で 1 つ掴んで置く。
  # 結果は <回>/pipeline_result に書く（両腕を同時に動かすとき、裏のプロセスから PLACED を戻せないため）。0 = 置けた・3 = 実機で失敗
  local k=$1 D=$2 NAME=$3 mat=$4 conf=$5 iso=$6 t0=$7 side=$8 res ts
  if [ $GO = 0 ]; then
    step "#$k 4/4 乾式の確かめ（--go なし・腕は動かない）"
    ts=$(now)
    GRASP_T_LAUNCH=$(now) $ARM_PY grasp/run_grasp.py "$D" --start="$Q" $RUN_ARGS 2>&1 | grep -E "DRY RUN|blocked|Blocked|止め|Stopping|Error" | cut -c1-200 || true
    tev "5 run_grasp.py (dry)" "$ts"
    tev "round total" "$t0" "\"result\":\"dry\""
    echo "$k,$NAME,$mat,$conf,$iso,ok,ok,dry,$(el $t0)" >> "$LOG"
    echo "dry" > "$D/pipeline_result"
    return 0
  fi
  [ -f ~/grasp_runs/STOP ] && { echo "[pipeline] STOP があるので腕を動かさない"; echo "stop" > "$D/pipeline_result"; return 3; }
  step "#$k 4/4 実機（$side 腕が動く）"
  # run_grasp.py --go は、腕を動かす前に全区間の当たりと可動域を確かめる（乾式と同じ確かめ）ので、別に乾式を打たない
  # 右腕の cuRobo は回り込み（MotionPlanner）を作らずに立てる（8 GB の GPU に左右の MotionPlanner が載らない・2026-11-23）。
  # 右腕は plan_grasp を使わず、関節の直線で近づく（--legacy-approach。区間ごとに当たりを確かめ、当たれば動く前に止まる）
  local extra=""
  [ "$side" = right ] && extra="--legacy-approach"
  ts=$(now)
  local rc=0
  armrun "$side" "$D" $RUN_ARGS $extra || rc=$?
  if [ $rc = 99 ]; then  # 常駐の腕が無い: 今までどおり（休みの姿勢へ戻って脱力する）
    rc=0; GRASP_T_LAUNCH=$(now) $ARM_PY -u grasp/run_grasp.py "$D" --go $RUN_ARGS $extra > "$D/run.log" 2>&1 || rc=$?
  fi
  [ $rc = 0 ] && res=placed || res=error
  # 動く前の確かめで止まった（腕は動いていない・2026-11-25 Shreyan）: 実機の失敗ではないので、この物を外して次の物へ進む
  [ $res = error ] && grep -q "動く前に止まった（腕は動かしていない）" "$D/run.log" && res=nomove
  tev "5 run_grasp.py --go (arm)" "$ts" "\"arm\":\"$side\""
  grep -E "done \(|back at start|Blocked|Traceback|モーター" "$D/run.log" | grep -v "check ok" | tail -4 | sed "s/^/[$side] /" || true
  [ $res != nomove ] && grep -qE "Blocked|Traceback|MotorLost|TrackingLost|モーターの返事" "$D/run.log" && res=error
  # 掴めたか（2026-11-25 Shreyan）: 持ち上げた所の爪の角度が -8° より閉じていなければ（何も挟まずに閉じた）、置けたとは数えない。
  # 18:32 の回は空で閉じて（-4.5°）「置けた」と数え、同じ木を 3 回取りにいった（ENPIRE の live.py の HELD_MAX_DEG と同じ線）
  if [ $res = placed ] && [ -f "$D/traj.csv" ]; then
    local gl; gl=$(awk -F, 'NR > 1 && $2 == "lift" {g = $10} END {print g}' "$D/traj.csv")
    if [ -n "$gl" ] && awk -v g="$gl" 'BEGIN{exit !(g > -8)}'; then
      echo "[pipeline] #$k $side 腕: 持ち上げた所の爪 ${gl}° = 何も挟んでいない（掴めなかった）"; res=missed
    fi
  fi
  echo "$k,$NAME,$mat,$conf,$iso,ok,ok,$res,$(el $t0)" >> "$LOG"
  tev "round total" "$t0" "\"result\":\"$res\",\"arm\":\"$side\""
  echo "[pipeline] #$k $side 腕: $res（$(el $t0) 秒）"
  if [ $res = placed ]; then
    echo "placed $($PY -c "import json,sys;p=json.load(open(sys.argv[1]));print('%.3f,%.3f'%tuple(p['place_target_xy']))" "$D/plan.json")" > "$D/pipeline_result"
    return 0
  fi
  if [ $res = nomove ] || [ $res = missed ]; then
    echo "$res" > "$D/pipeline_result"
    skip_here "$D"
    return 2
  fi
  echo "error" > "$D/pipeline_result"
  return 3
}

wait_premotion() {  # wait_premotion <動いている回の pid> <回>: その腕の動く前の計画（cuRobo の plan_grasp）が済むまで待つ。
  # 2026-11-25 Shreyan: 同じ GPU で次の物の計画を同時に回すと、plan_grasp が 1 回 1.2 秒 → 6〜8 秒になり、作れなかった。
  # 腕が動いている間は GPU を使わないので、そこから次の物を用意する
  local pid=$1 k=$2 t0; t0=$(now)
  for _ in $(seq 1 600); do
    grep -q "\"round\": $k, \"proc\": \"arm_server.py\".*\"stage\": \"run/pre-motion total" "$GRASP_TIMING_LOG" 2>/dev/null && break
    kill -0 "$pid" 2>/dev/null || break
    sleep 0.1
  done
  tev "6a wait for the moving arm's pre-motion planning" "$t0"
}

placed_xy() {  # 置けた回の置き先 "x,y"（無ければ何も出さない）
  [ -f "$1/pipeline_result" ] && awk '$1 == "placed" {print $2}' "$1/pipeline_result"
}

still_there() {  # still_there <物の回> <新しく撮った回>: 物の mask の中の深度が変わっていなければ 0（腕が前の物を動かしている間に用意した物の確かめ）
  $PY - "$1" "$2" <<'PYEOF'
import sys
import cv2
import numpy as np
od, nf = sys.argv[1], sys.argv[2]
m = cv2.imread(od + "/mask.png", 0)
if m is None:
    sys.exit(1)
m = m > 0
d0, d1 = np.load(od + "/depth.npy"), np.load(nf + "/depth.npy")
v = m & (d0 > 0) & (d1 > 0)
if v.sum() < 0.5 * m.sum():  # 腕などで隠れて半分も見えない: 確かめられない
    print(f"[pipeline] 次の物が半分も見えない（{int(v.sum())}/{int(m.sum())} 画素）"); sys.exit(1)
diff = np.abs(d1[v] - d0[v])
ok = np.median(diff) < 0.01 and np.mean(diff > 0.02) < 0.2
print(f"[pipeline] 次の物の深度の変わり: 中央値 {np.median(diff) * 1000:.1f} mm・2 cm 以上変わった画素 {np.mean(diff > 0.02) * 100:.0f} % → "
      + ("動いていない" if ok else "動いた"))
sys.exit(0 if ok else 1)
PYEOF
}

one() {  # 1 回。0 = 置けた（か乾式が通った）・1 = 拾う物が無い・2 = 計画か置き先が立たない・3 = 実機で失敗
  # 2026-11-25 Shreyan（速くする）:
  # - 腕が物を掴んで置いている間に、同じ 1 枚から次の物を選び、腕を決める。次の物が反対の腕の物なら計画と置き先まで済ませ、
  #   grasp/dual.py の通り道の確かめを通れば、すぐに同時に動かす。通らなければ、前の腕が終わったらすぐに動かす
  # - 次の物が同じ腕の物なら、切り出しと腕を決めるだけにして、計画はその腕が終わってから（その腕の cuRobo のサーバーの周りの物を、
  #   動いている回の途中で入れ替えないため）
  # - 前の腕が終わったら 1 枚撮り、次の物の深度が変わっていない（前の腕が動かしていない）ときだけ、用意した物を使う
  local k=$1 src=$2 rc=0
  local A_D A_NAME A_MAT A_CONF A_ISO A_T0 A_ARM
  if [ -n "$NEXT_D" ]; then
    A_D=$NEXT_D A_NAME=$NEXT_NAME A_MAT=$NEXT_MAT A_CONF=$NEXT_CONF A_ISO=$NEXT_ISO A_ARM=$NEXT_ARM; A_T0=$(now)
    NEXT_D=""
    echo "[pipeline] #$k 用意してあった物をすぐに掴む（$A_NAME・$A_MAT・$A_ARM 腕）"
  else
    prep_obj "$k" "$src" "" "" || return $?
    A_D=$P_D A_NAME=$P_NAME A_MAT=$P_MAT A_CONF=$P_CONF A_ISO=$P_ISO A_T0=$P_T0 A_ARM=$P_ARM
    LAST_FRAME=$A_D
  fi
  if [ $GO = 0 ] || [ "$OVERLAP" = 0 ]; then
    exec_obj "$k" "$A_D" "$A_NAME" "$A_MAT" "$A_CONF" "$A_ISO" "$A_T0" "$A_ARM" || rc=$?
    local xy; xy=$(placed_xy "$A_D"); [ -n "$xy" ] && PLACED="${PLACED:+$PLACED;}$xy"
    return $rc
  fi

  exec_obj "$k" "$A_D" "$A_NAME" "$A_MAT" "$A_CONF" "$A_ISO" "$A_T0" "$A_ARM" & local pa=$!
  wait_premotion "$pa" "$k"
  # ---- 腕が動いている間に、いちばん新しい 1 枚（最初の回はその回の 1 枚・次からは前の腕が終わった後に撮った確かめの 1 枚）から次の物。
  # 古い 1 枚には、もう置いた物が元の所に写っているので使わない
  local ts state="" pb="" brc=0; ts=$(now)
  local F=${LAST_FRAME:-$A_D}
  local BN; BN=$(basename "$F")_n$k
  rm -rf "$REPO/grasp/runs/$BN"; mkdir -p "$REPO/grasp/runs/$BN"
  for f in color.png depth.npy intrinsics.json floor_plane.json; do [ -f "$F/$f" ] && ln "$F/$f" "$REPO/grasp/runs/$BN/$f"; done
  local ac; ac=$($PY -c "import json,sys;c=json.load(open(sys.argv[1]))['chosen']['centroid'];print('%.3f,%.3f,0.04'%tuple(c))" "$A_D/pick.json")
  echo "[pipeline] #$k $A_ARM 腕が動いている間に、次の物を用意する"
  prep_obj "$k" "$REPO/grasp/runs/$BN" "$ac" "" 1 "$A_ARM" || brc=$?
  if [ $brc = 5 ] && [ "$DUAL" = 1 ]; then  # 次の物は動いている腕の物: 空いている腕で計画と置き先が立つか（切り出しは済んでいる）
    local idle=right; [ "$A_ARM" = right ] && idle=left
    echo "[pipeline] #$k 次の物を、空いている $idle 腕で掴めるか試す（両腕を同時に使う）"
    local brc2=0; prep_obj "$k" "$REPO/grasp/runs/$BN" "" "" 1 "" 1 "$idle" || brc2=$?
    if [ $brc2 = 0 ]; then brc=0
    else echo "[pipeline] #$k $idle 腕では立たない: $A_ARM 腕が終わってから計画する"; rm -f "$REPO/grasp/runs/$BN"/{plan.json,plan_left.json,plan_right.json}; fi
  fi
  local B_D=$P_D B_NAME=$P_NAME B_MAT=$P_MAT B_CONF=$P_CONF B_ISO=$P_ISO B_T0=$P_T0 B_ARM=$P_ARM
  case $brc in 0) state=planned;; 5) state=perceived;; *) state="";; esac
  if [ "$state" = planned ] && [ "$DUAL" = 1 ]; then  # 反対の腕の物: 通り道が離れていれば、すぐに同時に動かす
    local L=$A_D R=$B_D; [ "$A_ARM" = right ] && { L=$B_D; R=$A_D; }
    if { inproc "$RESIDENT" run dual "$L" "$R" || [ $? -ne 99 ] || $PLAN_PY grasp/dual.py "$L" "$R"; } 2>&1 | grep "^\[dual\]" \
        && grep -q '"ok": true' "$L/dual.json"; then
      exec_obj "$k" "$B_D" "$B_NAME" "$B_MAT" "$B_CONF" "$B_ISO" "$B_T0" "$B_ARM" & pb=$!
      state=started
    fi
  fi
  tev "6 next object while the arm moves" "$ts" "\"state\":\"${state:-none}\""
  local ra=0 rb=0
  wait $pa || ra=$?
  [ -n "$pb" ] && { wait $pb || rb=$?; }
  local d xy
  for d in "$A_D" ${pb:+"$B_D"}; do xy=$(placed_xy "$d"); [ -n "$xy" ] && PLACED="${PLACED:+$PLACED;}$xy"; done
  for d in "$A_D" ${pb:+"$B_D"}; do  # 動く前に止まった物を外す（exec_obj は裏のプロセスなので、PLACED はここで足す）
    if [ -f "$d/pipeline_result" ] && grep -qE "nomove|missed" "$d/pipeline_result"; then
      echo "[pipeline] $(basename "$d"): $(cat "$d/pipeline_result")（腕は動かずに止まった・掴めなかった）。この物を外して次へ"; skip_here "$d"
    fi
  done
  [ $ra = 3 ] || [ $rb = 3 ] && return 3
  local ret=0; [ $ra = 2 ] && ret=2
  [ "$state" = started ] && return $ret
  # ---- 用意した物を次の回にすぐ使う前に、1 枚撮って動いていないかを確かめる
  if [ -n "$state" ]; then
    ts=$(now)
    local V; V=$(capture); NEXT_FRAME=$V; LAST_FRAME=$V
    if still_there "$B_D" "$V"; then
      if [ "$state" = perceived ]; then  # 同じ腕の物: その腕が終わったので計画と置き先
        local ql qr; ql=$(arm_q left); qr=$(arm_q right); [ -n "$ql" ] && Q=$ql; [ -n "$qr" ] && QR=$qr
        if prep_obj "$k" "$B_D" "" "" 1 "" 1; then state=planned
          B_D=$P_D B_NAME=$P_NAME B_MAT=$P_MAT B_CONF=$P_CONF B_ISO=$P_ISO B_ARM=$P_ARM
        else state=""; fi
      fi
      if [ "$state" = planned ]; then
        NEXT_D=$B_D NEXT_NAME=$B_NAME NEXT_MAT=$B_MAT NEXT_CONF=$B_CONF NEXT_ISO=$B_ISO NEXT_ARM=$B_ARM; NEXT_FRAME=""
      fi
    fi
    tev "8 check + finish the prepared next object" "$ts" "\"ready\":$([ -n "$NEXT_D" ] && echo true || echo false)"
  fi
  return $ret
}

# 腕の cuRobo のサーバー（左 5558・右 5559）: 立っていなければ立て、回が全部終わったら自分で止める（2026-11-23 Shreyan）
srv() {  # $1 = left|right
  local port=5558 args="" log=$HOME/grasp_runs/logs/server_curobo_$1.log
  [ "$1" = right ] && port=5559 args="--arm right"  # 2026-11-25 Shreyan: 右腕も回り込みを作る（graph planner を温めずに立てると 8 GB に左右が載る）
  ss -ltn | grep -q ":$port " && return 0
  mkdir -p "$HOME/grasp_runs/logs"
  local tsrv; tsrv=$(now)
  echo "[pipeline] cuRobo（$1）を立てる"
  GRASP_CUROBO_IK_BATCH=8 GRASP_CUROBO_WARM_GRAPH=0 setsid nohup "$HOME/work/curobo/.venv/bin/python" grasp/curobo/server.py $args > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do ss -ltn | grep -q ":$port " && { tev "0 start cuRobo server ($1)" "$tsrv"; return 0; }; sleep 1; done
  echo "[pipeline] cuRobo（$1）が立たない: $log" >&2; return 1
}
stale_restart() {  # stale_restart <port>: その口の常駐が、grasp/・ik/ の .py を最後に直したより前に立っていれば止める（次の *_srv が立て直す）。
  # 2026-11-25 Shreyan: plan.py を直したのに計画の常駐が前のコードのままで、全部の計画が「知らない引数」で落ちた
  local pid age st newer
  pid=$(ss -ltnp 2>/dev/null | grep ":$1 " | grep -oE 'pid=[0-9]+' | head -1 | cut -d= -f2 || true)  # set -e: 無くても止めない
  [ -z "$pid" ] && return 0
  age=$(ps -o etimes= -p "$pid" 2>/dev/null | tr -d ' ' || true); [ -z "$age" ] && return 0
  st=$(( $(date +%s) - age ))
  newer=$(find "$REPO/grasp" "$REPO/ik" -name '*.py' -newermt "@$st" 2>/dev/null | head -1 || true)
  [ -z "$newer" ] && return 0
  echo "[pipeline] :$1 の常駐はコードを直す前に立った（$(basename "$newer") の方が新しい）。止めて立て直す"
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 50); do ss -ltn | grep -q ":$1 " || return 0; sleep 0.2; done
}
resident_srv() {  # 計画の常駐（胸のカメラ・計画・置き先・:5560）: 立っていなければ立てる
  ss -ltn | grep -q ":${RESIDENT##*:} " && return 0
  local t0 log=$HOME/grasp_runs/logs/server_camera.log; t0=$(now)
  mkdir -p "$HOME/grasp_runs/logs"
  echo "[pipeline] 計画の常駐（胸のカメラ）を立てる"
  setsid nohup $PLAN_PY grasp/tools/resident_plan.py --serve "tcp://*:${RESIDENT##*:}" > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do grep -q "で待つ" "$log" 2>/dev/null && { tev "0 start planning resident" "$t0"; return 0; }; sleep 0.5; done
  echo "[pipeline] 計画の常駐が立たない（$log）" >&2
}
recog_srv() {  # 切り出しと材質の常駐（:5561）: 立っていなければ立てる。読み込みに約 15 秒。終わっても止めない（次の回も読み込まずに済む）
  ss -ltn | grep -q ":${RECOG##*:} " && return 0
  local t0 log=$HOME/grasp_runs/logs/server_recog.log; t0=$(now)
  mkdir -p "$HOME/grasp_runs/logs"
  echo "[pipeline] 切り出しと材質の常駐（YOLOE・SigLIP2）を立てる"
  setsid nohup $PY grasp/tools/recog_server.py --bind "tcp://*:${RECOG##*:}" > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do grep -q "で待つ" "$log" 2>/dev/null && { tev "0 start recog server" "$t0"; return 0; }; sleep 1; done
  echo "[pipeline] 切り出しの常駐が立たない（$log）。回ごとに thor_snap.py を起動する" >&2
}
arm_srv() {  # arm_srv left|right: 常駐の腕（grasp/tools/arm_server.py・左 5562・右 5563）を立てる。回の間も腕をつないだまま支える
  local side=$1 port=5562 t0 log
  [ "$side" = right ] && port=5563
  [ "$side" = right ] && [ -z "${GRASP_CAN_PORT_RIGHT:-}" ] && return 0
  ss -ltn | grep -q ":$port " && return 0
  t0=$(now); log=$HOME/grasp_runs/logs/server_arm_$side.log
  echo "[pipeline] 常駐の腕（$side）を立てる"
  setsid nohup $ARM_PY grasp/tools/arm_server.py --arm "$side" > "$log" 2>&1 < /dev/null &
  for _ in $(seq 1 60); do grep -q "で待つ" "$log" 2>/dev/null && { tev "0 start arm server ($side)" "$t0"; return 0; }; sleep 0.5; done
  echo "[pipeline] 常駐の腕（$side）が立たない（$log）。この腕は回ごとに run_grasp.py を起動する" >&2
}
arm_q() {  # arm_q left|right: 常駐の腕の今の角度 "j1,...,j7"（常駐が無い・返事が途切れていれば何も出さない）
  local port=5562; [ "$1" = right ] && port=5563
  $PY - "tcp://localhost:$port" 2>/dev/null <<'PYEOF' || true
import sys, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 3000)
s.connect(sys.argv[1]); s.send_json({"op": "state"})
r = s.recv_json()
if r.get("lost") or "q" not in r:
    sys.exit(1)
print(",".join(f"{v:.2f}" for v in r["q"]))
PYEOF
}
arm_quit() {  # 常駐の腕を休みの姿勢へ戻して脱力し、止める（cuRobo のサーバーを止める前に・戻る直線の確かめに使う）
  local port
  for port in 5562 5563; do
    ss -ltn | grep -q ":$port " || continue
    echo "[pipeline] 常駐の腕（:$port）を休みの姿勢へ戻して止める: $($PY - "tcp://localhost:$port" 2>&1 <<'PYEOF'
import sys, zmq
s = zmq.Context.instance().socket(zmq.REQ); s.setsockopt(zmq.LINGER, 0); s.setsockopt(zmq.RCVTIMEO, 120000)
s.connect(sys.argv[1]); s.send_json({"op": "quit"}); print(s.recv_json())
PYEOF
)"
  done
}
arm_int() {  # Ctrl-C: 常駐の腕にも SIGINT を送る（回の途中なら run_grasp.py の止め方で元の姿勢へ戻る・待っていれば休みの姿勢へ戻る）
  echo "[pipeline] Ctrl-C: 常駐の腕にも止めを送る"
  local p
  for p in $(ss -ltnp 2>/dev/null | grep -E ':(5562|5563) ' | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u); do kill -INT "$p" 2>/dev/null || true; done
  exit 130
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
  # 段ごとの時間のまとめ（記録だけ）
  [ -n "${T0:-}" ] && tev "whole run" "$T0"
  python3 "$REPO/grasp/tools/timing_report.py" "$GRASP_TIMING_LOG" 2>/dev/null || true
  [ -n "$VIDEO" ] && echo "[pipeline] 録画: $(zreq '{"op":"record_stop"}' 2>/dev/null)"
  [ $GO = 1 ] && arm_quit
  [ $GO = 1 ] && stop_srv
  return 0
}
# 常駐がコードを直す前に立っていれば立て直す・計画の常駐が無ければ立てる（録画はこの常駐が持つので、録画を始める前に）
for p in "${RESIDENT##*:}" "${RECOG##*:}" 5558 5559; do stale_restart "$p"; done
resident_srv
# --record: 胸のカメラ（計画の常駐が持っている）で、回の全部を grasp/runs/pipeline_<日時>.mp4 に録る（2026-11-23 Shreyan）
if [ $RECORD = 1 ]; then
  VIDEO=$REPO/${LOG%.csv}.mp4
  zreq "{\"op\":\"record_start\",\"path\":\"$VIDEO\",\"fps\":15}" >/dev/null && echo "[pipeline] 録画を始めた: $VIDEO" \
    || { echo "[pipeline] 録画を始められない（計画の常駐 $RESIDENT が要る）" >&2; VIDEO=""; }
fi
[ $YOLOE = 1 ] && recog_srv
if [ $GO = 1 ]; then srv left; srv right; arm_srv left; arm_srv right; fi
trap finish EXIT
trap arm_int INT
T0=$(date +%s.%N); fails=0; PLACED=""
echo "[pipeline] 最大 $N 回・物が無くなる（拾える物が残らない）か、続けて 3 回掴めない／置けないか、STOP で終わる"
for k in $(seq 1 "$N"); do
  [ -f ~/grasp_runs/STOP ] && { echo "[pipeline] STOP"; break; }
  export GRASP_TIMING_ROUND=$k
  if [ $GO = 1 ]; then  # 腕は休みの姿勢ではなく上げた姿勢で待っていることがある: 今の角度で腕の点を消し、計画の始まりにする
    ql=$(arm_q left); qr=$(arm_q right)
    [ -n "$ql" ] && Q=$ql; [ -n "$qr" ] && QR=$qr
  fi
  if [ -n "$NEXT_D" ]; then src=$NEXT_D  # 用意してあった物（撮らない）
  elif [ -n "$NEXT_FRAME" ]; then src=$NEXT_FRAME; NEXT_FRAME=""  # 前の回の終わりに撮った 1 枚
  else src=$([ $LIVE = 1 ] && capture || echo "$CAP"); fi
  rc=0; one "$k" "$src" || rc=$?
  case $rc in
    1) echo "[pipeline] 拾う物が無い: 終わる"; break;;
    3) echo "[pipeline] 実機で失敗: 止める（$src/run.log）"; break;;
    2) fails=$((fails+1)); [ $fails -ge 3 ] && { echo "[pipeline] 続けて 3 回、掴めないか置けない: 止める"; break; };;
    0) fails=0;;
  esac
done
echo; column -s, -t "$LOG"; echo "[pipeline] 合計 $(el $T0) 秒・記録 $LOG"
