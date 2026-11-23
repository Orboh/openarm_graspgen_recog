#!/usr/bin/env bash
# Upload run records to S3 so they cannot be lost (2026-11-23 Sota: 実行の結果は毎回 S3 に上げる。専用のフォルダ).
# On 11/23 Thor's grasp/runs was wiped by a git checkout; only what was elsewhere survived.
#
#   bash grasp/tools/s3_sync.sh               # everything (records, raw captures, hand-eye, CSVs, RUNLOG)
#   bash grasp/tools/s3_sync.sh <dir> [...]   # just these folders (e.g. one capture after a run)
#
# Where: $GRASP_S3_ROOT (s3://orboh-models/openarm-grasp-runs) / $GRASP_S3_HOST (thor | pc) / ...
#   grasp/runs/<x>      <- <repo>/grasp/runs/<x>        (plans, run.log, traj.csv, wrist images, raw images)
#   grasp_runs/<x>      <- ~/grasp_runs/<x>              (Thor: raw captures, handeye_near_*, trial CSVs, logs)
#   RUNLOG.md           <- ~/rollouts/RUNLOG.md
# Only adds / updates (aws s3 sync, never --delete): deleting something here never deletes it on S3.
# Credentials: the SSO profile $GRASP_S3_PROFILE (kota0). When it has expired the folders are written to
# ~/grasp_runs/S3_PENDING and a warning is printed; the next run with valid credentials uploads them too.
# Never fails the caller (exit 0), so it can run after every trial.
set -u
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE=${GRASP_S3_PROFILE:-kota0}
ROOT=${GRASP_S3_ROOT:-s3://orboh-models/openarm-grasp-runs}
HOSTN=${GRASP_S3_HOST:-$( [ "$(whoami)" = orbou-thor ] && echo thor || echo pc )}
mkdir -p ~/grasp_runs
LOG=~/grasp_runs/s3_sync.log; PENDING=~/grasp_runs/S3_PENDING
LOCK=~/grasp_runs/.s3_sync.lock
now() { date "+%F %T"; }

# S3 destination of a local folder
dest() {
  local p; p=$(readlink -f "$1")
  case "$p" in
    "$(readlink -f "$REPO/grasp/runs")"/*) echo "$ROOT/$HOSTN/grasp/runs/${p#$(readlink -f "$REPO/grasp/runs")/}" ;;
    "$(readlink -f "$REPO/grasp/runs")")   echo "$ROOT/$HOSTN/grasp/runs" ;;
    "$(readlink -f ~/grasp_runs)"/*)       echo "$ROOT/$HOSTN/grasp_runs/${p#$(readlink -f ~/grasp_runs)/}" ;;
    "$(readlink -f ~/grasp_runs)")         echo "$ROOT/$HOSTN/grasp_runs" ;;
    *)                                     echo "$ROOT/$HOSTN/other/$(basename "$p")" ;;
  esac
}

if [ $# -gt 0 ]; then TARGETS=(); for a in "$@"; do TARGETS+=("$(readlink -f "$a")"); done; else
  TARGETS=("$REPO/grasp/runs"); [ -d ~/grasp_runs ] && TARGETS+=(~/grasp_runs)
fi
# earlier folders that could not be uploaded
[ -s "$PENDING" ] && mapfile -t OLD < "$PENDING" && TARGETS+=("${OLD[@]}")

if ! aws sts get-caller-identity --profile "$PROFILE" >/dev/null 2>&1; then
  for t in "${TARGETS[@]}"; do grep -qxF "$t" "$PENDING" 2>/dev/null || echo "$t" >> "$PENDING"; done
  echo "$(now) NO-CREDENTIALS profile=$PROFILE pending=$(wc -l < "$PENDING")" >> "$LOG"
  echo "[s3] WARNING: S3 の鍵が切れている（profile $PROFILE）。上げられなかった分は $PENDING に残した。" \
       "ログイン: aws sso login --profile $PROFILE --use-device-code" >&2
  exit 0
fi

exec 9>"$LOCK"; flock -w 1800 9 || { echo "$(now) LOCK-TIMEOUT" >> "$LOG"; exit 0; }
fail=()
for t in "${TARGETS[@]}"; do
  [ -e "$t" ] || continue
  d=$(dest "$t")
  if [ -d "$t" ]; then
    aws s3 sync "$t" "$d" --profile "$PROFILE" --only-show-errors --exclude "*.lock" >> "$LOG" 2>&1
  else
    aws s3 cp "$t" "$d" --profile "$PROFILE" --only-show-errors >> "$LOG" 2>&1
  fi
  if [ $? -eq 0 ]; then echo "$(now) ok $t -> $d" >> "$LOG"; else echo "$(now) FAIL $t -> $d" >> "$LOG"; fail+=("$t"); fi
done
# the run log of the robot (Thor) every time
if [ -f ~/rollouts/RUNLOG.md ]; then
  aws s3 cp ~/rollouts/RUNLOG.md "$ROOT/$HOSTN/RUNLOG.md" --profile "$PROFILE" --only-show-errors >> "$LOG" 2>&1 || fail+=(~/rollouts/RUNLOG.md)
fi
# what is still pending = what failed now
if [ ${#fail[@]} -gt 0 ]; then printf '%s\n' "${fail[@]}" > "$PENDING"; echo "[s3] WARNING: ${#fail[@]} folders failed to upload (see $LOG)" >&2
else rm -f "$PENDING"; fi
exit 0
