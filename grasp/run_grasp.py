#!/usr/bin/env python3
"""Step 6: execute plan.json on the real arm (runs on Thor, lerobot venv). **The arm moves.** Dry run by default.

Sequence (all joint-space interpolation, with gravity feed-forward):
  start (the current resting pose) → [open gripper] → pregrasp (15 deg/s) → grasp (6 deg/s, slow)
  → [close gripper] → lift (8 deg/s) → hold 2 s → grasp (put it back down) → [open gripper] → pregrasp → start → torque off
Before moving, FK-checks every segment at 20 points to make sure the TCP and link7 stay above the floor (+2 cm) (aborts if not).
GRASP_CUROBO があれば cuRobo の腕の球と置き場の周りの物（本体・カメラ・机）でも確かめ、移動の区間（物の近く以外）が当たるときは
cuRobo で回り込む経由点を作って順に動かす（grasp/curobo/）。物の近くの区間は直線のままで、当たれば止まる。
Writes the joint-angle trajectory to traj.csv in the run folder.

    ~/projects/lerobot/.venv/bin/python run_grasp.py <cap_dir>          # check only
    ~/projects/lerobot/.venv/bin/python run_grasp.py <cap_dir> --go     # run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

import numpy as np

REPO = os.path.expanduser(os.environ.get("OPENARM_REPO", "~/projects/openarm-orboh"))  # PC dry checks: OPENARM_REPO=~/work/openarm-orboh
URDF = os.path.join(REPO, "urdf", "openarm_v10_bimanual.urdf")
J = [f"joint_{i}" for i in range(1, 8)]
# 腕が入ってはいけない箱（plan.py の KEEPOUT と同じ・2026-11-21 北九州で胸のカメラの取り付け部に当たった）。無ければ確かめない
KEEPOUT = [np.array([float(v) for v in b.split(",")]) for b in os.environ.get("GRASP_KEEPOUT_BOXES", "").split(";") if b.strip()]
TCP_LEN, TIP_LEN = 0.2145, 0.2295  # link7 から指先の中心・爪の先（plan.py の CFG と同じ値。東京の爪の実測）
# gripper.pos (lerobot deg). fully open -66 / fully closed +6.1。腕ごとの値は main で plan.json の腕から決める（右は *_RIGHT）
G_OPEN = float(os.environ.get("GRASP_GRIPPER_OPEN", "-60.0")); G_CLOSE = float(os.environ.get("GRASP_GRIPPER_CLOSE", "2.0"))
FLOOR_Z = float(os.environ.get("GRASP_FLOOR_Z", "0.27"))                # floor height in the base frame (from the T_base_cam fit)
if os.environ.get("GRASP_SITE") and "GRASP_FLOOR_Z" not in os.environ:  # 東京の台車以外で床の高さを決め忘れたら動かさない
    sys.exit("[grasp] GRASP_FLOOR_Z が無い（GRASP_SITE=" + os.environ["GRASP_SITE"] + "）。東京の台車の値のまま床の確認をしない。grasp/config/sites/ で決める")
class MotorLost(SystemExit):
    """モーターの返事が途切れた（非常停止・電源断）。これ以上は腕に指令を送らず、記録も先へ進めない。"""


class _SettleCam:
    """手首のカメラの grab の待ち（settle_s）だけを変える包み（grasp/wrist.py の WristCam は変えない）。"""

    def __init__(self, cam, settle_s: float):
        self.cam, self.settle_s = cam, settle_s

    def grab(self, *args, **kw):
        kw.setdefault("settle_s", self.settle_s)
        return self.cam.grab(*args, **kw)

    def __getattr__(self, name):
        return getattr(self.cam, name)


class Blocked(SystemExit):
    """区間が床・keep-out の箱・置き場の周りの物（cuRobo）に当たる。移動の区間なら cuRobo で回り込める（route）。
    SystemExit の仲間なので、cuRobo が無いときは今までどおり止まる。"""


CUROBO = None  # GRASP_CUROBO（cuRobo のサーバー grasp/curobo/server.py）があれば、腕の球と置き場の箱でも区間を確かめる
if os.environ.get("GRASP_CUROBO") or os.environ.get("GRASP_CUROBO_RIGHT"):  # 右腕は GRASP_CUROBO_RIGHT（main で腕を決めて向ける）
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        from grasp.curobo import client as CUROBO
    except ImportError as e:
        sys.exit(f"[grasp] GRASP_CUROBO があるのに grasp/curobo/client.py が読めない（{e}）。当たりを確かめずに動かさない")


def clear_lift_dz(tip_z: float, obj_top_z: float, min_dz: float = 0.06, clear: float = 0.03) -> float:
    """置いて離した後に真上へ上げる量 [m]: 爪の先が置いた物の上端 + clear に来るまで。少なくとも min_dz（今までの運ぶ高さ）。"""
    return float(max(min_dz, obj_top_z + clear - tip_z))


def shift_box(box: dict, delta, name: str = "placed") -> dict:
    """cuRobo の箱（name・dims・pose [x, y, z, qw, qx, qy, qz]）を delta [m] だけ平行に動かした箱。"""
    p = [float(v) for v in box["pose"]]
    p[:3] = [p[i] + float(delta[i]) for i in range(3)]
    return {"name": name, "dims": [float(v) for v in box["dims"]], "pose": p}


def main(argv=None, *, robot=None, plan_wait=None, t_trigger=None) -> int:
    """robot: 接続済みのロボット（常駐のプロセス grasp/tools/grasp_daemon.py が使い回す）。渡したときは接続を切らず、終わりに脱力だけする。
    plan_wait: --early のとき、計画（<cap_dir>/plan.json）ができるまで待って、立ったら True を返す口。
    t_trigger: 合図の時刻（time.monotonic()）。あれば「合図から動き出しまで」の秒数を出す。"""
    ap = argparse.ArgumentParser()
    ap.add_argument("cap_dir")
    ap.add_argument("--go", action="store_true")
    ap.add_argument("--early", action="store_true",
                    help="計画を待たずに、休みの姿勢から真上へ上げて爪を開き始め、その間に裏で計画（plan_wait）と確かめを済ませる。"
                         "計画が立たない・確かめを通らなければ、物の近くへは行かずに休みの姿勢へ戻る（2026-11-23 Munenori・"
                         "「計画して」から動き出すまでを 1 秒以内にする）。--go と一緒に、grasp/tools/grasp_daemon.py から使う")
    ap.add_argument("--no-smooth", dest="smooth", action="store_false",
                    help="lerobot の soft_move_to_position（一定の速さ・いきなり止まる）で動かす。既定は grasp/motion.py の滑らかな動き")
    ap.add_argument("--correct-vias", action="store_true", help="回り込みの経由点ごとにも位置の補正をかける（前の動き）")
    ap.add_argument("--no-replan-place", dest="replan_place", action="store_false",
                    help="閉じた後の実際の角度で運ぶ・置く姿勢を解き直さない（既定は解き直す・2026-11-22）")
    ap.add_argument("--probe-hold", type=float, default=0.0, help="--probe のとき掴む姿勢で止めておく秒数（人がずれを測る間）")
    ap.add_argument("--probe", action="store_true", help="don't grasp: stop at the grasp pose, take one chest frame, and go back (to check the position error)")
    ap.add_argument("--transit-dps", type=float, default=15.0, help="joint speed for free-space moves [deg/s] (OpenArm exception, Sota 11/20)")
    ap.add_argument("--gripper-dps", type=float, default=None,
                    help="グリッパーだけを開閉する段（閉じる・離す）の速さ [deg/s]。無ければ今までどおり max(--near-dps, 15)。"
                         "2026-11-22 北九州: 閉じるのに 20 deg/s で約 3 秒かかったので速くできるようにした")
    ap.add_argument("--lift-dps", type=float, default=None, help="掴んだ後に持ち上げる速さ [deg/s]（無ければ --near-dps・2026-11-23）")
    ap.add_argument("--carry-dps", type=float, default=None, help="持って置き先の上へ運ぶ速さ [deg/s]（無ければ --near-dps）")
    ap.add_argument("--place-dps", type=float, default=None,
                    help="置き先へ下ろす・置いた所から上がる・元の場所に下ろす速さ [deg/s]（無ければ --near-dps）")
    ap.add_argument("--close-wait", type=float, default=0.5, help="爪を閉じてから持ち上げるまで待つ秒数")
    ap.add_argument("--fix-s", type=float, default=0.6, help="位置の補正（足りない分の送り直し）1 回の長さ [s]")
    ap.add_argument("--near-dps", type=float, default=6.0, help="joint speed near the object/floor: approach, close, put down, release [deg/s]")
    ap.add_argument("--hold-home", action="store_true",
                    help="end every trial at --home and keep torque ON (disconnect without disabling torque); "
                         "release later with grasp/tools/release_arm.py (Sota 11/20)")
    ap.add_argument("--wrist-snap", action="store_true", help="take one wrist-camera frame right before closing (evidence only)")
    ap.add_argument("--wrist-servo", action="store_true",
                    help="掴む姿勢の手前で手首の D405 で爪から見た物の位置を測り、横を直して少しずつ下ろしてから閉じる"
                         "（grasp/wrist_approach.py・左腕だけ・2026-11-23）。爪の模型は GRASP_WRIST_JAW か --wrist-jaw")
    ap.add_argument("--wrist-jaw", default=os.environ.get("GRASP_WRIST_JAW"), help="手首のカメラから見た爪の模型（wrist_jaw_measure.py が書く JSON）")
    ap.add_argument("--servo-near", type=float, default=0.04, help="--wrist-servo で最初に測る所の下限: 掴む姿勢から近づく向きに戻す量 [m]（爪の先が物の上の面より 2.5 cm 上に来るまでは戻す）")
    # wrist servo と止める時間を速くする口（2026-11-23 Munenori: 掴む直前の wrist servo に約 25 秒かかった）。既定は今までどおり
    ap.add_argument("--servo-dps", type=float, default=4.0, help="--wrist-servo の中で動く関節の速さ [deg/s]（--near-dps を超えない）")
    ap.add_argument("--servo-settle", type=float, default=0.6, help="--wrist-servo で止まってから手首のカメラで撮るまで待つ秒数")
    ap.add_argument("--servo-step", type=float, default=0.02, help="--wrist-servo で下ろす 1 段の長さ [m]（段ごとに止まって測り直す）")
    ap.add_argument("--hold-s", type=float, default=2.0, help="持ち上げてから止めておく秒数（掴めたかを見る間）")
    ap.add_argument("--servo-one-shot", action="store_true",
                    help="--wrist-servo で 1 回だけ測り、横の直しと下ろす動きを 1 回の動きにまとめる（閉じる向きに 12 mm を超えて"
                         "ずれている・物が画像の端で切れているときは今までどおり何度か直す・2026-11-23）")
    ap.add_argument("--servo-t-shift", type=float, default=0.0,
                    help="--wrist-servo で、物に対して手を爪の厚みの向き（爪の座標の +Yg）へ寄せる量 [m]。手首のカメラから見た爪の模型の"
                         "厚みの向きのずれを直す（2026-11-23: 毎回、爪の奥側（−Yg）の長辺で木材を挟んでいたので −0.02）")
    ap.add_argument("--servo-depth", type=float, default=0.030, help="--wrist-servo で、物の一番太い所を爪の先からこれだけ奥へ [m]")
    ap.add_argument("--home", default=None, help="start/end pose YAML (joint_1..7, lerobot deg), e.g. config/home_left_cart-sym-v4.yaml: "
                    "go there first (via an 8 cm raise), use it as the start, come back to it at the end")
    ap.add_argument("--avoid", action="store_true",
                    help="運ぶ区間（lift→置き先の上）と戻る区間（置き先の上→raise）を、胸のカメラで割り込む物を見ながら動く"
                         "（grasp/reactive.py・止まって経路を作り直す）。GRASP_CUROBO と plan_place.py の置き先が要る")
    ap.add_argument("--legacy-approach", action="store_true",
                    help="cuRobo の plan_grasp を使わず、前の動き（route の回り込み・関節の直線で差し込む）にする")
    ap.add_argument("--pregrasp-m", type=float, default=0.10, help="plan_grasp の pregrasp: 掴む姿勢から爪の向きに戻す量 [m]（plan.py --pregrasp と同じ 10 cm）")
    ap.add_argument("--plan-target-pad", type=float, default=0.03,
                    help="plan_grasp で近づく軌道を作るときの拾う物の箱の余白 [m]。作れなければ 2 cm・--target-pad に下げる")
    ap.add_argument("--raise-m", type=float, default=0.08,
                    help="休みの姿勢から最初に真上へ上げる量 [m]。解けなければ 8・5・3 cm へ下げる（2026-11-23: 休みの姿勢の爪のすぐ横に"
                         "立てた木材（高さ 9 cm）があると、8 cm では爪が木材の高さに残り、cuRobo の plan_grasp が出発点で当たって作れなかった）")
    ap.add_argument("--lift-m", type=float, default=0.10, help="plan_grasp の持ち上げ: 掴む姿勢から真上へ [m]（plan.py と同じ 10 cm）")
    ap.add_argument("--no-avoid-target", dest="avoid_target", action="store_false",
                    help="移動の区間（休み↔raise↔pregrasp）で拾う物を cuRobo の障害物に入れない（前の動き）。"
                         "既定は入れて、当たる直線は回り込む（2026-11-23: raise→pregrasp で手が木材をずらした）")
    ap.add_argument("--target-grow", type=float, default=0.04,
                    help="よける箱に、色の点から横にこれ以内の深度の点（机より上）も入れる [m]。色で物の一部しか切り出せないとき用。0 で入れない")
    ap.add_argument("--target-arm-r", type=float, default=0.05,
                    help="拾う物の箱に深度の点を足すとき、今の腕のグリッパーの軸からこの半径 [m] の点は腕自身として足さない（0 で足す）")
    ap.add_argument("--target-pad", type=float, default=0.010,
                    help="移動の区間でよける拾う物の箱の余白 [m]（cuRobo の余裕 10 mm とは別。1.5 cm だと pregrasp が箱に掛かり回り込みを作れなかった・2026-11-23）")
    ap.add_argument("--avoid-dps", type=float, default=10.0, help="--avoid で見ながら動くときの関節の速さ [deg/s]")
    ap.add_argument("--start", default=None, help="start pose (lerobot deg, comma-separated). Default = read from the robot (only with --go)")
    ap.add_argument("--home-right", default=None, help="右腕のときの --home（左の初期姿勢を右腕に使わない）。無ければ GRASP_HOME_RIGHT")
    a = ap.parse_args(argv)
    if a.early:
        if not a.go:
            sys.exit("[run] --early は --go と一緒にだけ使う（乾式の確かめは計画ができてから --early なしで打つ）")
        if plan_wait is None:
            sys.exit("[run] --early には計画を待つ口（plan_wait）が要る（grasp/tools/grasp_daemon.py から使う）")
        if a.home or a.avoid or a.probe:
            sys.exit("[run] --early は --home・--avoid・--probe と一緒に使えない（休みの姿勢から真上へ上げる動きだけを先に始める）")
    if a.early:  # 計画はまだ無い。腕は GRASP_ARM（無ければ左）で上げ始め、計画の腕が違えば物の近くへ行かずに戻る
        plan = None
        side = os.environ.get("GRASP_ARM", "left")
    else:
        plan = json.load(open(os.path.join(a.cap_dir, "plan.json")))
        # どちらの腕か（plan.py が書く・2026-11-22 Munenori）。2026-11-22 より前の plan.json は左
        side = plan.get("arm", "left")
    if side not in ("left", "right"):
        sys.exit(f"[run] plan.json の arm が left でも right でもない: {side!r}")
    os.environ["GRASP_ARM"] = side
    global CUROBO, G_OPEN, G_CLOSE
    if CUROBO is not None:
        CUROBO.set_arm(side)  # 右で GRASP_CUROBO_RIGHT が無ければ止まる（左のサーバーで右腕を見ない）
        if not CUROBO.enabled():
            CUROBO = None
    if side == "right":
        missing = [k for k in ("GRASP_CAN_PORT_RIGHT", "GRASP_ROBOT_ID_RIGHT", "GRASP_GRIPPER_OPEN_RIGHT", "GRASP_GRIPPER_CLOSE_RIGHT",
                               "GRASP_REST_RIGHT")
                   if k not in os.environ]
        if a.go and missing:  # 左の CAN・較正・グリッパーの角度で右腕を動かさない
            sys.exit(f"[run] 右腕の値が置き場の env に無い: {', '.join(missing)}。右腕を実機で確かめてから書く")
        G_OPEN = float(os.environ.get("GRASP_GRIPPER_OPEN_RIGHT", G_OPEN)); G_CLOSE = float(os.environ.get("GRASP_GRIPPER_CLOSE_RIGHT", G_CLOSE))
        if a.home:
            a.home = a.home_right or os.environ.get("GRASP_HOME_RIGHT")
            if not a.home:
                sys.exit("[run] 右腕なのに右の初期姿勢（--home-right か GRASP_HOME_RIGHT）が無い。左の --home を右腕に使わない")
    print(f"[run] 腕: {side}", flush=True)
    # IK はこの repo の物を先に読む（右腕の OpenArmIK があるのはこちら。OPENARM_REPO の古い写しには無い）
    sys.path.insert(0, REPO)
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ik.openarm_left_ik import OpenArmIK
    ik = OpenArmIK(URDF, side=side)

    def tcp_z(qd):
        M = ik.fk(np.deg2rad(qd)); return float((M.translation + M.rotation @ [0, 0, 0.2145])[2])

    def l7_z(qd):
        return float(ik.fk(np.deg2rad(qd)).translation[2])

    def check(q_from, q_to, name, allow_low=False, quiet=False):
        # the resting pose already sits close to the floor, so allow down to "the lower of the two ends' TCP heights, minus 1 cm"
        tmin = min(FLOOR_Z + 0.02, tcp_z(q_from) - 0.01, tcp_z(q_to) - 0.01)
        # link7 も同じ考え方（2026-11-21 北九州）: 机に寝かせた休みの姿勢は手首が机から 4 cm ほどで、5 cm の線を始めから下回る。
        # 区間の端が低いときだけ「低い方の端の高さ − 1 cm」まで許す。両端が高い区間の線（FLOOR_Z + 5 cm）は変わらない
        l7min = min(FLOOR_Z + 0.05, l7_z(q_from) - 0.01, l7_z(q_to) - 0.01)
        for s in np.linspace(0, 1, 20):
            q = np.deg2rad(np.asarray(q_from) + (np.asarray(q_to) - np.asarray(q_from)) * s)
            M = ik.fk(q); tcp = M.translation + M.rotation @ [0, 0, 0.2145]
            if KEEPOUT:  # 腕が胸のカメラと取り付け部の箱に入る動き（本体のフレームに当たる）を止める
                js = [ik.data.oMi[ik.model.getJointId(f"openarm_{side}_joint{j}")].translation.copy() for j in range(3, 8)]
                P = np.array(js + [(p0 + p1) / 2 for p0, p1 in zip(js[:-1], js[1:])]
                             + [M.translation + M.rotation @ [0, 0, TCP_LEN], M.translation + M.rotation @ [0, 0, TIP_LEN]])
                for b in KEEPOUT:
                    if np.any((P[:, 0] >= b[0]) & (P[:, 0] <= b[1]) & (P[:, 1] >= b[2]) & (P[:, 1] <= b[3]) & (P[:, 2] >= b[4]) & (P[:, 2] <= b[5])):
                        raise Blocked(f"[run] {name}: at s={s:.2f} the arm enters the keep-out box {b.round(3).tolist()}: "
                                         "it would hit the body frame / chest camera. Stopping")
            if M.translation[2] < l7min or (not allow_low and tcp[2] < tmin):
                raise Blocked(f"[run] {name}: at s={s:.2f} link7 z={M.translation[2]:.3f} TCP z={tcp[2]:.3f}, too close to the floor. Stopping")
        if CUROBO is not None:
            # 机は、端の姿勢がもともと机に近い区間（机に寝かせた休みの姿勢）だけ「低い方の端 − 1 cm」まで許す（上の link7 と同じ考え方）
            d_end = min(CUROBO.check_path([q], n=1)[0]["table/arm"][0] for q in (q_from, q_to))
            bad = CUROBO.violations([q_from, q_to], names=[name], table_margin=min(CUROBO.TABLE_MARGIN, d_end - 0.01))
            if bad:
                raise Blocked(f"[run] cuRobo: {CUROBO.describe(bad)}。腕が置き場の周りの物（本体・カメラ・机）に当たる。Stopping")
        if not quiet:
            print(f"[run] check ok: {name}")

    def load_obstacles(q_now):
        """記録のカメラの障害物（plan.py が残した点）を、今の腕の角度 q_now で腕の点を消し直してサーバーに入れる。
        記録に無ければ障害物を空にする（前の記録の障害物を残さない）。"""
        if CUROBO is None:
            return
        # 右腕は plan.py が書いた obstacles_right.json（撮影時の右腕の姿勢で右腕の点を消す）
        pf = os.path.join(a.cap_dir, "obstacle_points.npy")
        jf = os.path.join(a.cap_dir, "obstacles.json" if side == "left" else "obstacles_right.json")
        if not (os.path.exists(pf) and os.path.exists(jf)):
            CUROBO.set_obstacle_boxes([])
            print("[run] cuRobo: カメラの障害物の記録が無い（置き場の物だけで見る）", flush=True)
            return
        meta = json.load(open(jf))
        if meta.get("object_boxes") is not None:  # plan.py --obstacle-objects: ほかの物を 1 つずつ箱で（全区間で障害物）
            tgt["seen"] = list(meta["object_boxes"])
            CUROBO.set_obstacle_boxes(tgt["seen"])
            print(f"[run] cuRobo: ほかの物の箱 {len(tgt['seen'])} 個" + "".join(
                f"・{b['name']} 中心 {np.round(b['pose'][:3], 2).tolist()} 大きさ {np.round(b['dims'], 2).tolist()}" for b in tgt["seen"]), flush=True)
            return
        if meta.get("ignore_seen"):  # plan.py --ignore-seen: カメラで見た物は障害物にしない（置き場の物・机・もう片方の腕は見る）
            CUROBO.set_obstacle_boxes([])
            print("[run] cuRobo: カメラで見た物は障害物に入れない（plan.py --ignore-seen・置き場の物と机だけで見る）", flush=True)
            return
        # 腕自身の点を消す幅は plan.py と同じ（GRASP_SEEN_ARM_MARGIN(_RIGHT)・2026-11-22 右の爪の先が 3〜4.2 cm に写る）
        margin = float(os.environ.get("GRASP_SEEN_ARM_MARGIN" + ("_RIGHT" if side == "right" else ""), "0.03"))
        r = CUROBO.set_obstacle_points(np.load(pf), [list(q_now)] + meta["arm_qs"], meta["cam_pos"], meta["floor_z"],
                                       arm_margin=margin)
        tgt["seen"] = list(r["boxes"])
        print(f"[run] cuRobo: カメラで見た障害物の箱 {len(r['boxes'])} 個" + "".join(
            f"・{b['name']} 中心 {np.round(b['pose'][:3], 2).tolist()} 大きさ {np.round(b['dims'], 2).tolist()}" for b in r["boxes"]), flush=True)

    # 拾う物を、移動の区間（休み↔raise↔pregrasp）でだけ cuRobo の障害物に入れる（2026-11-23 Munenori）。
    # cap_resident_20261123_114711: raise→pregrasp は関節で最大 140° 回る関節の直線で、手が木材の上 0〜2 cm を横切り（FK で最小 1 mm）、
    # 掴む前に木材を押してずらしていた。拾う物は seen_points が障害物から外すので、今までどの確かめにも入っていなかった。
    # 物の近くの区間（pregrasp→grasp・持ち上げ・置く）と、持って運ぶ区間では外す（そこは手が物に触れる・物が手と動く）。
    tgt = {"seen": [], "box": None, "on": False}

    def target_box(pad=None, quiet=False):
        """object.npy の点（付け根の座標）を囲む軸に沿った箱 + pad（無ければ --target-pad）。GRASP_T_BASE_CAM か object.npy が無ければ None。"""
        tf, of = os.environ.get("GRASP_T_BASE_CAM"), os.path.join(a.cap_dir, "object.npy")
        if not tf or not os.path.exists(tf) or not os.path.exists(of):
            return None
        T = np.array(json.load(open(tf))["T_base_cam"])
        P = np.load(of) @ T[:3, :3].T + T[:3, 3]
        n_obj = len(P)
        # 色で切り出した点は物の一部のことがある（2026-11-23 cap_resident_20261123_120003: 立てた丸太（直径 5 cm）の影の側だけ
        # 1.8 cm 幅が木材になり、明るい側は色から外れた。--ignore-seen で障害物にも入らず、行きに手が丸太に触れた）。
        # 深度の点のうち、物の点から横に --target-grow 以内で、机より 1 cm 上から物の上面 + 2 cm までの点も物として箱に入れる
        df, kf = os.path.join(a.cap_dir, "depth.npy"), os.path.join(a.cap_dir, "intrinsics.json")
        if a.target_grow > 0 and os.path.exists(df) and os.path.exists(kf):
            from grasp.capture import depth_to_points  # lerobot の venv に scipy が無いので numpy だけで近さを見る
            S, _ = depth_to_points(np.load(df), np.array(json.load(open(kf))["K"]), zmax=1.2)
            S = S @ T[:3, :3].T + T[:3, 3]
            g = a.target_grow
            S = S[(S[:, 2] > FLOOR_Z + 0.01) & (S[:, 2] < P[:, 2].max() + 0.02)
                  & np.all(S[:, :2] > P[:, :2].min(0) - g, axis=1) & np.all(S[:, :2] < P[:, :2].max(0) + g, axis=1)]
            if len(S) and a.target_arm_r > 0 and start is not None:
                # 2026-11-23: 休みの姿勢の爪のすぐ横に物があると、爪の点まで物として足し、箱が爪を含んだ（cap_resident_20261123_155031:
                # 色の点 561・深度で足した点 4597・箱の中で爪が −30 mm）。今の腕のグリッパー（link7 から爪の先まで）のまわりの点は足さない
                Mg = ik.fk(np.deg2rad(start)); g0 = Mg.translation; ax = Mg.rotation @ [0, 0, 1.0]
                tseg = np.clip((S - g0) @ ax, 0.0, TIP_LEN + 0.01)
                S = S[np.linalg.norm(S - (g0 + tseg[:, None] * ax), axis=1) > a.target_arm_r]
            if len(S):
                dxy = np.min(np.linalg.norm(S[:, None, :2] - P[None, ::max(1, len(P) // 300), :2], axis=2), axis=1)
                P = np.vstack([P, S[dxy < g]])
        pad = a.target_pad if pad is None else pad
        lo, hi = P.min(0) - pad, P.max(0) + pad
        if not quiet:
            print(f"[run] 拾う物の箱: 色の点 {n_obj}・深度で足した点 {len(P) - n_obj}・物の外形 {np.round(np.ptp(P, 0), 3).tolist()} m", flush=True)
        lo[2] = max(lo[2], FLOOR_Z)  # 机より下には伸ばさない（机は置き場の箱が見る）
        return {"name": "target", "dims": (hi - lo).round(4).tolist(), "pose": ((lo + hi) / 2).round(4).tolist() + [1.0, 0.0, 0.0, 0.0]}

    def avoid_target(on):
        """拾う物の箱を cuRobo に入れる（on）・外す。カメラで見た障害物の箱はそのまま残す。"""
        if CUROBO is None or not a.avoid_target or tgt["on"] == on:
            return
        if on and tgt["box"] is None:
            tgt["box"] = target_box()
            if tgt["box"] is None:
                print("[run] cuRobo: 拾う物の点（object.npy・GRASP_T_BASE_CAM）が無いので、移動の区間で物をよけない", flush=True)
                return
            b = tgt["box"]
            print(f"[run] cuRobo: 移動の区間では拾う物もよける（中心 {np.round(b['pose'][:3], 3).tolist()}・"
                  f"大きさ {np.round(b['dims'], 3).tolist()}・余白 {a.target_pad * 100:.1f} cm）", flush=True)
        CUROBO.set_obstacle_boxes(tgt["seen"] + ([tgt["box"]] if on else []))
        tgt["on"] = on

    def floor_line(q_from, q_to):
        """check と同じ床の線（回り込みの確かめに渡す）。"""
        return {"tcp_min_z": min(FLOOR_Z + 0.02, tcp_z(q_from) - 0.01, tcp_z(q_to) - 0.01),
                "l7_min_z": min(FLOOR_Z + 0.05, l7_z(q_from) - 0.01, l7_z(q_to) - 0.01), "tcp_len": TCP_LEN}

    def route(q_from, q_to, name):
        """移動の区間を確かめ、当たるなら cuRobo で回り込む。返り値は通る姿勢の列（始点を除き終点を含む）。

        物の近くの区間（pregrasp↔grasp↔lift・置く）には使わない（そこは直線で動かし、当たれば check が止める）。
        回り込みの経由点の間も、床・keep-out の箱・cuRobo の確認（check）をもう一度通す。"""
        try:
            check(q_from, q_to, name)
            return [list(q_to)]
        except Blocked as e:
            if CUROBO is None:  # cuRobo の無い置き場（東京）は今までどおり止まる
                raise
            d_end = min(CUROBO.check_path([q], n=1)[0]["table/arm"][0] for q in (q_from, q_to))
            vias = CUROBO.detour(q_from, q_to, table_margin=min(CUROBO.TABLE_MARGIN, d_end - 0.01), floor=floor_line(q_from, q_to))
            if not vias:
                raise SystemExit(f"{e}（回り込む経路も作れない）")
            print(f"[run] {name}: 直線は当たるので回り込む（経由点 {len(vias)} 個）", flush=True)
            prev = q_from
            for k, v in enumerate(vias):
                check(prev, v, f"{name} 回り込み {k + 1}/{len(vias)}")
                prev = v
            return [list(v) for v in vias]

    pre = grasp = lift = carry = place = None  # 計画の姿勢（load_plan が入れる。--early では計画ができてから）
    import pinocchio as pin

    def raised_or_same(qd):
        """Raise for floor clearance when possible; the home pose is already high (and 8 cm more is out of reach)."""
        for dz in (0.08, 0.05, 0.03):
            try:
                return raised(qd, dz)
            except SystemExit:
                pass
        return list(qd)

    def raised_some(qd):
        """休みの姿勢から持ち上げる: --raise-m（既定 8 cm）で解けなければ 8・5・3 cm（電源を入れ直した後など、休みの姿勢が少し違うとき）。"""
        for dz in sorted({a.raise_m, 0.08, 0.05, 0.03}, reverse=True):
            if dz > a.raise_m:
                continue
            try:
                return raised(qd, dz)
            except SystemExit:
                pass
        raise SystemExit("[run] can't solve the lift (8/5/3 cm) from the start pose. Stopping")

    def raised(qd, dz=0.08):
        """Pose with the TCP raised dz straight up while keeping its orientation (lift off the floor first when leaving the resting pose)."""
        M = ik.fk(np.deg2rad(qd)); T = pin.SE3(M.rotation, M.translation + np.array([0, 0, dz]))
        q, ok, _, _ = ik.solve(T, np.deg2rad(qd), position_only=False, max_iter=300)
        if not ok or not ik.within_limits(q):
            raise SystemExit("[run] can't solve the lift from the start pose. Stopping")
        return list(np.rad2deg(q))

    def fk44(qd):
        M = ik.fk(np.deg2rad(qd)); T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation; return T

    def solve44(T, seed):
        """link7 の姿勢 T を解く。seed から解けなければ pregrasp から解き直す（2026-11-23: 掴む姿勢から解き始めると、
        9.6 cm 戻した測る所が関節の範囲を外れた・cap_resident_20261123_120003）。解き直した答えは seed から 45° 以内のときだけ使う
        （物の近くで別の枝に跳ばない）。"""
        for s in (seed, pre):
            q, ok, _, _ = ik.solve(pin.SE3(T[:3, :3], T[:3, 3]), np.deg2rad(s), position_only=False, max_iter=300)
            if ok and ik.within_limits(q) and (s is seed or np.max(np.abs(np.rad2deg(q) - np.asarray(seed))) <= 45.0):
                return list(np.rad2deg(q))
        return None

    servo_obj = servo_near = servo_near_q = None  # (物の中心（付け根の座標）, 水平の大きさ)・--wrist-servo のときだけ
    if a.wrist_servo:
        if side != "left":
            sys.exit("[run] --wrist-servo は左腕だけ（手首のカメラは左のグリッパーにある・2026-11-23）")
        if not a.wrist_jaw or not os.path.exists(a.wrist_jaw):
            sys.exit(f"[run] --wrist-servo の爪の模型が無い: {a.wrist_jaw}（GRASP_WRIST_JAW か --wrist-jaw）")
        from grasp.wrist_servo import JawModel
        from grasp.wrist_approach import ServoAbort, servo_approach, AX_L7_G, look_back
        servo_jaw = JawModel.load(a.wrist_jaw)

    def setup_servo():
        """--wrist-servo の、計画に頼る部分（物の中心・最初に測る所）。"""
        nonlocal servo_obj, servo_near, servo_near_q
        if not a.wrist_servo:
            return
        T_bc_s = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
        ob_s = np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc_s[:3, :3].T + T_bc_s[:3, 3]
        from grasp.wrist_approach import fits_between_jaws
        if not fits_between_jaws(ob_s):  # 縁や壁を掴む太い物: 物全体を真ん中へ寄せると掴む所がずれる。計画どおり近づく
            print("[run] --wrist-servo: 物が爪の間に丸ごと入らない太さなので、手首のカメラで合わせずに計画どおり近づく", flush=True)
            a.wrist_servo = False
            return
        servo_obj = (ob_s.mean(0), float(np.max(np.ptp(ob_s[:, :2], axis=0))))
        # 最初に測る所: 爪の先が物の上の面より 2.5 cm 上に来るまで戻す（--servo-near は下限・2026-11-23 の 1 回目で測る前に当たった）
        servo_near = look_back(fk44, grasp, float(np.percentile(ob_s[:, 2], 98)), near_min=a.servo_near)
        Tn = fk44(grasp); Tn[:3, 3] -= servo_near * (Tn[:3, :3] @ AX_L7_G)[:, 2]
        # 掴む姿勢から解き始めると関節の範囲を外れることがある（2026-11-23 cap_resident_20261123_120003: 9.6 cm 戻す所）。
        # 解けなければ pregrasp（同じ向きで 10 cm 戻した所）から解き直す
        servo_near_q = solve44(Tn, grasp)  # 解けなければ solve44 が pregrasp から解き直す
        if servo_near_q is None:
            sys.exit(f"[run] --wrist-servo: 掴む姿勢から {servo_near * 100:.1f} cm 戻した姿勢が解けない")

    def load_plan(p):
        """計画（plan.json の中身）の姿勢を入れる。--early では、裏で計画ができてから呼ぶ。"""
        nonlocal pre, grasp, lift, carry, place
        pre, grasp, lift = p["q_pre_deg"], p["q_grasp_deg"], p["q_lift_deg"]
        carry, place = p.get("q_carry_deg"), p.get("q_place_deg")
        setup_servo()

    if not a.early:
        load_plan(plan)

    # cuRobo の plan_grasp（2026-11-23 Munenori）: raise → pregrasp を、拾う物も障害物に入れて cuRobo の細かい軌道で動かす。
    # 前の route は cuRobo の経路を関節の直線にまとめ直していて、物のそばで関節が 129° 回る区間ができ、手が丸太に触れた
    # （cap_resident_20261123_121931）。差し込み（pregrasp → 掴む）と持ち上げも手先の直線に沿う細かい軌道にする。
    # 帰りは行きの軌道を逆にたどる（離した後は物は同じ所にある）。--legacy-approach か cuRobo の無い置き場は前の動き
    class Dense(list):
        """cuRobo の細かい軌道（始点を含む）。go_route が止まらずにたどる（grasp/motion.py の follow_path）。"""

    def check_dense(path, name, allow_low=False):
        """細かい軌道を 2° ごとに間引いて check（床・keep-out・cuRobo）を通す。"""
        P = [path[0]]
        for q in path[1:-1]:
            if max(abs(x - y) for x, y in zip(q, P[-1])) >= 2.0:
                P.append(q)
        P.append(path[-1])
        for k in range(len(P) - 1):
            check(P[k], P[k + 1], f"{name} {k + 1}/{len(P) - 1}", allow_low=allow_low, quiet=True)
        step = max(max(abs(x - y) for x, y in zip(p, q)) for p, q in zip(path[:-1], path[1:])) if len(path) > 1 else 0.0
        span = max(abs(x - y) for x, y in zip(path[0], path[-1]))
        print(f"[run] check ok: {name}（cuRobo の細かい軌道 {len(path)} 点・端から端の関節 最大 {span:.0f}°・"
              f"隣の点の差 最大 {step:.1f}°）", flush=True)

    use_gp = CUROBO is not None and not a.legacy_approach

    def grasp_plan(q_up):
        """q_up（raise）から plan_grasp で近づく・差し込む・持ち上げるの軌道を作って確かめ、pre・grasp・lift をその終点にする。"""
        nonlocal pre, grasp, lift, servo_near_q
        need = ["approach", "grasp"] + ([] if a.wrist_servo else ["lift"])
        # 計画の箱は確かめの箱（--target-pad）より大きくする。2026-11-23 cap_resident_20261123_121931 で、余白 1 cm だと
        # 丸太の 5 cm 以内で関節が 63〜73° 回り、3 cm だと 2〜3°（大きな振りを丸太から離れた所で済ませる）。4 cm は pregrasp が入り作れない
        pads = [a.plan_target_pad] + [p for p in (0.02, a.target_pad) if p < a.plan_target_pad] if a.avoid_target else [None]
        for k, pad in enumerate(pads):
            tb = target_box(pad, quiet=True) if pad is not None else None
            r = CUROBO.plan_grasp(q_up, fk44(grasp), approach=a.pregrasp_m, lift=0.0 if a.wrist_servo else a.lift_m, target=tb)
            if r["success"] and all(r[n] is not None for n in need):
                if pad is not None:
                    print(f"[run] plan_grasp: 拾う物の箱の余白 {pad * 100:.0f} cm で計画した"
                          + ("" if k == 0 else f"（{a.plan_target_pad * 100:.0f} cm では作れず下げた。物のそばで大きく回るかもしれない）"), flush=True)
                break
            print(f"[run] plan_grasp: 余白 {(pad or 0) * 100:.0f} cm では作れない（{r['status']}）", flush=True)
        else:
            raise SystemExit(f"[run] cuRobo の plan_grasp で掴む動きを作れない（{r['status']}）。動かさない")
        A, G, Lf = r["approach"], r["grasp"], r["lift"]
        if max(abs(x - y) for x, y in zip(A[0], q_up)) > 1.0:
            raise SystemExit("[run] plan_grasp の軌道が raise の姿勢から始まっていない。動かさない")
        avoid_target(True)  # 近づく区間は拾う物もよける（こちらの判定でも確かめる）
        try:
            check_dense(A, "raise→pregrasp（plan_grasp）")
        finally:
            avoid_target(False)
        check_dense(G, "pregrasp→grasp（plan_grasp）", allow_low=True)
        if Lf:
            check_dense(Lf, "grasp→lift（plan_grasp）", allow_low=True)
        pre, grasp = list(A[-1]), list(G[-1])
        if Lf:
            lift = list(Lf[-1])
        else:  # wrist servo: 持ち上げは直した後に解き直す。確かめの線だけ今の掴む姿勢から作る
            for dz in (0.10, 0.08, 0.05):
                try:
                    lift = raised(grasp, dz); break
                except SystemExit:
                    pass
        if a.wrist_servo:  # 測る所を新しい pregrasp から解き直す（前の掴む姿勢の枝のまま解くと、物のそばで関節が跳ぶ）
            Tn2 = fk44(grasp); Tn2[:3, 3] -= servo_near * (Tn2[:3, :3] @ AX_L7_G)[:, 2]
            servo_near_q = solve44(Tn2, pre)
            if servo_near_q is None:
                raise SystemExit(f"[run] --wrist-servo: 掴む姿勢から {servo_near * 100:.1f} cm 戻した姿勢が解けない（plan_grasp の後）")
        return {"up_pre": Dense(A), "pre_up": Dense(A[::-1]), "pre_grasp": Dense(G), "grasp_pre": Dense(G[::-1]),
                "grasp_lift": Dense(Lf) if Lf else None, "lift_grasp": Dense(Lf[::-1]) if Lf else None}

    gp = None
    home = None
    if a.home:
        import yaml
        hy = yaml.safe_load(open(a.home)); home = [float(hy[j]) for j in J]
    if not a.go:
        start = [float(v) for v in a.start.split(",")] if a.start else plan["q_start_deg"]
        load_obstacles(start)
        avoid_target(True)  # 移動の区間: 拾う物もよける
        if home is not None:  # rest -> raise -> home, then home is the start of the grasp sequence
            if max(abs(x - y) for x, y in zip(start, home)) >= 5.0:  # already held at home: nothing to check
                up0 = raised(start); route(start, up0, "rest→raise"); route(up0, home, "raise→home")
                route(home, up0, "home→raise over rest")
            start = home
        up = raised_or_same(start) if home is not None else raised_some(start)
        route(start, up, "start→raise"); route(up, start, "raise→start")
        if use_gp:
            gp = grasp_plan(up)  # 帰りは行きの逆なので同じ確かめで足りる
        else:
            route(up, pre, "raise→pregrasp"); route(pre, up, "pregrasp→raise")
        avoid_target(False)  # ここから物の近くの区間と運ぶ区間
        if gp is None:
            check(pre, grasp, "pregrasp→grasp", allow_low=True)
        if a.wrist_servo:
            check(pre, servo_near_q, "pregrasp→servo near", allow_low=True)
            print(f"[run] wrist servo: 物 {np.round(servo_obj[0], 3).tolist()}・大きさ {servo_obj[1] * 100:.1f} cm・"
                  f"測る所は掴む姿勢から {servo_near * 100:.1f} cm 戻した所・"
                  f"近い所 " + " ".join(f"{v:+.1f}" for v in servo_near_q))
        if gp is None or gp["grasp_lift"] is None:
            check(grasp, lift, "grasp→lift", allow_low=True)
        route(lift, pre, "lift→pregrasp")
        if carry is not None:
            route(lift, carry, "lift→carry"); check(carry, place, "carry→place", allow_low=True)
            check(place, carry, "place→carry", allow_low=True); route(carry, up, "carry→raise")
            print("[run] place at the new spot " + " ".join(f"{v:+.1f}" for v in place))
        print("[run] raise pose " + " ".join(f"{v:+.1f}" for v in up))
        print("[run] DRY RUN (add --go to move)"); return 0

    from lerobot.robots.openarm_follower import OpenArmFollower, OpenArmFollowerConfig
    from lerobot.motors.damiao.damiao_alignment import soft_move_to_position
    from grasp.motion import smooth_move_to_position
    from grasp.gravity import gravity_torque_fn
    grav = gravity_torque_fn(side)  # 手と爪の質量入り（2026-11-23）
    if side == "left":
        port, rid = os.environ.get("GRASP_CAN_PORT", "can_fol_l"), os.environ.get("GRASP_ROBOT_ID", "follower_left")
    else:  # 右は既定の値を持たない（上で無ければ止めている）
        port, rid = os.environ["GRASP_CAN_PORT_RIGHT"], os.environ["GRASP_ROBOT_ID_RIGHT"]
    print(f"[run] 右腕の CAN {port}・較正 {rid}" if side == "right" else f"[run] 左腕の CAN {port}・較正 {rid}", flush=True)
    own_robot = robot is None
    if own_robot:
        robot = OpenArmFollower(OpenArmFollowerConfig(port=port, side=side, id=rid, align_on_connect=False, cameras={},
                                                      disable_torque_on_disconnect=not (a.hold_home and a.home)))
        robot.connect()
    else:  # 常駐のプロセスが接続したまま渡したロボット（休みの姿勢で脱力している）。connect と同じくモーターを入れる
        robot.bus.enable_torque()

    def release():
        """終わり: 自分で接続したなら切る（設定どおり脱力）。渡されたロボットは接続を残して脱力だけする（--hold-home は除く）。"""
        if own_robot:
            robot.disconnect()
        elif not (a.hold_home and a.home):
            robot.bus.disable_torque()
    obs = robot.get_observation(); start = [float(obs[f"{j}.pos"]) for j in J]
    print("[run] 今の腕の角度 " + ",".join(f"{v:.1f}" for v in start), flush=True)
    log = open(os.path.join(a.cap_dir, "traj.csv"), "w", newline=""); w = csv.writer(log)
    w.writerow(["t", "phase"] + [f"{j}.pos" for j in J] + ["gripper.pos"])

    def rec(phase):
        o = robot.get_observation(); w.writerow([f"{time.time():.3f}", phase] + [f"{o[f'{j}.pos']:.2f}" for j in J] + [f"{o['gripper.pos']:.2f}"])

    def go(q, speed, g, phase, correct=True):
        goal = {j: float(q[i]) for i, j in enumerate(J)}; goal["gripper"] = float(g)
        t_go = time.monotonic()  # 区間ごとの時間の内訳（2026-11-23: wrist servo の 1 段が 8 秒かかった理由を見る）
        if a.smooth:  # 出だしと終わりをなだらかに（grasp/motion.py・2026-11-22）。一番速い所が speed になる
            smooth_move_to_position(robot.bus, goal, speed, torque_ff_fn=grav)
        else:
            soft_move_to_position(robot.bus, goal, torque_ff_fn=grav, max_speed_dps=speed)
        # The soft gains don't fully reach the target (first real run: J7 14.5° short, TCP 5 cm short).
        # Add the remaining error to the command and send it again (up to 3 times, at most 15° of added offset per joint).
        cmd = dict(goal); err = None; t_mv = time.monotonic(); n_fix = 0
        for _ in range(3 if correct else 0):
            o = robot.get_observation()
            err = {j: goal[j] - float(o[f"{j}.pos"]) for j in J}
            if max(abs(v) for v in err.values()) < 1.0:
                break
            for j in J:
                cmd[j] = goal[j] + float(np.clip(cmd[j] - goal[j] + err[j], -15, 15))
            soft_move_to_position(robot.bus, cmd, duration_s=a.fix_s, torque_ff_fn=grav); n_fix += 1
        age = max(time.monotonic() - robot.bus._last_update_ts.get(j, 0.0) for j in J)
        if age > 0.3:  # 返事が無いまま「done」と書かない（2026-11-21 18:29: 非常停止の後に置く・離すが done になった）
            raise MotorLost(f"[run] {phase}: モーターの返事が {age:.1f} 秒来ない（非常停止・電源・通信）。腕に指令を送るのをやめる")
        rec(phase)
        o = robot.get_observation(); e = max(abs(goal[j] - float(o[f"{j}.pos"])) for j in J)
        t_end = time.monotonic()
        print(f"[run] {phase} done (max joint error {e:.1f}°・動き {t_mv - t_go:.1f} 秒・補正 {n_fix} 回 {t_end - t_mv:.1f} 秒)", flush=True)

    def replan_place_from_grasp(carry_q, place_q, r_lc, r_cu):
        """閉じた直後の実際の角度で、運ぶ・置く姿勢を解き直す（2026-11-22）。爪を閉じると手首（柔らかい制御）が木材に回され、
        手の中の物の位置が計画とずれる（J7 2.6° で置き先が 15 mm ずれた・cap_resident_20261122_164524）。物は机の上で動かず手が動いたと
        見て、link7 の置く姿勢を L7_置く(計画) · L7_掴む(計画)^-1 · L7_掴む(読み) にする。解けない・確かめを通らないときは計画のまま。"""
        o = robot.get_observation()
        q_read = np.array([float(o[f"{j}.pos"]) for j in J])
        d = np.max(np.abs(q_read - np.asarray(grasp)))
        if d < 0.5:
            return carry_q, place_q, r_lc, r_cu

        def l7(qd):
            M = ik.fk(np.deg2rad(qd)); T = np.eye(4); T[:3, :3] = M.rotation; T[:3, 3] = M.translation
            return T
        D = np.linalg.inv(l7(grasp)) @ l7(q_read)  # 計画の掴む姿勢から、実際の掴む姿勢への手の動き（手の座標）
        new = []
        for q_old in (carry_q, place_q):
            Tn = l7(q_old) @ D
            q, ok, _, _ = ik.solve(pin.SE3(Tn[:3, :3], Tn[:3, 3]), np.deg2rad(q_old), position_only=False, max_iter=300)
            qd = np.rad2deg(q)
            if not ok or not ik.within_limits(q) or np.max(np.abs(qd - np.asarray(q_old))) > 15:
                print(f"[run] 置き先の解き直し: 解けないので計画のまま（閉じた後の差 最大 {d:.1f}°）", flush=True)
                return carry_q, place_q, r_lc, r_cu
            new.append(list(qd))
        try:
            r1 = route(lift, new[0], "lift→carry（解き直し）"); check(new[0], new[1], "carry→place（解き直し）", allow_low=True)
            check(new[1], new[0], "place→carry（解き直し）", allow_low=True); r2 = route(new[0], up, "carry→raise（解き直し）")
        except SystemExit as e:
            print(f"[run] 置き先の解き直し: 確かめを通らないので計画のまま（{e}）", flush=True)
            return carry_q, place_q, r_lc, r_cu
        moved = (l7(new[1]) @ np.linalg.inv(l7(place_q)))[:3, 3]
        print(f"[run] 置き先の解き直し: 閉じた後の差 最大 {d:.1f}° → 置く姿勢の手を {np.round(moved * 1000).tolist()} mm 動かした", flush=True)
        return new[0], new[1], r1, r2

    def plan_return(q_put):
        """置いた後の戻り（2026-11-23）: 爪の先が置いた物の上端 + 3 cm に来るまで（少なくとも 6 cm）真上へ上げる姿勢と、そこから
        raise までの経路（置いた物・置く面の箱も障害物に入れ、当たれば cuRobo で回り込む）。作れなければ raise の高さまで真上へ
        上げてから戻る。それも作れなければ今までどおり直線（試行は止めない・置き終えた後なので）。"""
        try:
            M = fk44(q_put)
            tip_z = float((M[:3, 3] + M[:3, :3] @ [0, 0, TIP_LEN])[2]) - 0.02  # 爪の先の角（傾き）の分を見込む
            placed = None
            tb = target_box(quiet=True)
            if tb is not None:
                G0, P0 = fk44(grasp), M
                delta = (P0[:3, 3] + P0[:3, :3] @ [0, 0, TCP_LEN]) - (G0[:3, 3] + G0[:3, :3] @ [0, 0, TCP_LEN])
                placed = shift_box(tb, delta)
            top = (placed["pose"][2] + placed["dims"][2] / 2) if placed else tip_z
            dz_need = clear_lift_dz(tip_z, top)
            dz, q_up = None, None
            # 必要な高さから 2 cm ずつ下げて、解ける一番高い所まで上げる（少なくとも 6 cm）
            for dz_try in sorted({round(v, 3) for v in np.arange(dz_need, 0.06 - 1e-9, -0.02)} | {0.06}, reverse=True):
                q_try = up_from(q_put, dz_try)
                if q_try is not None:
                    dz, q_up = dz_try, q_try
                    break
            if q_up is None:
                q_up = list(carry)  # 真上の姿勢が解けない: 今までの運ぶ姿勢（置く姿勢の上）
            check(q_put, q_up, "置いた後に真上へ", allow_low=True, quiet=True)
            boxes = list(tgt["seen"]) + ([placed] if placed else []) + surface_boxes()
            how = ""
            if CUROBO is not None and boxes:
                CUROBO.set_obstacle_boxes(boxes)
            try:
                try:
                    rt = route(q_up, up, "置いた後の戻り（置いた物をよける）")
                    how = "置いた物をよける（直線で通る）" if len(rt) == 1 else f"置いた物をよけて回り込む（{len(rt)} 区間）"
                except SystemExit as e:
                    dz2 = tcp_z(up) - tcp_z(q_up)
                    try:
                        if dz2 <= 0.01:
                            raise SystemExit("上げ足せない")
                        q_hh = raised(q_up, dz2)
                        check(q_up, q_hh, "置いた後に raise の高さまで", allow_low=True, quiet=True)
                        rt = [q_hh] + route(q_hh, up, "置いた後の戻り（raise の高さから）")
                        how = f"raise の高さまで {dz2 * 100:.0f} cm 上げてから戻る（{e}）"
                    except SystemExit as e2:
                        rt = [list(up)]
                        how = f"置いた物をよける経路が作れない（{e2}）。今までどおり直線で戻る"
            finally:
                if CUROBO is not None and boxes:
                    CUROBO.set_obstacle_boxes(list(tgt["seen"]))
            print(f"[run] 置いた後の戻り: 真上へ {('—' if dz is None else f'{dz * 100:.1f} cm')}（要る量 {dz_need * 100:.1f} cm・"
                  f"物の上端 {'不明' if not placed else f'{top:.3f}'}）・{how}", flush=True)
            return {"q_up": q_up, "route": rt}
        except BaseException as e:  # noqa: BLE001  表のスレッドで受ける
            return {"err": e}

    def up_from(q_from, dz, max_change=40.0):
        """q_from の手先を向きを保って真上へ dz 上げた姿勢。Pinocchio の IK（q_from から）で解けなければ cuRobo の IK で解く
        （構えの違う解しか無いことがある・2026-11-23 重量計の上に置いた姿勢）。関節の変化が max_change [deg] を超える解は使わない
        （物のすぐ上で腕が振れる）。解けなければ None。"""
        try:
            q = raised(q_from, dz)
            if np.max(np.abs(np.asarray(q) - np.asarray(q_from))) <= max_change:
                return q
        except SystemExit:
            pass
        if CUROBO is None or not hasattr(CUROBO, "ik"):
            return None
        T = fk44(q_from); T[2, 3] += dz
        try:
            Q, OK = CUROBO.ik(T[None], seed_deg=np.asarray(q_from, float)[None], return_seeds=8)
        except Exception:  # noqa: BLE001  IK を持たないサーバー（--no-ik）
            return None
        best = None
        for qd, ok in zip(Q[0], OK[0]):
            if ok and ik.within_limits(np.deg2rad(qd)):
                dmax = float(np.max(np.abs(qd - np.asarray(q_from))))
                if dmax <= max_change and (best is None or dmax < best[0]):
                    best = (dmax, list(qd))
        return None if best is None else best[1]

    def surface_boxes():
        """置く面（計画の常駐が <回>/place_target.json に書いた surface_box [x0, x1, y0, y1, 上面 z]）の箱。無ければ空。"""
        f = os.path.join(a.cap_dir, "place_target.json")
        try:
            sb = json.load(open(f)).get("surface_box") if os.path.exists(f) else None
        except (OSError, ValueError):
            sb = None
        if not sb:
            return []
        x0, x1, y0, y1, zt = (float(v) for v in sb)
        return [{"name": "surface", "dims": [x1 - x0, y1 - y0, max(zt - FLOOR_Z, 0.005)],
                 "pose": [(x0 + x1) / 2, (y0 + y1) / 2, (zt + FLOOR_Z) / 2, 1.0, 0.0, 0.0, 0.0]}]

    def go_route(path, speed, g, phase):
        """route の姿勢の列を順に動かす。位置の補正（足りない分の送り直し）は終点でだけかける（2026-11-22: 経由点ごとに止まって
        送り直すと、かくかく動いた。経由点の間も当たりを確かめてあり、cuRobo の余裕 10 mm の中で通る）。--correct-vias で前の動き"""
        if isinstance(path, Dense):  # cuRobo の細かい軌道: 止まらずにたどり、終点で今までどおり位置を補正する
            from grasp.motion import follow_path
            follow_path(robot.bus, J, path, speed, gripper_deg=g, torque_ff_fn=grav)
            go(path[-1], speed, g, phase)
            return
        for k, q in enumerate(path):
            last = k == len(path) - 1
            go(q, speed, g, phase if last else f"{phase} (回り込み {k + 1}/{len(path) - 1})", correct=last or a.correct_vias)

    def wrist_serial():
        """その腕の手首のカメラ（右は GRASP_WRIST_SERIAL_RIGHT。無ければ開かない）。"""
        if side == "right":
            if not os.environ.get("GRASP_WRIST_SERIAL_RIGHT"):
                raise RuntimeError("右の手首のカメラ（GRASP_WRIST_SERIAL_RIGHT）が無い")
            return os.environ["GRASP_WRIST_SERIAL_RIGHT"]
        return os.environ.get("GRASP_WRIST_SERIAL", "409122272149")

    watcher = None
    wcam = None
    g_now = float(obs["gripper.pos"])
    TR, NR = a.transit_dps, a.near_dps
    GR = a.gripper_dps if a.gripper_dps else max(NR, 15)  # グリッパーだけを開閉する段
    LR, CR, PR = (a.lift_dps or NR), (a.carry_dps or NR), (a.place_dps or NR)  # 持ち上げる・運ぶ・置く（2026-11-23 速くする）
    rest = list(start)  # where the arm hung limp before the trial: go back here before torque off
    at_home = home is not None and max(abs(x - y) for x, y in zip(start, home)) < 5.0
    s0 = home if home is not None else start
    up = r_start_up = r_up_start = r_up_pre = r_pre_up = r_lift_carry = r_carry_up = None

    def prep_plan():
        """計画に頼る動く前の準備（カメラで見た物・拾う物の箱・近づく軌道・運ぶ区間の経路）。--early では裏のスレッドで呼ぶ。"""
        nonlocal wcam, gp, r_up_start, r_up_pre, r_pre_up, r_lift_carry, r_carry_up
        if a.wrist_servo:  # 動く前に開く（開けなければ動かさない）
            from grasp.wrist import WristCam
            wcam = WristCam(wrist_serial())
        load_obstacles(start)
        avoid_target(True)  # 移動の区間: 拾う物もよける（運ぶ区間の前で外す）
        # --early の戻り（raise→start）は、上がってきた真上の線を逆にたどる（物はその間に動いていない。2026-11-23: 休みの姿勢の
        # 爪のすぐ横に物があると、休みの姿勢そのものが拾う物の箱（余白込み）に入り、物を入れた経路は作れない）
        if use_gp:
            gp = grasp_plan(up)
            r_up_pre, r_pre_up = gp["up_pre"], gp["pre_up"]
        else:
            r_up_pre = route(up, pre, "raise→pregrasp"); r_pre_up = route(pre, up, "pregrasp→raise")
        avoid_target(False)  # 運ぶ区間は物が手と動く。物の近くの区間（check）も、これより後は物を障害物に入れない
        if carry is not None:
            r_lift_carry = route(lift, carry, "lift→carry"); r_carry_up = route(carry, up, "carry→raise")

    def early_start():
        """--early: 真上へ上げて爪を開く間に、裏のスレッドで計画を待ち、計画に頼る準備を済ませる。
        計画が立たない・計画の腕が違う・確かめを通らない・Ctrl-C なら、物の近くへは行かずに休みの姿勢へ戻って脱力する。"""
        import threading
        res = {}

        def background():
            try:
                ok = plan_wait()
                if not ok:
                    raise SystemExit("[run] 計画が立たなかった")
                with open(os.path.join(a.cap_dir, "plan.json")) as f:
                    p = json.load(f)
                if p.get("arm", "left") != side:
                    raise SystemExit(f"[run] 計画の腕（{p.get('arm', 'left')}）が上げ始めた腕（{side}）と違う")
                load_plan(p)
                prep_plan()
                res["ok"] = True
            except BaseException as e:  # noqa: BLE001  ここの例外は表のスレッドで受けて戻る
                res["err"] = e

        old_iv = sys.getswitchinterval()
        sys.setswitchinterval(0.001)  # 裏の計算が、10 ms ごとのモーターへの指令を待たせないように
        th = threading.Thread(target=background, name="early-prep", daemon=True)
        th.start()
        try:
            if t_trigger is not None:
                print(f"[run] 合図から動き出しまで {time.monotonic() - t_trigger:.2f} 秒", flush=True)
            go_route(r_start_up, TR, g_now, "raise")
            go(up, TR, G_OPEN, "open gripper")
            t_w = time.monotonic()
            th.join()
            if "ok" not in res:
                raise SystemExit(f"{res.get('err')}" if res.get("err") is not None else "[run] 計画の準備が終わらない")
            if time.monotonic() - t_w > 0.05:
                print(f"[run] 計画と確かめを raise で {time.monotonic() - t_w:.1f} 秒待った", flush=True)
        except BaseException as e:  # noqa: BLE001
            sys.setswitchinterval(old_iv)
            if type(e).__name__ == "MotorLost":  # 非常停止・電源断: 戻る動きもしない
                print(f"{e}。戻る動きはしない。腕を支えて電源を確かめる", flush=True)
                release(); log.close()
                raise
            print(f"[run] 計画・確かめが通らない・止めた（{e or type(e).__name__}）。物の近くへは行かずに休みの姿勢へ戻る", flush=True)
            try:
                o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
                if max(abs(x - y) for x, y in zip(q_now, up)) < 3.0:
                    back = r_up_start  # raise にいる: 作っておいた戻りの経路（裏で作り直していればそれ）
                else:  # 上げる途中で止めた: 真上の線の上にいるので、休みの姿勢へ直線で戻る
                    back = [list(start)]
                go_route(back, min(TR, 20.0) if isinstance(e, KeyboardInterrupt) else TR, g_now, "back to start")
            finally:
                if wcam is not None:
                    wcam.close()
                release(); log.close()
                print("[run] back at start, torque off（物の近くへは行っていない）", flush=True)
            return 1
        sys.setswitchinterval(old_iv)
        return 0

    if a.early:
        # 計画に頼らない区間だけ（休みの姿勢 → 真上 → 休みの姿勢）を、置き場の物（本体・カメラ・机・もう片方の腕）と床で確かめる。
        # カメラで見た物と拾う物はまだ無い。真上へ 8 cm 上げる動きで、前に置いた物には近づかない
        try:
            if CUROBO is not None:
                CUROBO.set_obstacle_boxes([])
            up = raised_some(start)
            # 爪を真上へまっすぐ上げる細かい軌道（高さを 8 等分して IK で解き、止まらずにたどる）。関節の直線だと 14 cm で爪が横へ
            # 5.4 mm 振れ、休みの姿勢の爪から 1 cm の所に立てた木材を掠りうる（2026-11-23）。戻りも同じ線を逆にたどる
            M0 = ik.fk(np.deg2rad(start)); dz_up = float(ik.fk(np.deg2rad(up)).translation[2] - M0.translation[2])
            line, prev = [list(start)], np.deg2rad(start)
            for f in np.linspace(0.0, 1.0, 9)[1:]:
                q, ok, _, _ = ik.solve(pin.SE3(M0.rotation, M0.translation + np.array([0.0, 0.0, dz_up * f])), prev,
                                       position_only=False, max_iter=300)
                if not ok or not ik.within_limits(q) or np.max(np.abs(np.rad2deg(q - prev))) > 30.0:
                    raise SystemExit(f"[run] 真上へ上げる線が解けない（{dz_up * f * 100:.1f} cm）。動かさない")
                line.append(list(np.rad2deg(q))); prev = q
            up = line[-1]
            check_dense(line, "start→raise（計画の前・真上の線）")
            r_start_up, r_up_start = Dense(line), Dense(line[::-1])
        except BaseException:
            print("[run] 動く前に止まった（腕は動かしていない）", flush=True)
            release(); log.close()
            raise
        if early_start() != 0:
            return 1
    if not a.early:
        try:  # 動く前の準備（経路を作る・カメラの背景）。ここで止まったら、腕を動かさずにモーターとの接続を切る
            if home is not None and not at_home:
                up0 = raised(start); r_rest_up = route(start, up0, "rest→raise"); r_up_home = route(up0, home, "raise→home")
                r_home_up0 = route(home, up0, "home→raise over rest")
            # 動く前に移動の区間の経路を全部作る（回り込みを作れない区間があれば、腕を動かす前に止まる）
            up = raised_or_same(s0) if home is not None else raised_some(s0)
            r_start_up = route(s0, up, "start→raise"); r_up_start = route(up, s0, "raise→start")
            prep_plan()
            watcher = None
            if a.avoid:
                if CUROBO is None or carry is None:
                    raise SystemExit("[run] --avoid には GRASP_CUROBO と置き先（grasp/tools/plan_place.py）が要る")
                from grasp import reactive
                T_bc = np.array(json.load(open(os.environ["GRASP_T_BASE_CAM"]))["T_base_cam"])
                watcher = reactive.Watcher(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"), T_bc, FLOOR_Z)
                watcher.start()  # 動き出す前に背景（机の上の様子）を覚える。この間は何も割り込ませない
                print("[run] 胸のカメラで背景を覚える（約 2 秒・割り込ませない）", flush=True)
                t0 = time.time()
                while watcher.stamp == 0.0 and time.time() - t0 < 8.0 and watcher.error is None:
                    time.sleep(0.05)
                if watcher.stamp == 0.0:
                    raise SystemExit(f"[run] 胸のカメラの映像が来ない（{watcher.error}）。動かさない")

                # 持っているアルミの点（掴んだ姿勢の link7 の座標）。運ぶ間は手と一緒に動かし、そのまわりの点を割り込んだ物と見ない
                held_l7 = reactive.held_in_link7(np.load(os.path.join(a.cap_dir, "object.npy")) @ T_bc[:3, :3].T + T_bc[:3, 3], grasp)

                def carry_watch(path, g, phase, holding_now=False):
                    reactive.follow(robot.bus, grav, path, a.avoid_dps, g, watcher, T_bc[:3, 3], FLOOR_Z, floor_line, phase,
                                    held_l7=held_l7 if holding_now else None)
        except BaseException:
            print("[run] 動く前に止まった（腕は動かしていない）。接続を切る", flush=True)
            if watcher is not None:
                watcher.stop()
            if wcam is not None:
                wcam.close()
            release(); log.close()
            raise
    done = holding = motor_lost = False
    try:
        if home is not None:  # yesterday's start pose for every trial (Sota 11/20)
            if not at_home:
                go_route(r_rest_up, TR, g_now, "raise from rest")
            go_route(r_up_home, TR, g_now, "home") if not at_home else go(home, TR, g_now, "home")
            start = home
        if not a.early:  # --early では early_start が済ませた
            go_route(r_start_up, TR, g_now, "raise")
            go(up, TR, G_OPEN, "open gripper")
        go_route(r_up_pre, TR, G_OPEN, "pregrasp")
        if a.wrist_servo:
            try:
                def _timed_check(q0, q1, name, low=False):
                    t_c = time.monotonic(); check(q0, q1, name, allow_low=low)
                    print(f"[run] {name}: 確かめ {time.monotonic() - t_c:.2f} 秒", flush=True)
                grasp = servo_approach(grasp_q=grasp, pre_q=pre, fk=fk44, solve=solve44, go=go,
                                       check=_timed_check,
                                       cam=_SettleCam(wcam, a.servo_settle), jaw=servo_jaw, obj_base=servo_obj[0], obj_extent=servo_obj[1], g_open=G_OPEN,
                                       out_dir=a.cap_dir, near=servo_near, depth_target=a.servo_depth, speed=min(NR, a.servo_dps),
                                       step_m=a.servo_step, t_shift=a.servo_t_shift, one_shot=a.servo_one_shot,
                                       log=lambda m: print(m, flush=True))
            except (ServoAbort, SystemExit, RuntimeError) as e:  # RuntimeError: 手首のカメラのフレームが来ない
                if type(e).__name__ == "MotorLost":
                    raise
                print(f"[run] wrist servo を止めた: {e}。pregrasp に戻してから戻る", flush=True)
                go(pre, NR, G_OPEN, "back to pregrasp (servo)")
                raise KeyboardInterrupt
            lift = None
            for dz in (0.10, 0.08, 0.05):  # 直した後の姿勢から真上に持ち上げる姿勢を解き直す
                try:
                    lift = raised(grasp, dz); check(grasp, lift, f"servo grasp→lift {dz * 100:.0f} cm", allow_low=True); break
                except SystemExit:
                    lift = None
            if lift is None:
                print("[run] wrist servo: 直した後の持ち上げが解けない。閉じずに pregrasp に戻す", flush=True)
                go(pre, NR, G_OPEN, "back to pregrasp (servo)")
                raise KeyboardInterrupt
        else:
            go_route(gp["pre_grasp"], NR, G_OPEN, "approach") if gp else go(grasp, NR, G_OPEN, "approach")
        if a.probe:
            import cv2, pyrealsense2 as rs
            time.sleep(1.0)
            pipe = rs.pipeline(); rc = rs.config(); rc.enable_device(os.environ.get("GRASP_CHEST_SERIAL", "344422070898"))
            rc.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rc.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            prof = pipe.start(rc); sc = prof.get_device().first_depth_sensor().get_depth_scale(); al = rs.align(rs.stream.color)
            for _ in range(20):
                pipe.wait_for_frames()
            fs = al.process(pipe.wait_for_frames())
            cv2.imwrite(os.path.join(a.cap_dir, "probe_color.png"), np.asanyarray(fs.get_color_frame().get_data()))
            np.save(os.path.join(a.cap_dir, "probe_depth.npy"), np.asanyarray(fs.get_depth_frame().get_data()).astype(np.float32) * sc)
            o = robot.get_observation(); np.save(os.path.join(a.cap_dir, "probe_q.npy"), np.array([o[f"{j}.pos"] for j in J]))
            pipe.stop()
            # left wrist D405: look at the jaws and the block up close
            try:
                pw = rs.pipeline(); rw = rs.config(); rw.enable_device(wrist_serial())
                rw.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rw.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pr = pw.start(rw); scw = pr.get_device().first_depth_sensor().get_depth_scale(); alw = rs.align(rs.stream.color)
                for _ in range(20):
                    pw.wait_for_frames()
                fw = alw.process(pw.wait_for_frames())
                cv2.imwrite(os.path.join(a.cap_dir, "probe_wrist_color.png"), np.asanyarray(fw.get_color_frame().get_data()))
                np.save(os.path.join(a.cap_dir, "probe_wrist_depth.npy"), np.asanyarray(fw.get_depth_frame().get_data()).astype(np.float32) * scw)
                pw.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[run] wrist camera failed: {e}", flush=True)
            print("[run] probe: saved the chest and wrist frames at the grasp pose", flush=True)
            if a.probe_hold > 0:  # 2026-11-22: 爪の間の真ん中と机の印のずれを人が測る間、掴む姿勢で止めておく
                print(f"[run] probe: {a.probe_hold:.0f} 秒止める（爪の間の真ん中と印のずれを測る）", flush=True)
                time.sleep(a.probe_hold)
            go(pre, 10, G_OPEN, "back to pregrasp"); go_route(r_pre_up, 12, G_OPEN, "back to raise")
            done = True
            raise KeyboardInterrupt
        if a.wrist_snap:  # evidence only: what the wrist D405 sees right before closing (no correction)
            try:
                import cv2, pyrealsense2 as rs
                pw = rs.pipeline(); rw = rs.config(); rw.enable_device(wrist_serial())
                rw.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30); rw.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
                pr = pw.start(rw); scw = pr.get_device().first_depth_sensor().get_depth_scale(); alw = rs.align(rs.stream.color)
                for _ in range(15):
                    pw.wait_for_frames()
                fw = alw.process(pw.wait_for_frames())
                cv2.imwrite(os.path.join(a.cap_dir, "wrist_at_grasp.jpg"), np.asanyarray(fw.get_color_frame().get_data()))
                np.save(os.path.join(a.cap_dir, "wrist_at_grasp_depth.npy"), (np.asanyarray(fw.get_depth_frame().get_data()).astype(np.float32) * scw).astype(np.float16))
                pw.stop()
            except Exception as e:  # noqa: BLE001
                print(f"[run] wrist snapshot failed: {e}", flush=True)
        go(grasp, GR, G_CLOSE, "close gripper", correct=False); time.sleep(a.close_wait)
        holding = True  # 離すまで: 途中で止めたら元の場所に下ろしてから戻る（finally）
        if carry is not None and a.replan_place:
            carry, place, r_lift_carry, r_carry_up = replan_place_from_grasp(carry, place, r_lift_carry, r_carry_up)
        if gp and gp["grasp_lift"] and not a.wrist_servo:  # 掴んだ所から手先の直線で真上へ
            go_route(gp["grasp_lift"], LR, G_CLOSE, "lift")
        else:
            go(lift, LR, G_CLOSE, "lift")
        time.sleep(a.hold_s); rec("hold")
        if carry is not None:  # self-reset: put it down at a new spot (grasp/tools/plan_place.py), next trial starts there
            if a.avoid:  # 割り込む物を見ながら運び、最後に位置を合わせる
                carry_watch([lift] + r_lift_carry, G_CLOSE, "carry", holding_now=True)
                go(carry, CR, G_CLOSE, "carry to the new spot")
            else:
                go_route(r_lift_carry, CR, G_CLOSE, "carry to the new spot")
            go(place, PR, G_CLOSE, "put down at the new spot")
            ret = {}
            if not a.avoid:  # 爪を開く間に、置いた物をよける戻りを裏で作る（2026-11-23）
                import threading
                o = robot.get_observation()
                q_put = [float(o[f"{j}.pos"]) for j in J]
                th_ret = threading.Thread(target=lambda: ret.update(plan_return(q_put)), name="return", daemon=True)
                th_ret.start()
            go(place, GR, G_OPEN, "release", correct=False)
            holding = False
            if not a.avoid:
                th_ret.join()
                if "err" in ret:
                    raise ret["err"]
                go(ret["q_up"], PR, G_OPEN, "up from the new spot")
                go_route(ret["route"], TR, G_OPEN, "back to raise")
            else:
                go(carry, PR, G_OPEN, "up from the new spot")
            if a.avoid:
                carry_watch([carry] + r_carry_up, G_OPEN, "back to raise")
                go(up, TR, G_OPEN, "back to raise")
        else:
            if gp and gp["lift_grasp"] and not a.wrist_servo:
                go_route(gp["lift_grasp"], PR, G_CLOSE, "put back down")
            else:
                go(grasp, PR, G_CLOSE, "put back down")
            go(grasp, GR, G_OPEN, "release", correct=False)
            holding = False
            if gp and not a.wrist_servo:  # 差し込んだ道を逆に（wrist servo は掴む姿勢を直したので直線で戻る）
                go_route(gp["grasp_pre"], NR, G_OPEN, "back to pregrasp")
            else:
                go(pre, NR, G_OPEN, "back to pregrasp")
            go_route(r_pre_up, TR, G_OPEN, "back to raise")
        done = True
    except SystemExit as e:
        # 非常停止・電源断（ここの MotorLost と grasp/reactive.py の MotorLost の両方）: 戻る動きもしない
        # （指令が届かない・電源が戻った瞬間に跳ぶ）。ほかの SystemExit は今までどおり戻りの処理へ
        if type(e).__name__ != "MotorLost":
            raise
        print(f"{e}。戻る動きはしない。腕を支えて電源を確かめる", flush=True)
        motor_lost = True
    except KeyboardInterrupt:
        TR = min(TR, 20.0)  # an aborted trial may be next to the object: no fast moves on the way back
        print("[run] stopping: returning to the start pose", flush=True)
    finally:
        try:
            if motor_lost:
                raise SystemExit(1)
            g_end = g_now if g_now > -10 else G_OPEN
            if done:  # raise にいる: 作っておいた経路で戻る
                go_route(r_up_start, TR, g_end, "back to start")
            else:  # 途中で止めた: 今の位置から経路を作り直す。作れなければ今までどおり直線で戻る
                o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
                if holding:  # 物を持ったまま: 掴んだ場所の上へ戻り、元の場所に下ろして離してから戻る（持った物を机に押し付けない）
                    print("[run] 物を持っている: 元の場所に下ろしてから戻る", flush=True)
                    if watcher is not None:
                        carry_watch([q_now] + route(q_now, lift, "now→lift"), G_CLOSE, "back to the grasp spot", holding_now=True)
                    else:
                        go_route(route(q_now, lift, "now→lift"), TR, G_CLOSE, "back to the grasp spot")
                    go(lift, NR, G_CLOSE, "over the grasp spot")
                    go(grasp, NR, G_CLOSE, "put back down (abort)")
                    go(grasp, GR, G_OPEN, "release (abort)", correct=False)
                    go(pre, NR, G_OPEN, "back to pregrasp (abort)")
                    o = robot.get_observation(); q_now = [float(o[f"{j}.pos"]) for j in J]
                    g_end = G_OPEN
                try:
                    back = None
                    if not holding and max(abs(x - y) for x, y in zip(q_now, pre)) < 3.0:
                        # pregrasp にいる（wrist servo を止めた後など）: 動く前に作った pregrasp→raise→休みの経路で戻る
                        # （2026-11-23: 今の位置から作ると机の線に掛かって作れず、直線で戻った）
                        if isinstance(r_pre_up, Dense):  # plan_grasp の行きの軌道を逆にたどって raise へ、そこから休みへ
                            go_route(r_pre_up, TR, g_end, "back to raise (abort)")
                            back = r_up_start
                        else:
                            back = list(r_pre_up) + list(r_up_start)
                        print("[run] pregrasp にいる: 作っておいた pregrasp→raise→休みの経路で戻る", flush=True)
                    if back is None and not holding:  # 離した後: 物の上を横切らないように物もよけて作る。作れなければ（物のすぐそばにいる）よけずに作る
                        avoid_target(True)
                        try:
                            back = route(q_now, start, "now→start（物をよける）")
                        except SystemExit as e:
                            print(f"[run] 物をよける戻りの経路を作れない（{e}）。物をよけずに作る", flush=True)
                        finally:
                            avoid_target(False)
                    if back is None:
                        back = route(q_now, start, "now→start")
                except SystemExit as e:
                    print(f"[run] 戻りの経路を作れない（{e}）。直線で戻る", flush=True)
                    back = [list(start)]
                go_route(back, TR, g_end, "back to start")
            if home is not None and a.hold_home:
                print("[run] holding at home with torque ON (release: grasp/tools/release_arm.py)", flush=True)
            elif home is not None and not at_home:  # the home pose drops when limp (11/20: J7 swung 63 deg): back to the resting pose first
                go_route(r_home_up0, TR, g_end, "back to raise over rest")
                go(rest, min(TR, 20.0), g_end, "back to rest")
        finally:
            if watcher is not None:
                watcher.stop()
            if wcam is not None:
                wcam.close()
            release(); log.close()
            if motor_lost:
                print("[run] モーターの返事が途切れたので、戻る動きはしていない（腕は止まった所）", flush=True)
            else:
                print("[run] back at start, " + ("holding (torque on)" if (a.hold_home and home is not None) else "torque off"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
