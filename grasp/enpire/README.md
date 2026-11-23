# grasp/enpire — ENPIRE 式の自動ループ（OpenArm 左腕・上からの把持）

NVIDIA GEAR の ENPIRE（`NVlabs/ENPIRE`・Apache-2.0）の「auto-research contract」を、この把持タスクに小さく写したもの。ENPIRE 本体は YAM 用で import しない。考え方だけを借りる（2026-11-20・案）。

| ENPIRE の決まり | ここでの形 |
|---|---|
| 仮説を 1 つ立て、方策側を最小に変える | `loop.py run --hypothesis … --set k=v`、`auto` は 1 つの値だけ変える案を順に試す（`policy.SEARCH_SPACE`） |
| 判定・リセット・安全・試行数・記録の形は方策側から変えない | 判定は `contract.py`（固定・`VERIFIER_VERSION`）。変えたら版を上げ、前の回とは比べない |
| 失敗した試行も残す | `grasp/runs/enpire/iter_NNN/trials/*/result.json` と `overlay.jpg` |
| 準備の点検が落ちたら止める | GraspGenX の口・Thor の ssh を見て、落ちていれば回を `aborted` にする |
| 前の最良を超えたときだけ残す | `best.json` の `tracks`（source・hand-eye・判定の版の組ごとに最良を持つ）と `best_params.json`。比べるのは同じ組の中だけ |

## 判定（contract.py）の段

`perception`（撮る→切り出す→把持を選ぶ）→ `plan`（腕の座標に移して IK が届く）→ `real`（腕で持ち上がった）。

述語: `object_found`・`grasp_found`・`grasp_on_object`（開いた顎の間に物体の点が 20 点以上）・`top_down`（真下から 30° 以内）・`width_ok`（顎の帯の中の物体幅 ≤ 110 mm）・`reachable`・`lifted`。成否は当てはまる述語の全部、失敗の原因は最初に落ちた述語。合格の線は 70%（Kota の依頼の線）。

## 打つもの

```bash
cd ~/work/openarm-orboh
python3 grasp/enpire/dashboard.py &            # → http://localhost:8765（2 秒ごとに更新）
PY="env PYTHONPATH=$HOME/work/GraspGenX:$PWD $HOME/work/GraspGenX/.venv/bin/python"   # GraspGenX の server を :5556 に立てておく
$PY -m grasp.enpire.loop run --hypothesis "baseline" --source 'offline:grasp/runs/cap_*'   # 保存済みの撮影で回す
$PY -m grasp.enpire.loop auto --source 'offline:grasp/runs/cap_*' --max-iters 8          # 自動で値を探す
$PY -m grasp.enpire.loop run --hypothesis "…" --t-base-cam <T_base_cam.json> --source 'offline:grasp/runs/cap_*'   # plan 段（IK）
$PY -m grasp.enpire.loop run --hypothesis "…" --source thor:5   # Thor の胸カメラで 5 枚撮る（腕は動かない・1 枚ごとに置き直す 15 秒）
```

`--source live:N`（腕で実際に掴む）は今は止める。hand-eye の確かめ、IK と動かす段（`run_grasp.py`）、人がロボットの前で頼むこと（AGENTS.md §3-8）がそろってから足す。

## 2026-11-20 時点の結果

- perception 段: 保存済みの 4 シーン（アルミの棒）で 4/4。1 つの値だけ変える案 4 つは、どれも越えず（`dbscan_eps` 0.025 で 3/4 に下がる）
- plan 段: 今の ICP の hand-eye（1 cm 以内の点が約 20%）では 0/4 が届かない。hand-eye が怪しいので、IK の結論にはまだ使わない
- 気づいた点: 切り出しが棒の一部しか取れず、手前の手の方へ漏れる場面がある。今の判定はこれを落とさない（物体全体を取れたかの述語は無い）

### 2026-11-20 夜（ArUco の hand-eye `grasp/config/T_base_cam.json` が入った後）

- plan 段・18〜19 時の撮影 9 シーン: 点数が一番の候補だけ IK にかけると 2/9（iter_006）。真上から 25° 以内の候補を点数順に 20 個まで試し、最初に届くものを選ぶ（`select_ik_tries=20`）と 4/9（iter_007・この組の最良）
- 残りの失敗は 4 シーンが「どの候補も届かない」、1 シーンが切り出しの失敗。もう一つのセッションが出した「真上から床に届くのは手の真下の 10 cm くらいの帯だけ」と合う

### 2026-11-20 夜・実機（live、`grasp/enpire/live.py` + `grasp/tools/run_grasp_servo.py`）

- iter_008〜017: 実機の段をつなぎ、空振りの原因を絞った。腕の関節は 1° 以内で届いているが、手先のモデルが姿勢によって数 cm ずれる。手首カメラ（D405）で爪の間の木片を測ると、把持の位置で厚み方向に 4〜5 cm・深さで約 3 cm ずれていた。プレグラスプで合わせても、把持までに J3・J5 が 36° 動くあいだに 4 cm ずれる
- 判定の版 .3: 「上から」を 30° → 45°（2026-11-20 宮嶋の判断。30° 以内ではカメラに見えて届く把持が無い）
- iter_018（v5.1）: 把持の 4 cm 手前で、爪の形から出した軸（閉じる・厚み・接近）でずれを測って直し、届く深さまで下ろす。木片を 0.38 m に置き直して **2/2 成功**（保持中のグリッパー -15°・手首カメラで爪の間に木片を確認）
- まだ: 置き場所を変えた 10 回 × 木片とペットボトル（Kota の完了条件）。木片 0.52 m は届く範囲の端で、掴めない

### 2026-11-20 20:00〜21:00・アルミの棒と、棒に触れた木片（iter_019〜025）

- 木片が棒に触れていると、一つの塊として切り出される → `thor_snap.py --exclude-wood`（棒を狙う）/ `--exclude-bright`（木片を狙う）。手首カメラも同じ色で分ける
- 棒（高さ 3 cm）は物体の中心を狙うと爪先が床に当たる → `plan.py --raise-to-clear`。棒は長いので中心からの許容を 8 cm に
- iter_021: 棒を狙った試行は、左の爪が横の木片に当たって外れた（宮嶋の指摘）→ `plan.py --avoid-obstacles`。届く区間の横に木片が付いていて、当たらない掴み方は 0
- A（先に木片をどける）iter_022〜025: 実際の爪は計画より 3.5〜5 cm 浅い → `--extra-depth 0.04`。持ち上げの高さを 10→5 cm まで下げて先に確かめる。閉じる向きは毎回 1 cm 以内まで合うが、木片が画面の上で切れたままで厚み方向（-45〜-60 mm）を直せず、爪が当たって止まる。可動域の端の姿勢ばかりで IK の枝も飛ぶ
- まだ: 物体どうしが触れた状態は未達。間を空けて置けば棒単体の手順（iter_021 までの直しを含む）で試せる
