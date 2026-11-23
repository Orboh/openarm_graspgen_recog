# openarm-grasp — OpenArm 左腕の深度ベース把持 (Uses YOLOE 26+SigLIP2+ Rulebased picking order for target object selection)

Kota の依頼「タスク依頼：深度ベース把持の最小構成（OpenArm × RealSense）」の実装。学習はしない。チェストの D435i の点群から物体を切り出し、把持モデル（GraspGenX か GraspNet-baseline）で掴み方を出し、IK を解いて OpenArm の左腕で掴む。

2026-11-21 に `Orboh/openarm-orboh` の枝 `feat/grasp-graspgenx`（PR #82）から、`grasp/`・`ik/openarm_left_ik.py`・`urdf/` を履歴ごと分けた。PR #82 はそのまま残してある（閉じるかは Kento が決める）。

## 信州タケエイ（2026-11-25）に持っていく版 — main がそれ

**main が最新版。PC と Thor の両方が main で動く**（2026-11-23 にそろえた）。直すときは、PC で直して commit・push してから、Thor で `git pull` する。rsync で写すのはやめた。

| リポ | Thor（実機を動かす所） | PC | 使い道 |
|---|---|---|---|
| Orboh/openarm-grasp（これ） | `~/projects/openarm-grasp`（main・`~/grasp_work` はここへの近道） | `~/work/openarm-grasp`（main） | 撮影・計画・`run_grasp.py`・腕の補正・試行のスクリプト |
| Orboh/openarm-orboh | `~/projects/openarm-orboh`（main） | `~/work/openarm-orboh`（main） | `scripts/canup.sh`（CAN）、`urdf/`、GR00T alframe の姿勢セット |
| Orboh/lerobot（fork） | `~/projects/lerobot`（**ブランチ `feat/bi-openarm-soft-move-to`**・`a720ec19`） | なし | モーター（Damiao）・腕（OpenArmFollower）。main ではない（Orboh/lerobot の main と分かれている） |
| NVlabs/GraspGenX | `~/work/GraspGenX` | `~/work/GraspGenX` | 掴み方の生成（変更なし） |

Thor だけにある物（git に入れていない）:
- alframe モデルの Thor 用の行（Thor の openarm-orboh の `kitakyushu-alframe-v1/set.yaml`）。信州タケエイの台車の姿勢セットは、このリポの `grasp/config/pose_sets/shinshu-takeei-cart-v1/` に入れた（使い方は `grasp/config/pose_sets/README.md`）。
- 試行の記録 `grasp/runs/`、`~/rollouts/RUNLOG.md`。

現場での立ち上げ（Thor）:

```bash
cd ~/projects/openarm-grasp && git pull                 # main
~/projects/openarm-orboh/scripts/canup.sh               # CAN（can_fol_l / can_fol_r が ERROR-ACTIVE）。非常停止の後も打つ
bash grasp/tools/thor_servers.sh start                  # perceive 5558・GraspGenX 5556・cuRobo 5559（約 2 分）・VLM 5560
bash grasp/tools/thor_servers.sh status
# 胸カメラを付け替えたら、ハンドアイを取り直す（handeye_marker_poses.py → handeye_near.py --poses → fit_handeye_markers.py --left-corr --floor）
bash grasp/tools/left_trials.sh 10                      # 左腕のテスト（tmux で）。止める: touch ~/grasp_runs/STOP
bash grasp/tools/remove_all.sh 6                        # 見えている物を 1 つずつ肩の横へ取り除く
```

## いまの結果（2026-11-21）

| モデル | 物体・置き方 | 成功 | 記録 |
|---|---|---|---|
| GraspNet-baseline | 木片・寝かせる（10 か所・腕が自分で置き直す） | **9/10** | ENPIRE iter_038 |
| GraspGenX | 木片・寝かせる（同じ流れ） | 最良 4/8 | ENPIRE iter_034 |

- 外れは、木片が爪の厚み方向に 3.4 cm 以上ずれたとき。爪の厚み 35 mm に対して木片は 50 mm なので、許される幅は ±2.5 cm くらいしかない。
- 外れ方は腕の姿勢（J6）で変わる。原因の調べは途中（Vault の作業記録 §9）。
- 判定の「上から」は 45° 以内に緩めてある（Kota の条件は真上から）。

## 中身

| パス | 何 |
|---|---|
| `grasp/capture.py`・`segment.py`・`infer.py`・`plan.py`・`run_grasp.py` | 手順 1〜6: 撮る → 切り出す → 掴み方 → IK → 実機 |
| `grasp/handeye.py`・`grasp/tools/handeye_*.py` | 手順 4: ArUco のハンドアイ |
| `grasp/tools/graspnet_server.py` | 手順 7: GraspNet-baseline のサーバー（port 5557）。`plan.py --model graspnet` で使う |
| `grasp/tools/plan_place.py`・`release_arm.py`・`run_grasp_servo.py` | 置き直しの計画、保持からの脱力、手首カメラでの位置合わせ |
| `grasp/tools/make_joint_sweep.py`・`teach_fit.py` | 運動学の較正、爪の位置の推定 |
| `grasp/enpire/` | ENPIRE 式の自動ループ（判定は `contract.py` で固定）。実機の段は `live.py`、画面は `dashboard.py`（http://localhost:8765） |
| `grasp/config/` | ハンドアイ `T_base_cam.json`、爪の実測、GraspGenX 用の爪の寸法、手首カメラの爪の位置、初期姿勢 |
| `ik/openarm_left_ik.py` | Pinocchio の左腕 IK |
| `urdf/` | OpenArm の URDF（`_umi` は左手に実物の指を付けた版） |

記録（`grasp/runs/`）は git に入れない。この PC の `grasp/runs/` と S3 `s3://orboh-models/enpire/20261121-openarm-graspgenx/`（AWS profile `kota0`）にある。

## 動かす環境

- この PC: GraspGenX のサーバー（`~/work/GraspGenX`・port 5556。設定は `grasp/README.md`）、計画・ENPIRE は GraspGenX の venv で動かす。
- GraspNet-baseline（port 5557）: conda の `graspnet` 環境。

  ```bash
  conda create -n graspnet python=3.10 && conda activate graspnet
  conda install -c nvidia/label/cuda-12.1.1 cuda-nvcc cuda-cudart-dev cuda-libraries-dev
  pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
  pip install "numpy<2" scipy open3d Pillow tqdm tensorboard gdown transforms3d trimesh scikit-image \
      "opencv-python-headless<4.11" pyzmq msgpack msgpack-numpy grasp_nms matplotlib scikit-learn h5py dill autolab_core cvxopt pywavefront
  git clone https://github.com/graspnet/graspnet-baseline ~/work/graspnet-baseline
  git clone https://github.com/graspnet/graspnetAPI ~/work/graspnetAPI
  export CUDA_HOME=$CONDA_PREFIX TORCH_CUDA_ARCH_LIST=8.6
  (cd ~/work/graspnet-baseline/pointnet2 && pip install --no-build-isolation .)
  (cd ~/work/graspnet-baseline/knn && pip install --no-build-isolation .)
  (cd ~/work/graspnetAPI && pip install --no-deps .)
  mkdir -p ~/work/graspnet-baseline/logs && gdown 1hd0G8LN6tRpi4742XOTEisbTXNZ-1jmk -O ~/work/graspnet-baseline/logs/checkpoint-rs.tar
  python grasp/tools/graspnet_server.py --port 5557
  ```

- Thor: 腕（CAN `can_fol_l`・LeRobot の venv）とチェストカメラ。実行の script は `live.py` が Thor に写して打つ。

### 1 台で動かす（腕・カメラ・GPU が同じ PC・2026-11-21 北九州の収集機）

置き場ごとの値は環境変数で差し替える（無ければ上の東京の値）。置き場の値は `grasp/config/sites/<置き場>.env` にまとめる。

| 環境変数 | 何 | 東京の既定 |
|---|---|---|
| `GRASP_ROBOT_HOST` | 腕とカメラの PC。`local` ならこの PC の bash で打ち、scp の代わりに cp（`grasp/remote.py`） | `orbou-thor@orboh-thor` |
| `GRASP_CAMERA_PY` | カメラを開く Python | `/usr/bin/python3` |
| `GRASP_CAN_PORT`・`GRASP_ROBOT_ID` | 左腕の CAN と較正（`calibration/robots/openarm_follower/<ID>.json`） | `can_fol_l`・`follower_left` |
| `GRASP_CHEST_SERIAL`・`GRASP_WRIST_SERIAL` | 胸の D435i・左手首の D405 | `344422070898`・`409122272149` |
| `GRASP_GRIPPER_OPEN`・`GRASP_GRIPPER_CLOSE` | グリッパーを開く・閉じる指令の角度（lerobot の度） | `-60`・`2` |
| `GRASP_GRIPPER_JSON`・`GRASP_SWEEP_JSON` | 爪の寸法（`plan.py`・`enpire/contract.py`）と、GraspGenX に渡す爪の箱（`plan.py`・`enpire/policy.py`） | `grasp/config/openarm_left_gripper.json`・`openarm_custom_hand_sweep.json` |
| `GRASP_T_BASE_CAM` | ハンドアイ（胸のカメラの位置と向き）のファイル。`GRASP_SITE` があるのに無いと `plan.py` が止まる | `grasp/config/T_base_cam.json` |
| `GRASP_FLOOR_Z` | 腕の付け根の座標で見た床（机の板）の高さ [m]。床に当たらないかの確認に使う | `0.27` |
| `GRASP_REST` | 脱力の前に下ろす休みの姿勢（lerobot の度） | `20.3,-3.6,-0.9,92.8,-2.5,-11.0,22.7` |
| `GRASP_SITE` | 置き場の名前。これがあると `GRASP_FLOOR_Z`・`GRASP_REST` が無いときに腕を動かす script が止まる | 無し |

```bash
ln -s ~/projects/openarm-grasp ~/grasp_work        # local では作業場をこの repo にする（一度だけ）
set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
```

ハンドアイ（`grasp/config/T_base_cam.json`）・初期姿勢・爪の実測は置き方と機体ごとに取り直す。

### cuRobo の当たり判定（2026-11-21・北九州）

腕を cuRobo（`~/work/curobo`・v0.9.0・Apache-2.0）の球で表し、周りの物（`grasp/config/sites/<置き場>.world.yml`: 純正の本体のメッシュ・胸のカメラの箱・机の板）との距離を測る。`GRASP_CUROBO` があると:

- `plan.py`: pregrasp → grasp → lift で当たる候補を捨てる
- `run_grasp.py`: 移動の区間（休みの姿勢・home・raise・pregrasp の間、置き直しの移動）が当たるか床に近いときは、cuRobo の MotionPlanner で回り込む経由点を作り（`grasp/curobo/planner.py`）、経由点ごとに位置を合わせながら順に動かす。経由点の間も床・keep-out の箱・cuRobo の確認をもう一度通す。回り込みを作れなければ、腕を動かす前に止まる。物の近くの区間（pregrasp ↔ grasp ↔ lift・置く）は直線のままで、当たれば止まる
- サーバーが立っていなければ両方とも止まる。`GRASP_KEEPOUT_BOXES` は腕の太さの分だけ大きく、cuRobo が通す回り込みも止めることがある

```bash
set -a; source grasp/config/sites/kitakyushu-collect.env; set +a
~/work/curobo/.venv/bin/python grasp/curobo/server.py                     # 常駐（port 5558・起動に約 35 秒・GPU 約 2 GB）
~/work/curobo/.venv/bin/python grasp/curobo/check_runs.py grasp/runs/<cap>  # 記録の軌跡を調べる（動かない）
```

- cuRobo の入れ方: `git clone --branch v0.9.0 https://github.com/NVlabs/curobo ~/work/curobo && cd ~/work/curobo && uv venv --python 3.11 && uv pip install --python .venv/bin/python ".[cu12-torch]" pyzmq msgpack`
- 腕のモデルを作り直す: `python3 grasp/curobo/make_urdf.py` の後に、`grasp/curobo/` で `~/work/curobo/.venv/bin/python -m curobo.examples.getting_started.build_robot_model --urdf $PWD/openarm_left_umi.urdf --asset-path $PWD --tool-frames openarm_left_link7 --output $PWD/openarm_left_umi.yml --seed 0`
- 手の本体（モーターの箱）は `make_urdf.py` の `GRIPPER_BODY_*`（実物と合うと 2026-11-21 に確かめた）

## 右腕を足す・両腕で計算して良い方で掴む（2026-11-22・作業者: Munenori）

右リーダーに左と同じグリッパー（so101 のフィンレイ指）を鏡写しに付け、1 つの物に対して左右の計画を同時に計算し、良い方の腕で掴む（2 本が同時に動くことはない）。

- 計画: `plan.py --arm both`（`left`・`right` も可。既定は `GRASP_ARM` か `left`）。左右を別のプロセスで同時に計算し、GraspGenX の点数の高い方の腕を選ぶ。点数の差が `--arm-tie`（0.05）以内なら、今の姿勢から pregrasp までの関節の動きが小さい方を選ぶ。`plan.json` に `"arm"` を書き、腕ごとの結果は `plan_left.json`・`plan_right.json` にも残す。左だけのときより約 0.5 秒長い。ENPIRE からは `--plan-args "... --arm both"` で渡す
- 実行: `run_grasp.py` は `plan.json` の腕を動かす。右の値（`GRASP_CAN_PORT_RIGHT`・`GRASP_ROBOT_ID_RIGHT`・`GRASP_GRIPPER_OPEN_RIGHT`・`GRASP_GRIPPER_CLOSE_RIGHT`）が置き場の env に無いと、`--go` は腕につなぐ前に止まる。左の値に落とさない（`grasp/arm.py`）。右で `--home` を使うときは `--home-right` か `GRASP_HOME_RIGHT` が要る
- 当たり判定: 右は 2 つ目のサーバー `server.py --arm right`（port 5559・`GRASP_CUROBO_RIGHT`）。左の cuRobo のモデル（`openarm_left_umi.yml` の球）を鏡に写して使うので、球を当てはめ直せば右にも効く。URDF の右腕の各 link の形は左の鏡写しではない（同じ部品を右の枠に置いた物・手首で最大 58 mm 違う）ので、球は link ごとに右の座標へ置き直す（`grasp/curobo/model.py` の `right_local_center`）。グリッパーは鏡写しのまま
- 左右の対応: `q_右 = [-1,-1,-1,+1,-1,-1,-1] * q_左` で、link の枠と指先が鏡写しになる（link4 だけ肘の軸に沿って 63 mm ずれる）。`tests/test_arm.py` が URDF で確かめる
- 撮影時の腕の点を消すのに、右は `GRASP_REST_RIGHT` を使う。無いと右腕自身が障害物に見えて、右の計画と移動が止まる（当たらない側に倒れる）
- **まだ実機で確かめていない**。右腕を付けたら、`grasp/config/sites/kitakyushu-collect.env` の右腕の節にある 1〜6 を確かめてから、`#` の付いた行を外す

```bash
~/work/curobo/.venv/bin/python grasp/curobo/server.py               # 左（5558）
~/work/curobo/.venv/bin/python grasp/curobo/server.py --arm right   # 右（5559）
PYTHONPATH=.:~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/plan.py <cap> --q "$GRASP_REST" --arm both ...
```

## 拾う物を名前で選ぶ（2026-11-22・作業者: Munenori）

何も付けないと、机の上で**いちばん大きいかたまり**を拾う。北九州の記録では、そのやり方だと奥の段ボールを拾っていた（`--zmin 0.20 --band none` のとき）。`--target <名前>` を付けると、登録した物の色と大きさが合うかたまりだけを選ぶ。

- 登録は `grasp/config/objects/<名前>.json`: 色（HSV）・色が合う点の割合の下限・外形の範囲・一緒に置いたときに消す相手（`separate_from`）。今あるのは `wood`（木片）と `alframe`（アルミのバー・北九州の実物ではまだ確かめていない）。一覧は `thor_snap.py --list-targets` で見る
- 合うかたまりが無ければ `object.npy` を作らず止まる。いちばん大きいかたまりには戻らない。どのかたまりを何で外したかは、撮った記録の `target.json` に残る。図（`overlay.jpg`）では、選んだかたまりが赤、外したかたまりが黄色になる
- 流れの中で使うときは `live.py prepare --snap-args "--hmax 0.10 --target wood"`。`plan.json` に `target` が入り、手首カメラの位置合わせ（`run_grasp_servo.py`）にも同じ見分け方が渡る。オフラインの ENPIRE ループでは params の `"target": "wood"` を使う
- 新しい物を足すときは json を 1 つ足し、`PYTHONPATH=. ~/work/GraspGenX/.venv/bin/python -m unittest discover -s tests -q` を通す
- `--yoloe`（本 repo で追加）は別の選び方: YOLOE-26 のマスク + 孤立度・ロボットからの距離・積みの高さの規則で対象を決め、SigLIP2（`grasp/classification.py`）はその対象 1 つだけを後から材質で分ける。`--target` の色・大きさ指定より、見たことの無い物や材質の見分けが要るときに向く

## 回し方（腕が動くのは execute と batch だけ）

```bash
PY="env PYTHONPATH=$HOME/work/GraspGenX:$PWD $HOME/work/GraspGenX/.venv/bin/python"
$PY -m grasp.enpire.live new-iter --hypothesis "..." --executor grasp/run_grasp.py \
    --home grasp/config/home_left_cart-sym-v4.yaml --auto-place \
    --exec-args "--transit-dps 75 --near-dps 20 --hold-home" \
    --plan-args "--model graspnet --max-tilt 45 --rounds 5 --min-score 0.5 --max-try 400 --floor-clear 0.006 --thick-tol 0.015"
$PY -m grasp.enpire.live prepare --iter iter_NNN --snap-args "--hmax 0.10"     # 撮影・計画・空のチェック（動かない）
$PY -m grasp.enpire.live batch --iter iter_NNN --n 10 --snap-args "--hmax 0.10" --i-have-operator-go   # 許可を取ってから
$PY -m grasp.enpire.live close --iter iter_NNN
```

## 守ること

- 腕を動かすのは、人がロボットの前にいて頼まれたときだけ。動かす前に毎回許可を取る。
- 速さ: OpenArm だけの例外で、移動は 75°/秒、物体の近くは 20°/秒まで。
- 保持（`--hold-home`）の後は `grasp/tools/release_arm.py --go` で休みの姿勢へ下ろしてから脱力する。初期姿勢のまま脱力すると腕が落ちる。
- 顧客の実名・鍵・トークンを書かない。

## ライセンス

- **GraspNet-baseline は学術・非営利の研究用のみ**（`~/work/graspnet-baseline/LICENSE`）。商用は不可と読める。この repo にそのコードは入れていない（`graspnet_server.py` は import して呼ぶだけ）。
- GraspGenX: コードは Apache-2.0、モデルは NVIDIA Open Model License。
- この repo のコードの扱いは Orboh の社内用（未定）。
