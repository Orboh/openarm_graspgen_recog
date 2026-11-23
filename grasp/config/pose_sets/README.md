# 開始姿勢セット（OpenArm 両腕・GR00T alframe 用）

2026-11-23（宮嶋: 姿勢セットは openarm-orboh ではなく openarm-grasp に入れる）。

- `shinshu-takeei-cart-v1/`: 信州タケエイの台車での開始姿勢と較正（11/16〜17 の現場の試験で、Thor の `~/projects/openarm-orboh/config/pose_sets/` にあった物の写し。`.bak-*` は除いた）。モデルの行の `dir` は `~/` で書いてある。
- 読み書きの道具は openarm-orboh の `scripts/pose_set.py`。このフォルダを読ませるときは、置き場を環境変数で指す:

  ```bash
  PS_ROOT=~/projects/openarm-grasp/grasp/config/pose_sets python3 ~/projects/openarm-orboh/scripts/pose_set.py ...
  ```

- Thor の openarm-orboh 側の `kitakyushu-alframe-v1/set.yaml` にある、alframe ckpt の Thor 上の写しの行（`ft_bi_openarm_ti_alframe_t1-mix_v1-ckpt10000-thor`・`~/models/ft_bi_openarm_ti_alframe_t1-mix_v1/checkpoint-10000`）は、Thor の手元にだけある。
