# grasp/ — depth-based grasping, minimal setup (OpenArm left arm × chest D435i × GraspGenX)

The request is Kota's Claude Doc "タスク依頼：深度ベース把持の最小構成" (2026-11-20). No training: point cloud → largest cluster → GraspGenX → IK → real arm.
Done when: wood block and PET bottle, 10 placements each, ≥70% success grasping from above.

## Status (2026-11-20)

| Step | File | State |
|---|---|---|
| 1 capture point cloud | `capture.py` | code written. **Not run on the robot** (collection PC + D435i 344422070898) |
| 2 cut out the object | `segment.py` | **passed** offline on real RGB-D samples (plane removal + DBSCAN largest cluster) |
| 3 grasp estimation | `infer.py` | **passed** with the GraspGenX server (stand-in gripper robotiq_2f_85, ~1.5–2 s/call, top score 0.84–0.89) |
| 3' register the custom hand | `config/openarm_custom_hand_sweep.json`, `config/openarm_left_gripper.json` | **measured on the real arm** 2026-11-20: tip gap 110 mm open (-66°) / 0 closed, flange→tip 140 mm |
| 4 hand-eye | `handeye.py` | not started |
| 5 IK | `plan.py` (uses `ik/openarm_left_ik.py`) | not started. ⚠️ The URDF TCP is the flange (offset 0) |
| 6 real arm | `run_grasp.py` | not started |
| 7 GraspNet-baseline comparison | `infer.py` backend | not started |

## GraspGenX server (separate env)

```bash
git clone https://github.com/NVlabs/GraspGenX ~/work/GraspGenX && cd ~/work/GraspGenX   # commit 44eb1d77
uv venv --python 3.11 && uv pip install --prerelease=allow -e ".[visualization,serve]" open3d opencv-python-headless
# uv sync picks up the 3.13 interpreter and fast-inference doesn't resolve, so install with uv pip
.venv/bin/python client-server/graspgenx_server.py --config ext/graspgenx_checkpoints/release \
    --assets_dir ext/gripper_descriptions/gripper_descriptions/assets --port 5556
```
Checkpoints download automatically on first import (the model is under the **NVIDIA Open Model License**; code is Apache-2.0).

## Offline smoke test (no robot)

```bash
PYTHONPATH=~/work/GraspGenX ~/work/GraspGenX/.venv/bin/python grasp/tools/smoke_offline.py \
    --sample ~/work/GraspGenX/assets/sample_data/real_world/00 --gripper-name robotiq_2f_85
# For a real capture: --cap grasp/runs/cap_xxx
```

## Using the custom hand with GraspGenX (no URDF needed)

The GraspGenX server's `infer_object` accepts 12 numbers (gripper base frame, +Z approach, +X closing):
the inner finger volume at fully open (`extents_open`/`offset_open`) and half open (`extents_mid`/`offset_mid`)
as boxes. Measure them with calipers and write them to `config/openarm_custom_hand_sweep.json` → `--sweep-json`.
For IK you also need the flange→fingertip-midpoint (TCP) offset, which comes from the same measurement.

## Coordinate conventions (most failures come from frame mismatches)
- Camera: RealSense optical frame (+x right, +y down, +z forward)
- GraspGenX grasp: origin = gripper base, +Z approach, +X closing
- GraspNet (graspnetAPI): **x axis is the approach direction** (convert it)
- OpenArm base `openarm_body_link0`: +x forward, +y left, +z up
