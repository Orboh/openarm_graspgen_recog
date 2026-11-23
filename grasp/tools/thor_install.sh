#!/bin/bash
# 2026-11-21 Sota（Claude Code）: 計算を PC から Thor へ移すとき、3 つの venv を作る。使い方: GGX_COMMIT=<GraspGenX の commit> bash thor_install.sh perceive|graspgenx|curobo
# GraspGenX の重み（ext/）は PC から ~/work/GraspGenX_ext へ rsync し、thor_servers.sh がリンクする
# runs on Thor: build the three venvs (perceive / GraspGenX / cuRobo). Logs: ~/grasp_runs/logs/install_*.log
set -u
export PATH=$HOME/.local/bin:$PATH
TORCH_IDX=https://download.pytorch.org/whl/cu130
L=$HOME/grasp_runs/logs; mkdir -p $L
which=$1
case $which in
perceive)
  mkdir -p ~/work/perceive && cd ~/work/perceive && uv venv --python 3.12 -q .venv &&
  uv pip install --python .venv/bin/python torch==2.9.1 torchvision --index-url $TORCH_IDX &&
  uv pip install --python .venv/bin/python "transformers==5.23.0" accelerate "numpy<2" opencv-python-headless pillow pyzmq msgpack msgpack-numpy hydra-core iopath huggingface-hub &&
  ( [ -d sam2 ] || git clone -q https://github.com/facebookresearch/sam2.git ) &&
  SAM2_BUILD_CUDA=0 uv pip install --python .venv/bin/python --no-build-isolation -e ./sam2 &&
  .venv/bin/python -c "import torch,transformers,sam2;print('perceive ok',torch.__version__,torch.cuda.is_available(),transformers.__version__)"
  ;;
graspgenx)
  cd ~/work && ( [ -d GraspGenX/.git ] || git clone -q https://github.com/NVlabs/GraspGenX.git ) && cd GraspGenX &&
  git checkout -q $GGX_COMMIT &&
  uv venv --python 3.12 -q .venv &&
  uv pip install --python .venv/bin/python torch==2.9.1 torchvision --index-url $TORCH_IDX &&
  uv pip install --python .venv/bin/python -e ".[serve]" open3d==0.20.0 pin scipy pyzmq msgpack msgpack-numpy opencv-python-headless matplotlib pyyaml pycollada lxml rtree shapely manifold3d mapbox-earcut fast-simplification xxhash colorlog psutil yourdfpy &&
  .venv/bin/python -c "import torch,open3d,pinocchio,graspgenx;print('graspgenx ok',torch.__version__,torch.cuda.is_available())"
  ;;
curobo)
  cd ~/work && ( [ -d curobo/.git ] || git clone -q --branch v0.9.0 https://github.com/NVlabs/curobo.git ) && cd curobo &&
  uv venv --python 3.12 -q .venv &&
  uv pip install --python .venv/bin/python ".[cu13-torch]" pyzmq msgpack --extra-index-url $TORCH_IDX &&
  .venv/bin/python -c "import torch,curobo;print('curobo ok',torch.__version__,torch.cuda.is_available())"
  ;;
esac
echo "[install] $which exit $?"
