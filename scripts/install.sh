#!/bin/bash

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT=$(realpath "$SCRIPT_DIR/..")

LIBS_DIR="$TOOL_ROOT/libs"
ENVS_DIR="$TOOL_ROOT/.envs"
TOOL_NAME="ReluSplitter"
VENV_DIR="$ENVS_DIR/$TOOL_NAME"

UV_DIR="$ENVS_DIR/uv"
UV="$UV_DIR/uv"

if [ ! -f "$UV" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_DIR" sh
fi

rm -rf "$VENV_DIR"
"$UV" venv --python 3.11 "$VENV_DIR"

# PyTorch with CUDA 12.1
"$UV" pip install \
    --python "$VENV_DIR" \
    --extra-index-url https://download.pytorch.org/whl/cu121 \
    "torch==2.2.2+cu121" \
    "torchvision==0.17.2+cu121" \
    "triton==2.2.0"

# Remaining dependencies
"$UV" pip install \
    --python "$VENV_DIR" \
    "numpy==1.26.4" \
    "onnx==1.16.0" \
    "onnxruntime==1.18.1" \
    "coloredlogs==15.0.1" \
    "humanfriendly==10.0" \
    "networkx==3.3" \
    "psutil==5.9.0" \
    "pyyaml==6.0.2" \
    "sympy==1.12" \
    "tqdm==4.67.1" \
    "termcolor==3.1.0" \
    "beartype==0.20.2" \
    "ninja==1.11.1.4" \
    "pytest==8.3.5" \
    "pytest-order==1.3.0" \
    "setuptools"

cd "$TOOL_ROOT"
git submodule update --init --recursive
"$UV" pip install --python "$VENV_DIR" -e "$LIBS_DIR/auto_LiRPA"
sed -i '865s/self.ori_state_dict)/self.ori_state_dict, strict=False)/' "$LIBS_DIR/auto_LiRPA/auto_LiRPA/bound_general.py"
