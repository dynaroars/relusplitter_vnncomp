#!/bin/bash


SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT=$SCRIPT_DIR/..

CONDA_HOME=.envs/conda
CONDA=$CONDA_HOME/bin/conda

# resolve the paths relative to the script's directory
LIBS_DIR=$(realpath "$TOOL_ROOT/libs")
ENVS_DIR=$(realpath "$TOOL_ROOT/.envs")
TOOL_NAME="ReluSplitter"
CONDA_PREFIX=$ENVS_DIR/$TOOL_NAME
ENV_FILE_PATH=$ENVS_DIR/ReluSplitter.yaml


if [ ! -d $CONDA_HOME ]; then
    wget -O conda.sh https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
    # wget -O conda.sh https://repo.anaconda.com/miniconda/Miniconda3-py310_23.3.1-0-Linux-x86_64.sh
    chmod 755 conda.sh
    ./conda.sh -bf -p $CONDA_HOME
    rm ./conda.sh
fi

$CONDA env remove --prefix $CONDA_PREFIX -y
$CONDA env create --prefix $CONDA_PREFIX -f $ENV_FILE_PATH -y

cd $TOOL_ROOT
git submodule update --init  --recursive
$CONDA_PREFIX/bin/pip install -e $LIBS_DIR/auto_LiRPA
sed -i '865s/self.ori_state_dict)/self.ori_state_dict, strict=False)/' $LIBS_DIR/auto_LiRPA/auto_LiRPA/bound_general.py
