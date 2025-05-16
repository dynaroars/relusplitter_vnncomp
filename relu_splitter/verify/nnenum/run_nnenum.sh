#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$TOOL_ROOT/libs/nnenum/src
export OPENBLAS_NUM_THREADS=1 
export OMP_NUM_THREADS=1

# cmd="$CONDA_HOME/envs/nnenum/bin/python -m nnenum.nnenum $@"
cmd="$TOOL_ROOT/.envs/nnenum/bin/python -m nnenum.nnenum $@"
echo $cmd
$cmd