#!/bin/bash

# export MKL_SERVICE_FORCE_INTEL=1
cmd="$TOOL_ROOT/.envs/neuralsat/bin/python  $TOOL_ROOT/libs/neuralsat/neuralsat-pt201/main.py $@"
# cmd="$CONDA_HOME/envs/neuralsat/bin/python $TOOL_ROOT/libs/neuralsat/neuralsat-pt201/main.py $@"
echo $cmd
$cmd
