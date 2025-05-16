#!/bin/bash

# export MKL_SERVICE_FORCE_INTEL=1
# cmd="$CONDA_HOME/envs/marabou/bin/Marabou  $@"
# cmd="/home/lli/tools/Marabou-2.0.0/build/Marabou  --milp  $@"
cmd="/home/lli/tools/relusplitter/libs/Marabou-2.0.0/build/Marabou $@"
echo $cmd
$cmd
