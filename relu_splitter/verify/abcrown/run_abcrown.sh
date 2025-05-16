#!/bin/bash

# export MKL_SERVICE_FORCE_INTEL=1
cmd="$TOOL_ROOT/.envs/abcrown/bin/python  $TOOL_ROOT/libs/alpha-beta-CROWN/complete_verifier/abcrown.py $@"
# cmd="$CONDA_HOME/envs/abcrown_24/bin/python $TOOL_ROOT/libs/alpha-beta-CROWN/complete_verifier/abcrown.py $@"
echo $cmd
$cmd
