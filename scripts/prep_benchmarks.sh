#!/bin/bash

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
TOOL_ROOT="$(realpath "$SCRIPT_DIR/..")"

SEED=$1
if [ -z "$SEED" ]; then
    echo "Usage: $0 <seed>"
    exit 1
fi

PYTHON=$TOOL_ROOT/.envs/ReluSplitter/bin/python
if [ ! -d "$TOOL_ROOT/.envs/ReluSplitter" ]; then
    echo "Please run scripts/install.sh first"
    exit 1
fi

INPUT_DIR=$TOOL_ROOT/Seed_Inputs


# ACAS Xu included
# Matmul and Add was merged to GEMM
cd $INPUT_DIR
cd acasxu_converted


# MNIST FC
cd $INPUT_DIR
rm -rf mnist_fc_vnncomp2022
git clone https://github.com/pat676/mnist_fc_vnncomp2022.git
cd mnist_fc_vnncomp2022
$PYTHON generate_properties.py $SEED


# OVAL21
cd $INPUT_DIR
rm -rf oval21-benchmark
git clone https://github.com/alessandrodepalma/oval21-benchmark.git
cd oval21-benchmark
$PYTHON generate_properties.py $SEED
mv oval21_instances.csv instances.csv
sed -i -E 's|^([^,]+\.onnx),([^,]+\.vnnlib),|nets/\1,vnnlib/\2,|' instances.csv

# CIFAR100_TinyImageNet_ResNet
# cd $INPUT_DIR
# rm -rf CIFAR100_TinyImageNet_ResNet
# git clone https://github.com/Lucas110550/CIFAR100_TinyImageNet_ResNet.git
# cd CIFAR100_TinyImageNet_ResNet
# $PYTHON generate_properties.py $SEED
# awk -F',' 'BEGIN{OFS=","} { $1="onnx/"$1; $2="generated_vnnlib/"$2; print }' cifar100_tinyimagenet_instances.csv > instances.csv

# # SRI ResNet A/B
# cd $INPUT_DIR
# rm -rf sri_resnet_a
# git clone https://github.com/mnmueller/vnn_comp_22_bench_sri.git sri_resnet_a
# cd sri_resnet_a
# git checkout ResNet_A
# cd src
# $PYTHON generate_properties.py $SEED
# cd ..
# mv specs/instances.csv .

# cd $INPUT_DIR
# rm -rf sri_resnet_b
# git clone https://github.com/mnmueller/vnn_comp_22_bench_sri.git sri_resnet_b
# cd sri_resnet_b
# git checkout ResNet_B
# cd src
# $PYTHON generate_properties.py $SEED
# cd ..
# mv specs/instances.csv .


# Cifar biasfield 
cd $INPUT_DIR
rm -rf cifar_biasfield_vnncomp2022
git clone https://github.com/pat676/cifar_biasfield_vnncomp2022.git
cd cifar_biasfield_vnncomp2022
$PYTHON generate_properties.py $SEED
