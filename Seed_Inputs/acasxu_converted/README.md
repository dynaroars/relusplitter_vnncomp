# ACASXU_converted

This repository contains converted versions of the ACAS Xu networks. The original networks have been updated to a more up-to-date IR version (ONNX IR version 4) using onnxsim. Additionally, the original MatMul and Add layers have been combined into Gemm layers to improve compatibility. The specification files (vnnlib) are the same.

To ensure correctness, each model is validated against the original model using 100 randomly generated inputs, with the absolute and relative tolerances for output errors set to 1e-6.

See `/scripts/acasxu_convert.py` for details.

```
These is the acasxu benchmark category. The folder contains .onnx and .vnnlib files used for the category. The acasxu_instances.csv containts the full list of benchmark instances, one per line: onnx_file,vnn_lib_file,timeout_secs
 
This benchmark uses the ACAS Xu networks (from "Reluplex: An efficient SMT solver for verifying deep neural networks"), properties 1-4 run on all networks.

The .vnnlib and .csv files were created with the included generate.py script.

This benchmark is the same as in 2021/2022, and used to compare year-to-year improvements.

```