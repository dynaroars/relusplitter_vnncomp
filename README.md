# ReluSplitter - VNNCOMP-25

## Usage 

To generate benchmark instances:

```bash
python3 generate_properties.py <SEED>
```

The script will install the tool, prepare the seed benchmarks, and generate new benchmarking instances. The resulting instances will be placed under `./Generated_Instances`. It was tested on AWS EC2 t2.large (Canonical, Ubuntu, 24.04, amd64 noble image). The Conda environment + seed instances + generated instance will take about 30 GB, so we recommend using an instance with 50 GB storage. The script should take ~35 min to execute.


## Benchmark Generation

This benchmark uses selected instances from previous VNNCOMP iterations as seed inputs to generate more challenging instances. The networks include both fully connected (FC) and convolutional (Conv) architectures. All networks use a standard feed-forward structure. The timeout for the instances were reduced based on results from prior iterations, and the generated instances have 3x timeout.

For each seed instance, we keep both the original seed instance and a generated instance for comparison.


## Seed Benchmarks Used:
For each benchmark, we used the `generate_property.py` script provided by the original author to generate the seed instances (except for ACAS Xu). Specifically, the following benchmarks were used:

<!-- - ACAS Xu (https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/acasxu)
    - <small> The networks in this benchmark were simplified (we merged the Matmul and Add nodes to GEMM nodes to make it compatible with higher onnx version) </small>
    - Sampled instances: 30
    - Timeout:
        - Original instances: 30s
        - Generated instances: 90s -->
- ACAS Xu (https://github.com/stanleybak/vnncomp2021/tree/main/benchmarks/acasxu)
    - removed due to onnx versioning issue.
- mnist_fc (https://github.com/pat676/mnist_fc_vnncomp2022.git)
    - Sampled instances: 30
    - Timeout:
        - Original instances: 30s
        - Generated instances: 90s
- Oval21 (https://github.com/alessandrodepalma/oval21-benchmark.git)
    - Sampled instances: 30
    - Timeout:
        - Original instances: 60s
        - Generated instances: 180s
- Cifar Biasfield (https://github.com/pat676/cifar_biasfield_vnncomp2022.git)
    - Sampled instances: 40
    - Timeout:
        - Original instances: 60s
        - Generated instances: 180s























# ReluSplitter

ReluSplitter is a DNN verification (DNNV) benchmark generation tool for ReLU-based network. It takes input as a DNNV instance (network + property) and generates a modified network s.t. the modified network carries the same semantic of the original network, but it is harder to verify the property on the modified network. ReluSplitter achieves this by systematically destabilizing stable neurons in the network. 

![Overview](stuff/figs/tool_overview.PNG)


## Features
Currently, our tool accepts input in the standard `onnx` and `vnnlib` format, and
supports two common layer types.
- Fully-connected layer with ReLU activation
- Convolutional layer with ReLU activation

Technically, any network with above layers can be split. However, you might encounter compatibility issues due to factors like ONNX version differences or complex network structures. If you believe your input should work but doesn't, feel free to [open an issue](#) or contact us.


### Why ReluSplitter?

- ✅ **Semantic-preserving**: The generated network always retains the same semantics as the original. For example, if the original instance is `UNSAT`, the modified instance will also be `UNSAT`.  
- ⚡ **Fast generation**: A new DNNV instance can be created within seconds—significantly faster than alternative benchmark generation approaches that require training or distillation.
