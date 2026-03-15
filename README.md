# ReluSplitter - VNNCOMP-26

## Usage 

To generate benchmark instances:

```bash
python3 generate_properties.py <SEED>
```


The script will install the tool, sample the seed benchmarks, and generate new benchmarking instances. The generated instances will be saved to the repository root under `./onnx`, `./vnnlib`, and `instances.csv`.

## Benchmark Generation


This benchmark uses a small set of MNIST-based instances as seeds to generate more challenging test cases. The networks are small Fully-Connected FNNs, and the properties are local robustness on correctly classified handwritten digit images.

20 instances are sampled at random as seeds. For each seed, ReluSplitter generates five additional models with 20%, 40%, 60%, 80%, and 100% unstable neurons in the first layer, respectively. These models are verified against the same property as their corresponding seed.

All instances use a uniform 180s timeout.

# ReluSplitter

ReluSplitter is a DNN verification (DNNV) benchmark generation tool for ReLU-based networks. It takes a DNNV instance (network + property) as input and generates a modified network that maintains the same semantics as the original but is significantly harder to verify. ReluSplitter achieves this by systematically destabilizing stable neurons within the network.

- [GitHub](https://github.com/dynaroars/relusplitter)
- [Paper](https://lli-debu.top/pubs/ASE25/ASE_main-980.pdf)


