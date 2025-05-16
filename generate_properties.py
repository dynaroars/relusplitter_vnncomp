import os
import subprocess
import sys
import random
from pathlib import Path




TOOL_NAME = "ReluSplitter"
TOOL_ROOT = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.path.join(TOOL_ROOT, "libs")
ENVS_PATH = os.path.join(TOOL_ROOT, ".envs")
PYTHON_EXE = os.path.join(ENVS_PATH, TOOL_NAME, "bin", "python")

INPUT_DIR = os.path.join(TOOL_ROOT, "Seed_Inputs")
SELECTED_INSTANCES_CSV = os.path.join(INPUT_DIR, "selected_instances.csv")

# OUTPUT_DIR = Path(TOOL_ROOT) / "Generated_Instances"
OUTPUT_DIR = Path(TOOL_ROOT)
ONNX_OUTPUT_DIR = Path(OUTPUT_DIR) / "onnx"
VNNLIB_OUTPUT_DIR = Path(OUTPUT_DIR) / "vnnlib"
GENERATED_INSTANCES_CSV = "instances.csv"

RANDOM_SEED = None

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(ONNX_OUTPUT_DIR, exist_ok=True)
os.makedirs(VNNLIB_OUTPUT_DIR, exist_ok=True)

os.environ["TOOL_NAME"] = TOOL_NAME
os.environ["TOOL_ROOT"] = TOOL_ROOT
os.environ["LIB_PATH"] = LIB_PATH
os.environ["PYTHONPATH"] = os.pathsep.join(
    [
        os.environ.get("PYTHONPATH", ""),
        LIB_PATH,
    ]
)
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"

acasxu_dir = Path(INPUT_DIR) / "acasxu_converted"
mnist_dir = Path(INPUT_DIR) / "mnist_fc_vnncomp2022"
oval21_dir = Path(INPUT_DIR) / "oval21-benchmark"
cifar_biasfield_dir = Path(INPUT_DIR) / "cifar_biasfield_vnncomp2022"


acasxu_num_samples = 0
mnist_num_samples = 40
oval21_num_samples = 30
cifar_biasfield_num_samples = 40

acasxu_timeout = 30
mnist_timeout = 30
oval21_timeout = 60
cifar_biasfield_timeout = 60

split_timeout_ratio = 3



def load_instances(benchmark_dir):
    instances_csv = benchmark_dir / "instances.csv"
    assert instances_csv.exists(), f"Cannot find {instances_csv}"
    with open(instances_csv, "r") as f:
        lines = f.readlines()
        
    instances = []
    for line in lines:
        cols = [i.strip() for i in line.split(",")]
        assert len(cols) == 3, f"Invalid line in {instances_csv}: {line}"
        onnx, vnnlib, timeout = benchmark_dir / cols[0], benchmark_dir / cols[1], int(float(cols[2]))
        instances.append((onnx, vnnlib, timeout, benchmark_dir.name))

    print(f"Loaded {len(instances)} instances from {instances_csv}")
    for onnx, vnnlib, timeout, _ in instances:
        assert onnx.exists(), f"Cannot find {onnx}"
        assert vnnlib.exists(), f"Cannot find {vnnlib}"

    return instances

def sample_instances_acasxu(acasxu_instances, n_samples):
    # only use prop 1,3,4
    num_per_vnnlib = n_samples // 3
    acasxu_instances = [i for i in acasxu_instances if "prop_10" not in i[1].stem]

    prop1_instances = [i for i in acasxu_instances if "prop_1" in i[1].stem]
    # prop2_instances = [i for i in acasxu_instances if "prop_2" in i[1].stem]
    prop3_instances = [i for i in acasxu_instances if "prop_3" in i[1].stem]
    prop4_instances = [i for i in acasxu_instances if "prop_4" in i[1].stem]
    sampled_instances = []
    sampled_instances.extend(random.sample(prop1_instances, num_per_vnnlib))
    # sampled_instances.extend(random.sample(prop2_instances, num_per_vnnlib))
    sampled_instances.extend(random.sample(prop3_instances, num_per_vnnlib))
    sampled_instances.extend(random.sample(prop4_instances, n_samples - 2 * num_per_vnnlib))
    return sampled_instances

def sample_instances_cifar_biasfield(cifar_biasfield_instances, n_samples):
    return random.sample(cifar_biasfield_instances, n_samples)

def sample_instances_mnist(mnist_instances, n_samples):
    num_per_model = n_samples // 2
    # x2_instances = [i for i in mnist_instances if "x2" in i[0].stem]
    x4_instances = [i for i in mnist_instances if "x4" in i[0].stem]
    x6_instances = [i for i in mnist_instances if "x6" in i[0].stem]
    sampled_instances = []
    # sampled_instances.extend(random.sample(x2_instances, num_per_model))
    sampled_instances.extend(random.sample(x4_instances, num_per_model))
    sampled_instances.extend(random.sample(x6_instances, n_samples - 1 * num_per_model))
    return sampled_instances

def sample_instances_oval21(oval21_instances, n_samples):
    num_per_model = n_samples // 3
    base_instances = [i for i in oval21_instances if "base" in i[0].stem]
    deep_instances = [i for i in oval21_instances if "deep" in i[0].stem]
    wide_instances = [i for i in oval21_instances if "wide" in i[0].stem]
    sampled_instances = []
    sampled_instances.extend(random.sample(base_instances, num_per_model))
    sampled_instances.extend(random.sample(deep_instances, num_per_model))
    sampled_instances.extend(random.sample(wide_instances, n_samples - 2 * num_per_model))
    return sampled_instances

def get_timeout(onnx, vnnlib):
    if "ACASXU" in str(onnx):
        return acasxu_timeout
    elif "cifar_bias_field" in str(onnx):
        return cifar_biasfield_timeout
    elif "mnist-net_256" in str(onnx):
        return mnist_timeout
    elif "cifar_base_kw" in str(onnx) or "cifar_deep_kw" in str(onnx) or "cifar_wide_kw" in str(onnx):
        return oval21_timeout
    else:
        raise ValueError(f"Unknown timeout for {onnx}")

def get_split_mode(onnx, vnnlib):
    if "ACASXU" in onnx.stem or "mnist-net_256" in onnx.stem:
        return "fc"
    elif "cifar" in onnx.stem or "resnet" in onnx.stem:
        return "conv"
    else:
        raise ValueError(f"Unknown mode for {onnx}")

def run_splitter(onnx, vnnlib, mode, seed, output_onnx, output_vnnlib):
    cmd = [
            PYTHON_EXE, os.path.join(os.environ["TOOL_ROOT"], "main.py"), "split",
            "--net", str(onnx),
            "--spec", str(vnnlib),
            "--mode", mode,
            # "--n_splits", n_splits,
            "--output", str(output_onnx),
            "--seed", str(seed),
        ]
    # print(" ".join(cmd))
    try:
        res = subprocess.run(cmd, check=True, env=os.environ, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    except subprocess.CalledProcessError as e:
        print("Command failed:")
        print("Return code:", e.returncode)
        print("stdout:\n", e.stdout)
        print("stderr:\n", e.stderr)
        raise  # optional: remove if you don't want to crash
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        raise

def rsplit_gen_instance(onnx, vnnlib, benchmark, timeout, seed):
    # run tool to generate new instances
    # include original instances
    # generated instances will have longer timeout
    timeout = get_timeout(onnx, vnnlib)
    mode = get_split_mode(onnx, vnnlib)

    output_fname_onnx = f"{onnx.stem}_RSPLITTER_{vnnlib.stem}.onnx"
    output_fname_vnnlib = None   # Not Used

    output_path_onnx = ONNX_OUTPUT_DIR / f"{output_fname_onnx}"
    output_path_vnnlib = None

    print(f"\t\tGenerating onnx for input {onnx.stem}, {vnnlib.stem}...")
    run_splitter(onnx, vnnlib, mode, seed, output_path_onnx, output_path_vnnlib)

    return (onnx, vnnlib, timeout), (output_path_onnx, vnnlib, split_timeout_ratio * timeout)


def make_banner(text):
    print("=" * 80)
    print("=" * 80)
    
    print("""
.______       _______  __       __    __       _______..______    __       __  .___________.___________. _______ .______      
|   _  \     |   ____||  |     |  |  |  |     /       ||   _  \  |  |     |  | |           |           ||   ____||   _  \     
|  |_)  |    |  |__   |  |     |  |  |  |    |   (----`|  |_)  | |  |     |  | `---|  |----`---|  |----`|  |__   |  |_)  |    
|      /     |   __|  |  |     |  |  |  |     \   \    |   ___/  |  |     |  |     |  |        |  |     |   __|  |      /     
|  |\  \----.|  |____ |  `----.|  `--'  | .----)   |   |  |      |  `----.|  |     |  |        |  |     |  |____ |  |\  \----.
| _| `._____||_______||_______| \______/  |_______/    | _|      |_______||__|     |__|        |__|     |_______|| _| `._____|
                                                                                                                              
       ======================================================================================                             
____    ____ .__   __. .__   __.          ______   ______   .___  ___. .______    ___    _____                                
\   \  /   / |  \ |  | |  \ |  |         /      | /  __  \  |   \/   | |   _  \  |__ \  | ____|                               
 \   \/   /  |   \|  | |   \|  |  ______|  ,----'|  |  |  | |  \  /  | |  |_)  |    ) | | |__                                 
  \      /   |  . `  | |  . `  | |______|  |     |  |  |  | |  |\/|  | |   ___/    / /  |___ \                                
   \    /    |  |\   | |  |\   |        |  `----.|  `--'  | |  |  |  | |  |       / /_   ___) |                               
    \__/     |__| \__| |__| \__|         \______| \______/  |__|  |__| | _|      |____| |____/                                
                                                                                                                              
    """)

    print("=" * 80)
    print("\n")
    print(text)
    print(text)
    print(text)
    print("\n")
    print("=" * 80)
    print("=" * 80)



        

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_properties.py <RANDOM_SEED>")
        sys.exit(1)

    assert sys.argv[1].isdigit(), f"Random seed must be an integer, got {sys.argv[1]}"
    RANDOM_SEED = sys.argv[1]
    print(f"Generating instances with random seed {RANDOM_SEED}")


    # run scripts/install.sh
    make_banner("Installing...")
    subprocess.run(["./scripts/install.sh"])

    os.environ["RANDOM_SEED"] = RANDOM_SEED
    random.seed(int(RANDOM_SEED))

    # run prep_benchmark
    # ./scripts/prep_benchmarks.sh random_seed
    make_banner("Preparing seed benchmarks...")
    subprocess.run(["./scripts/prep_benchmarks.sh", str(RANDOM_SEED)])


    make_banner("Starting benchmark generation...")

    # Load instances from the specified directories
    print("Loading instances...")
    acasxu_instances = load_instances(acasxu_dir)
    cifar_biasfield_instances = load_instances(cifar_biasfield_dir)
    mnist_instances = load_instances(mnist_dir)
    oval21_instances = load_instances(oval21_dir)


    # sample seed instances from each
    print("Sampling instances...")
    acasxu_samples = sample_instances_acasxu(acasxu_instances, acasxu_num_samples)
    cifar_biasfield_samples = sample_instances_cifar_biasfield(cifar_biasfield_instances, cifar_biasfield_num_samples)
    mnist_samples = sample_instances_mnist(mnist_instances, mnist_num_samples)
    oval21_samples = sample_instances_oval21(oval21_instances, oval21_num_samples)

    selected_instances = acasxu_samples + mnist_samples + oval21_samples + cifar_biasfield_samples
    
    # write to selected_instances.csv
    with open(SELECTED_INSTANCES_CSV, "w") as f:
        for onnx, vnnlib, timeout, _ in selected_instances:
            f.write(f"{onnx},{vnnlib},{timeout}\n")

    # copy original instances to Generated_Instances
    print("Moving sampled seed instances to Generated_Instances...")
    os.system(f"rm -rf {ONNX_OUTPUT_DIR} {VNNLIB_OUTPUT_DIR}")
    os.system(f"mkdir -p {ONNX_OUTPUT_DIR} {VNNLIB_OUTPUT_DIR}")
    for onnx, vnnlib, timeout, benchmark in selected_instances:
        onnx_dir = ONNX_OUTPUT_DIR / f"{benchmark}_{onnx.stem}.onnx"
        vnnlib_dir = VNNLIB_OUTPUT_DIR / f"{benchmark}_{vnnlib.stem}.vnnlib"
        os.system(f"cp {onnx} {onnx_dir}")
        os.system(f"cp {vnnlib} {vnnlib_dir}")

    selected_instances = [( ONNX_OUTPUT_DIR / f"{benchmark}_{onnx.stem}.onnx", VNNLIB_OUTPUT_DIR / f"{benchmark}_{vnnlib.stem}.vnnlib", timeout, benchmark) for onnx, vnnlib, timeout, benchmark in selected_instances]

    for onnx, vnnlib, timeout, _ in selected_instances:
        assert onnx.exists(), f"Cannot find {onnx}"
        assert vnnlib.exists(), f"Cannot find {vnnlib}"
    

    print("Generating new instances...")
    final_instances = []
    # run tool to generate new instances
    # for onnx, vnnlib, timeout, benchmark in selected_instances:
    for i, (onnx, vnnlib, timeout, benchmark) in enumerate(selected_instances):
        print(f"{i+1}/{len(selected_instances)}: Generating new instances for {onnx.stem}, {vnnlib.stem}...")
        # gives two tuples
        # the timeout of original instance might be changed
        original_instance, new_instance = rsplit_gen_instance(onnx, vnnlib, benchmark, timeout, RANDOM_SEED)
        original_instance = (original_instance[0].relative_to(OUTPUT_DIR), original_instance[1].relative_to(OUTPUT_DIR), original_instance[2])
        new_instance = (new_instance[0].relative_to(OUTPUT_DIR), new_instance[1].relative_to(OUTPUT_DIR), new_instance[2])
        final_instances.append(original_instance)
        final_instances.append(new_instance)

    # write to generated_instances.csv
    os.chdir(OUTPUT_DIR)
    with open(GENERATED_INSTANCES_CSV, "w") as f:
        for onnx, vnnlib, timeout in final_instances:
            assert onnx.exists(), f"Cannot find {onnx}"
            assert vnnlib.exists(), f"Cannot find {vnnlib}"
            assert isinstance(timeout, int), f"Timeout must be an integer, got {timeout}"
            print(f"Writing {onnx}, {vnnlib}, {timeout} to {GENERATED_INSTANCES_CSV}")
            f.write(f"{onnx},{vnnlib},{timeout}\n")

    make_banner("Done!")
    print(f"Generated instances are saved in {OUTPUT_DIR}")


