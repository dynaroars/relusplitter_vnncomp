import os
import sys
import random
import subprocess
import csv
from pathlib import Path

from relu_splitter.experiment_utils import get_layer_sizes

TOOL_ROOT   = Path(__file__).resolve().parent.parent

LIB_PATH    = TOOL_ROOT / "libs"
ENVS_PATH   = TOOL_ROOT / ".envs"
PYTHON_EXE  = ENVS_PATH / "ReluSplitter" / "bin" / "python"
ANYWHERE_MAIN = TOOL_ROOT / "anywhere_main.py"

SEED_BENCHMARK_DIR = TOOL_ROOT / "seed_benchmark"

OUTPUT_DIR        = TOOL_ROOT 
# OUTPUT_DIR        = TOOL_ROOT / "generated_benchmark"
ONNX_OUTPUT_DIR   = OUTPUT_DIR / "onnx"
VNNLIB_OUTPUT_DIR = OUTPUT_DIR / "vnnlib"
GENERATED_INSTANCES_CSV = OUTPUT_DIR / "instances.csv"

# ---------------------------------------------------------------------------
# Generation settings — adjust as needed
# ---------------------------------------------------------------------------
N_SAMPLES         = 20        # number of seed instances to sample
DESTAB_PERCENTS   = [0.2, 0.4, 0.6, 0.8, 1.0]   # fractions of neurons to destabilise
SEED_TIMEOUT      = 60        # timeout (s) written for seed instances
SPLIT_TIMEOUT_RATIO = 3       # generated instance timeout = SEED_TIMEOUT * ratio
MODE              = "gemm"    # "gemm" for FC networks, "conv" for CNNs
MAX_RETRY         = 5
# ---------------------------------------------------------------------------

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"


def sample_instances(benchmark_dir: Path, n: int, seed: int) -> None:
    """Sample n rows from instances.csv and write to sampled_instances.csv (no header)."""
    src = benchmark_dir / "instances.csv"
    assert src.exists(), f"No instances.csv found in {benchmark_dir}"

    with open(src, newline="") as f:
        rows = [r for r in csv.reader(f) if r and r[0].strip()]

    if len(rows) < n:
        print(f"[warn] only {len(rows)} instances available, sampling all")
    sampled = random.sample(rows, min(n, len(rows)))

    dst = benchmark_dir / "sampled_instances.csv"
    with open(dst, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(sampled)
    print(f"Sampled {len(sampled)}/{len(rows)} instances -> {dst}")


def load_seed_instances(benchmark_dir: Path) -> list[tuple[Path, Path]]:
    """Load (onnx_path, vnnlib_path) pairs from sampled_instances.csv (no header)."""
    csv_path = benchmark_dir / "sampled_instances.csv"
    assert csv_path.exists(), f"sampled_instances.csv not found in {benchmark_dir}"

    instances = []
    with open(csv_path, newline="") as f:
        for row in csv.reader(f):
            if not row or not row[0].strip():
                continue
            onnx_path   = Path(row[0].strip())
            vnnlib_path = Path(row[1].strip())
            if not onnx_path.is_absolute():
                onnx_path   = benchmark_dir / onnx_path
                vnnlib_path = benchmark_dir / vnnlib_path
            instances.append((onnx_path, vnnlib_path))

    print(f"Loaded {len(instances)} seed instances from {csv_path}")
    for onnx, vnnlib in instances:
        assert onnx.exists(),   f"Missing ONNX:   {onnx}"
        assert vnnlib.exists(), f"Missing vnnlib: {vnnlib}"
    return instances


def run_splitter(onnx: Path, vnnlib: Path, output: Path, split_idx: int, n: int, seed: int) -> bool:
    """Run split in-process (avoids per-call Python startup overhead). Returns True on success."""
    import torch
    from relu_splitter.anywhere import ReluSplitter_Anywhere
    from relu_splitter.utils.onnx_utils import check_models_closeness

    default_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        import logging
        from relu_splitter.utils.logger import default_logger
        default_logger.setLevel(logging.CRITICAL)

        rsa = ReluSplitter_Anywhere(str(onnx), str(vnnlib))
        conf = {
            "seed": seed,
            "split_activation": "relu",
            "n_splits": n,
            "create_baseline": False,
            "candidiate_strat": "random",
            "bounding_method_tight": "backward",
            "bounding_method_loose": "ibp",
            "param_conf": {
                "gemm_tau_strat": "random",
                "stable_tau_strat": "random",
                "stable_tau_margin": (5.0, 15.0),
                "cap_tau": 50.0,
                "split_scale_strat": "fixed",
                "fixed_scales": (1.0, -1.0),
                "random_scale_range": (0.1, 5.0),
            },
            "additional_activation_conf": {
                "leakyrelu_alpha": 0.01,
                "prelu_slope_range": (0.01, 0.25),
            },
        }
        new_model, baseline = rsa.split(MODE, split_idx, conf)
        new_model.save(output, ir_version=8)
        closeness = check_models_closeness(
            rsa.model, [new_model, baseline], rsa.input_shape,
            device=default_device, n=100, atol=5e-5, rtol=5e-5,
        )
        assert closeness[0][0] and closeness[1][0], "Closeness check failed"
        return True
    except Exception as e:
        print(f"[FAIL] {onnx.stem} n={n} seed={seed}: {e}")
        if output.exists():
            output.unlink()
        return False


def generate_for_instance(args):
    """Worker function for Pool.starmap. Returns list of (output_onnx, vnnlib) or []."""
    onnx, vnnlib, seed = args

    layer_dict  = get_layer_sizes("fc", onnx)   # {layer_idx: layer_size}
    layer_idx   = min(layer_dict.keys())
    layer_size  = layer_dict[layer_idx]

    onnx_stem   = onnx.stem.replace("~", "")
    vnnlib_stem = vnnlib.stem.replace("~", "")

    cnts = [max(1, int(layer_size * p)) for p in DESTAB_PERCENTS]
    out_fnames = [
        ONNX_OUTPUT_DIR / f"{onnx_stem}~{vnnlib_stem}~pct{pct}~cnt{cnt}~seed{seed}.onnx"
        for pct, cnt in zip(DESTAB_PERCENTS, cnts)
    ]

    params = [
        (onnx, vnnlib, fname, layer_idx, cnt, seed)
        for fname, cnt in zip(out_fnames, cnts)
    ]
    results = [run_splitter(*p) for p in params]

    retry = 0
    while not all(results):
        for fname in out_fnames:
            if fname.exists():
                fname.unlink()
        retry += 1
        if retry >= MAX_RETRY:
            print(f"[ERROR] Max retries reached for {onnx.stem}, skipping.")
            return []
        seed += 1
        print(f"  Retry {retry} for {onnx.stem} (new seed={seed})")
        params = [(onnx, vnnlib, fname, layer_idx, cnt, seed) for fname, cnt in zip(out_fnames, cnts)]
        results = [run_splitter(*p) for p in params]

    return list(zip(out_fnames, [vnnlib] * len(out_fnames)))


if __name__ == "__main__":
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Random seed: {seed}")
    random.seed(seed)

    ONNX_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    VNNLIB_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Sample N_SAMPLES instances from instances.csv -> sampled_instances.csv
    sample_instances(SEED_BENCHMARK_DIR, N_SAMPLES, seed)

    # Load the sampled instances
    seed_instances = load_seed_instances(SEED_BENCHMARK_DIR)

    # Copy seed onnx/vnnlib into output dirs
    import shutil
    staged = []
    for onnx, vnnlib in seed_instances:
        dst_onnx   = ONNX_OUTPUT_DIR   / onnx.name
        dst_vnnlib = VNNLIB_OUTPUT_DIR / vnnlib.name
        shutil.copy2(onnx,   dst_onnx)
        shutil.copy2(vnnlib, dst_vnnlib)
        staged.append((dst_onnx, dst_vnnlib))

    print(f"\nGenerating instances ({len(DESTAB_PERCENTS)} percentages x {len(staged)} seeds)...")

    final_instances = []   # (onnx, vnnlib, timeout) — all relative to OUTPUT_DIR

    for i, (onnx, vnnlib) in enumerate(staged):
        print(f"\n[{i+1}/{len(staged)}] {onnx.name}  x  {vnnlib.name}")

        # Original seed instance (included as-is)
        final_instances.append((
            onnx.relative_to(OUTPUT_DIR),
            vnnlib.relative_to(OUTPUT_DIR),
            SEED_TIMEOUT * SPLIT_TIMEOUT_RATIO,
        ))

        # Generated instances for each destabilisation percentage
        generated = generate_for_instance((onnx, vnnlib, seed))
        for gen_onnx, gen_vnnlib in generated:
            final_instances.append((
                gen_onnx.relative_to(OUTPUT_DIR),
                gen_vnnlib.relative_to(OUTPUT_DIR),
                SEED_TIMEOUT * SPLIT_TIMEOUT_RATIO,
            ))

    # Write instances.csv
    os.chdir(OUTPUT_DIR)
    with open(GENERATED_INSTANCES_CSV, "w") as f:
        for onnx_rel, vnnlib_rel, timeout in final_instances:
            assert Path(onnx_rel).exists(),   f"Missing: {onnx_rel}"
            assert Path(vnnlib_rel).exists(), f"Missing: {vnnlib_rel}"
            f.write(f"{onnx_rel},{vnnlib_rel},{timeout}\n")

    print(f"\nDone. {len(final_instances)} instances written to {GENERATED_INSTANCES_CSV}")
