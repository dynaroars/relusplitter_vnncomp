from pathlib import Path
from relu_splitter.anywhere import ReluSplitter_Anywhere
import sys
import os
from pathlib import Path
import torch
import argparse

from relu_splitter.utils.logger import default_logger as logger

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_parser():
    parser = argparse.ArgumentParser(description='ReLUSplitter (Anywhere)')
    parser.add_argument("--verbosity", type=int, default=20, help="Logging verbosity level (10:DEBUG, 20:INFO, 30:WARNING, 40:ERROR)")

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    info_parser = subparsers.add_parser('info', help='Display model information')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')

    split_parser = subparsers.add_parser('split', help='Split ReLU activations in the model')
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--output', type=str, default='splitted.onnx', help='Output path for the new model')
    split_parser.add_argument('--output_ir_version', type=int, default=None, help='IR version for the output model (e.g., 7, 8). If not set, uses onnx default.')
    split_parser.add_argument('--baseline', type=str, default=None, help='Output path for the baseline model')
    split_parser.add_argument('--input_shape', type=int, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')
    split_parser.add_argument('--mode', type=str, default='all', help='Mode for splitting',
                              choices=['gemm', 'conv', 'all'])
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index of the layer to split')
    split_parser.add_argument('-n', type=int, default=None, help='Number of destablizing splits')
    split_parser.add_argument('--seed', type=int, default=35, help='Seed for random number generation')
    split_parser.add_argument('--activation', type=str, default='relu', help='Activation function to split',
                                choices=['relu', 'leakyrelu', 'prelu'])
    
    split_parser.add_argument("--candidiate_strat", type=str, default="random", help="Strategy for selecting candidate neurons to split. Options: random, bound_width",
                                choices=["random", "bound_width"])
    
    # split_conf
    # if error raise when splitting on conv, try set '--bounding_method_tight ibp'. 
    split_parser.add_argument("--bounding_method_tight", type=str, default="backward", help="Bounding method for tight bounds in stable neuron handling. Options: backward, ibp")
    split_parser.add_argument("--bounding_method_loose", type=str, default="ibp", help="Bounding method for loose bounds in stable neuron handling. Options: backward, ibp")

    split_parser.add_argument("--gemm_tau_strat", type=str, default="random", help="Strategy for selecting tau in Gemm splitting. Options: random, midpoint",
                                choices=["random", "midpoint"])
    split_parser.add_argument("--stable_tau_strat", type=str, default="random", help="Strategy for selecting tau in stable neuron handling. Options: random, BigTau, SmallTau",
                                choices=["random", "big", "small"])
    split_parser.add_argument("--stable_tau_margin", type=float, nargs=2, default=(5.0, 15.0), help="Margin for stable tau selection (min, max), e.g. (10.0, 50.0) -> tau in [10.0, 50.0]")
    split_parser.add_argument("--cap_tau", type=float, default=50.0, help="Cap for tau value to avoid numerical instability in splitting")

    split_parser.add_argument("--scale_strat", type=str, default="fixed", help="Strategy for selecting scale in Gemm splitting. Options: random, fixed",
                                choices=["random", "fixed"])
    split_parser.add_argument("--fixed_scale", type=float, nargs=2, default=(1.0, -1.0), help="If provided, use this fixed scale for all splits instead of random sampling, default is (1.0, -1.0)")
    split_parser.add_argument("--random_scale_range", type=float, nargs=2, default=(0.1, 5.0), help="If scale_strat is random, sample scale from this range (min, max), e.g. (0.1, 100.0) -> spos in [0.1, 100.0], sneg in [-100.0, -0.1]")


    split_parser.add_argument('--leakyrelu_alpha', type=float, default=0.01, help='Alpha value for LeakyReLU if selected as activation to split')
    split_parser.add_argument('--prelu_slope_range', type=float, nargs=2, default=(0.01, 0.25), help='Range for sampling PReLU slope if selected as activation to split, e.g. (0.1, 0.5) -> slope in [0.1, 0.5]')
    
    
    split_parser.add_argument('--closeness_check', action='store_true', default=True, help='Enable closeness check between original and splitted models')

    
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if args.command == 'info':
        ReluSplitter_Anywhere.info(args.net)

    elif args.command == 'split':
        rsa = ReluSplitter_Anywhere(args.net, args.spec, input_shape=args.input_shape)
        conf = {
            "seed": args.seed,
            "split_activation": args.activation,
            "n_splits": args.n,
            "create_baseline": args.baseline is not None,
            "candidiate_strat": args.candidiate_strat,
            "bounding_method_tight": args.bounding_method_tight,
            "bounding_method_loose": args.bounding_method_loose,
            "param_conf": {
                "gemm_tau_strat": args.gemm_tau_strat,
                "stable_tau_strat": args.stable_tau_strat,
                "stable_tau_margin": args.stable_tau_margin,
                "cap_tau": args.cap_tau,
                "split_scale_strat": args.scale_strat,
                "fixed_scales": args.fixed_scale,
                "random_scale_range": args.random_scale_range
            },
            "additional_activation_conf":
            {
                "leakyrelu_alpha": args.leakyrelu_alpha,
                "prelu_slope_range": args.prelu_slope_range
            }
        }
        new_model, baseline = rsa.split(args.mode, args.split_idx, conf)
        new_model.save(Path(args.output), ir_version=args.output_ir_version)
        print(f"Saved splitted model to {args.output}")
        if baseline is not None and args.baseline is not None:
            baseline.save(Path(args.baseline), ir_version=args.output_ir_version)
            print(f"Saved baseline model to {args.baseline}")
        
        if args.closeness_check:
            from relu_splitter.utils.onnx_utils import check_models_closeness
            closeness_results = check_models_closeness(
                rsa.model,
                [new_model, baseline],
                rsa.input_shape,
                device=default_device,
                n=100,
                atol=5e-5,
                rtol=5e-5
            )
            logger.info(f"Performed closeness check with 10 random samples on device {default_device}.")
            logger.info(f"Closeness results [Split]: {closeness_results[0]}")
            logger.info(f"Closeness results [Baseline]: {closeness_results[1]}")
            assert closeness_results[0][0] and closeness_results[1][0], "Closeness check failed!"

    else:
        parser.print_help()



