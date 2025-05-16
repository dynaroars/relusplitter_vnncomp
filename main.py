from relu_splitter.utils.logger import logger

import torch
import onnx
import argparse
import sys
from pathlib import Path
from termcolor import colored


from relu_splitter.core import ReluSplitter
from relu_splitter.model import WarppedOnnxModel
from relu_splitter.verify import init_verifier


default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_parser():
    parser = argparse.ArgumentParser(description='Parser for network verification arguments')
    parser.add_argument('--verbosity', type=int, default=20, help='Verbosity level (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR, 50=CRITICAL)')

    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')

    # Subparser for the split command
    split_parser = subparsers.add_parser('split', help='split command help')

    # shared parameters
    split_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    split_parser.add_argument('--spec', type=str, required=True, help='Path to the VNNLIB file')
    split_parser.add_argument('--output', type=str, default='splitted.onnx', help='Output path for the new model')
    split_parser.add_argument('--input_shape', type=int, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')
    split_parser.add_argument('--bounding_method', type=str, default='backward', help='Bounding method to be used with auto-LiRPA, see https://auto-lirpa.readthedocs.io/en/latest/api.html')
    split_parser.add_argument('--create_baseline', action='store_true', help='Create baseline model')
    split_parser.add_argument('--closeness_check', action='store_true', help='Enable closeness check')
    split_parser.add_argument('--atol', type=float, default=1e-4, help='Absolute tolerance for closeness check')
    split_parser.add_argument('--rtol', type=float, default=1e-4, help='Relative tolerance for closeness check')
    split_parser.add_argument('--mode', type=str, default='fc', help='Mode for splitting',
                              choices=['fc', 'conv', 'all'])
    split_parser.add_argument('--bias_method', type=str, default='normal', help='Method for bias computation', choices=['normal', 'optimized'])


    
    split_parser.add_argument('--split_idx', type=int, default=0, help='Index of the layer to split')
    split_parser.add_argument('--mask', type=str, default='stable+', help='Mask for splitting',
                              choices=['stable+', 'stable-', 'stable', 'unstable', 'all', 'unstable_n_stable+'])
    split_parser.add_argument('--n_splits', type=int, default=None, help='Number of splits (strict)')
    split_parser.add_argument('--scale_factor', type=float, nargs=2, default=[1.0,-1.0], help='Scale factor for the split')
    
    # split_parser.add_argument('--device', type=str, default=default_device, help='Device for the model closeness check',)
    # conv parameters
    # split_parser.add_argument('--conv_strategy', type=str, default='random', help='Splitting strategy',
    #                           choices=['random', 'reluS+', 'reluS-'])
    # fc parameters
    # split_parser.add_argument('--fc_strategy', type=str, default='random', help='Splitting strategy',
    #                           choices=['single', 'random', 'reluS+', 'reluS-', 'adaptive'])

    
    split_parser.add_argument('--seed', type=int, default=0, help='Seed for random number generation')

    split_parser.add_argument('--verify', type=str, default=False, help='run verification with verifier',
                              choices=['neuralsat', 'abcrown', 'marabou', 'nnenum'])

    # Subparser for the info command
    info_parser = subparsers.add_parser('info', help='Info command help')
    info_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    info_parser.add_argument('--spec', type=str, required=False, help='Path to the VNNLIB file')
    info_parser.add_argument('--input_shape', type=int, required=False, nargs='+', default=None, help='Optional input shape of the model (e.g., 1 3 224 224)')


    # Subparser for the baseline command
    baseline_parser = subparsers.add_parser('baseline', help='baseline command help')
    baseline_parser.add_argument('--net', type=str, required=True, help='Path to the ONNX file')
    baseline_parser.add_argument('--output', type=str, default='baseline.onnx', help='Output path for the new model')
    baseline_parser.add_argument('--n_splits', type=int, default=None, help='Number of splits')
    baseline_parser.add_argument('--split_idx', type=int, default=0, help='Index for splitting')
    baseline_parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance for closeness check')
    baseline_parser.add_argument('--rtol', type=float, default=1e-5, help='Relative tolerance for closeness check')

    return parser


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    logger.setLevel(args.verbosity)

    if args.command == 'split':
        onnx_path = Path(args.net)
        spec_path = Path(args.spec)
        output_path = Path(args.output)
        input_shape = args.input_shape

        conf = {
            'split_mask': args.mask,
            'fc_strategy': "random",
            'conv_strategy': "max_unstable",
            'n_splits': args.n_splits,
            'split_idx': args.split_idx,
            'scale_factor': args.scale_factor,
            'random_seed': args.seed,
            'atol': args.atol,
            'rtol': args.rtol,
            # 'device': args.device,
            'create_baseline': args.create_baseline,
            'closeness_check': args.closeness_check,
            'bounding_method': args.bounding_method,
            'bias_method': args.bias_method,
        }

        relusplitter = ReluSplitter(onnx_path, 
                                    spec_path, 
                                    input_shape = input_shape,
                                    logger = logger)

        new_model, baseline = relusplitter.split(args.split_idx, args.mode , conf)
        new_model.save(output_path)
        logger.info(f'Split model saved to {output_path}')
        if args.create_baseline:
            baseline_path = output_path.with_name(output_path.stem + '_baseline.onnx')
            baseline.save(baseline_path)
            logger.info(f'Baseline model saved to {baseline_path}')

        if args.verify:
            verifier = init_verifier(args.verify)
            verifier.set_logger(logger)
            # abc_conf_path = "/home/lli/tools/relusplitter/experiment/config/mnistfc.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp23/tllVerifyBench.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/cifar2020_2_255.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/experiment/config/reach_probability.yaml"
            abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/oval22.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/collins-rul-cnn.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/cifar_biasfield.yaml"
            # abc_conf_path = "/home/lli/tools/relusplitter/libs/alpha-beta-CROWN/complete_verifier/exp_configs/vnncomp22/vggnet16.yaml"
            conf = {
                'onnx_path': onnx_path,
                'vnnlib_path': spec_path,
                'log_path': Path('veri_1.log'),
                'verbosity': 1,
                'num_workers': 12,
                'config_path': abc_conf_path,
            }
            logger.info(f'Start verification using {args.verify}')

            print("Original instance:")
            print(colored(verifier.execute(conf), 'green'))

            conf['onnx_path'], conf['log_path'] = output_path, Path('veri_2.log')
            print("Splitted instance:")
            print(colored(verifier.execute(conf), 'yellow'))
            
            if args.create_baseline:
                conf['onnx_path'], conf['log_path'] = baseline_path, Path('veri_3.log')
                print("Baseline instance:")
                print(colored(verifier.execute(conf), 'blue'))
        logger.info(f'=== Done ===')

    elif args.command == 'info':
        if args.spec is not None:
            ReluSplitter.info(Path(args.net), Path(args.spec), args.input_shape)
        else:
            ReluSplitter.info_net_only(Path(args.net))
    elif args.command == 'baseline':
        onnx_path = Path(args.net)
        output_path = Path(args.output)
        new_model = ReluSplitter.get_fc_baseline_split(onnx_path, args.n_splits, args.split_idx, args.atol, args.rtol)
        new_model.save(output_path)
    else:
        parser.print_help()
        sys.exit(1)



