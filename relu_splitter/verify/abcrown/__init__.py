import pathlib
from ..verifier import Verifier

default_config = f"{pathlib.Path(__file__).parent}/onnx_with_one_vnnlib.yaml"

class AlphaBetaCrown(Verifier):
    name = "abcrown"
    @classmethod
    def _gen_prog(cls, prog_conf):
        model_path = prog_conf.get('onnx_path')
        property_path = prog_conf.get('vnnlib_path')
        config_path = prog_conf.get('config_path', default_config)

        cmd = f"{pathlib.Path(__file__).parent}/run_abcrown.sh"

        # device = prog_conf.get('device', None)
        # batch = prog_conf.get('batch', None)
        # disable_restart = prog_conf.get('disable_restart', None)
        timeout = prog_conf.get('timeout', None)

        # if device is not None:
        #     cmd += f" -d {device}"
        # if batch is not None:
        #     cmd += f" -b {batch}"
        # if disable_restart is not None:
        #     cmd += f" -r {disable_restart}"
        if timeout is not None:
            cmd += f" --timeout {timeout}"
        
        cmd += f" --onnx_path {model_path} --vnnlib_path {property_path} --config {config_path}"
        return cmd

    @classmethod
    def _analyze(cls, lines):
        veri_ans = None
        veri_time = None
        for l in lines:
            if "Result: " in l:
                veri_ans = l.strip().split()[-1]
                continue
            elif "Time: " in l: 
                # Only accept time if the result is found
                # This is to avoid the case where the time is printed before the result
                # ======
                # pruning-in-iteration extra time: 0.009193658828735352
                # Time: prepare 0.0028    bound 0.4125    transfer 0.0014    finalize 0.0047    func 0.4214    
                # Accumulated time: func 0.4214    prepare 0.0032    bound 0.4125    transfer 0.0014    finalize 0.0047    
                # Current worst splitting domains lb-rhs (depth):
                # ======
                if veri_ans:
                    veri_time = float(l.strip().split()[-1])
            elif 'CUDA out of memory' in l:
                veri_ans = 'out_of_memory'
                veri_time = -1

            # ERROR found by AdaGDVB
            elif "RuntimeError: cannot reshape tensor of 0 elements into shape [1, 0, -1] because the unspecified dimension size -1 can be any value and is ambiguous" in l:
                veri_ans = 'error'
                veri_time = -1
            elif "AttributeError: 'LiRPANet' object has no attribute 'split_indices'" in l:
                veri_ans = 'error'
                veri_time = -1
            elif "CUDA error: an illegal memory access was encountered" in l:
                veri_ans = 'error'
                veri_time = -1
            elif "TORCH_USE_CUDA_DSA" in l:
                veri_ans = 'error'
                veri_time = -1     
            if veri_ans and veri_time:
                break

        return veri_ans, veri_time