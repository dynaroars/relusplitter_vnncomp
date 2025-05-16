import pathlib
from ..verifier import Verifier


class NNEnum(Verifier):
    name = "nnenum"
    @classmethod
    def _gen_prog(cls, prog_conf):
        model_path = prog_conf.get('onnx_path')
        property_path = prog_conf.get('vnnlib_path')

        cmd = f"{pathlib.Path(__file__).parent}/run_nnenum.sh {model_path} {property_path}"

        return cmd

    @classmethod
    def _analyze(cls, lines):
        veri_ans = None
        veri_time = None

        for l in lines[:]:
            if "Result: network is SAFE" in l:
                veri_ans = "unsat"
            elif "Result: network is UNSAFE with confirmed counterexample" in l:
                veri_ans = "sat"

            if "Runtime:" in l:
                if "(" not in l:
                    veri_time = float(l.split()[-2])
                else:
                    veri_time = float(l[str.index(l, "(") + 1 :].split()[0])

            if "reached during execution" in l:
                veri_ans = "timeout"
                veri_time = float(l[str.index(l, "(") + 1 : str.index(l, ")")])

            error_pattern = [
                "FloatingPointError: underflow encountered in multiply",
                "underflow encountered in divide",
                "Exception occured during execution"
            ]
            if any([True for x in error_pattern if x in l]):
                veri_ans = "error"
                veri_time = -1

            if "Proven safe before enumerate" in l:
                veri_ans = "sat"
                veri_time = cls.get_exec_time(lines)

            if veri_ans and veri_time:
                break
            

        return veri_ans, veri_time