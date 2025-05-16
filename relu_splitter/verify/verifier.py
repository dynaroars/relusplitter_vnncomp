import subprocess
import sys
import logging
import os
from pathlib import Path


RESMONITOR_PATH = f"{os.environ.get('LIB_PATH')}/resmonitor.py"


TimeoutError = "timeout"
OutOfMemoryError = "memout"
ModelNotFoundError = "instance_not_generated"
VnnlibNotFoundError = "vnnlib_not_generated"
UnknownError = "error"
VerifiedResult = "unsat"
FalsifiedResult = "sat"
UnknownResult = "unknown"



class Verifier:
    logger = logging.getLogger(__name__)

    @classmethod
    def set_logger(cls, logger):
        cls.logger = logger

    @classmethod
    def _gen_prog(cls, prog_conf):
        raise NotImplementedError("_gen_prog not implemented in sub class {cls.__name__}")

    @classmethod
    def gen_prog(cls, prog_conf):
        ram  = prog_conf.get('ram', None)
        time = prog_conf.get('timeout', None)
        verbosity = prog_conf.get('verbosity', 1)
        prog = cls._gen_prog(prog_conf)


        cmd = f"python3 {RESMONITOR_PATH}"

        if ram is not None:
            cmd += f" -M {ram}"
        if time is not None:
            cmd += f" -T {int(time)}"
        if verbosity == 0:
            cmd += " -q"
        elif verbosity == 1:
            cmd += " -v"

        cmd += f" {prog}"

        return cmd

    @classmethod
    def execute(cls, prog_conf):
        force_rerun = prog_conf.get('force_rerun', False)
        if force_rerun == True:
            cls.logger.info(f"Force rerun is set to True, rerunning verification")
        if force_rerun == True or cls.finished(prog_conf) == False:
            cmd = cls.gen_prog(prog_conf)
            log_path = Path(prog_conf.get('log_path'))
            cls.logger.debug(f"config: {prog_conf}")
            cls.logger.info(f"Executing verification ... log path: {log_path}")
            cls.logger.info(cmd)

            log_path.parent.mkdir(parents=True, exist_ok=True)
            with open(log_path, "w") as veri_log_fp:
                veri_log_fp.write(f"[LLI-verify] ONNX:{prog_conf['onnx_path']}, VNNLIB:{prog_conf['vnnlib_path']}\n")
                sp = subprocess.Popen(cmd, shell=True, stdout=veri_log_fp, stderr=veri_log_fp)

                cls.logger.debug(f"Verification process pid: {sp.pid}, waiting...")
                sp.wait()
                veri_log_fp.write(f"[LLI-verify] Verification finished ({sp.returncode})\n")
            if sp.returncode != 0:
                cls.logger.error(f"Verification failed with return code {sp.returncode}")
                cls.logger.error(f"Verification log path: {log_path}")
                return None
            else:
                cls.logger.info("Verification finished successfully")
                return cls.analyze(prog_conf)
        else:
            cls.logger.info(f"Verification already finished, skipping execution [{prog_conf['log_path']}]")
            return cls.analyze(prog_conf)

            

    @classmethod
    def finished(cls, prog_conf):
        log_path = prog_conf.get('log_path')
        if not os.path.exists(log_path):
            cls.logger.info(f"Log file {log_path} does not exist")
            return False
        with open(log_path, "r") as fp:
            lines = fp.readlines()
        if len(lines) == 0:
            cls.logger.info(f"Log file {log_path} is empty")
            return False
        for l in lines:
            if "[LLI-verify] Verification finished" in l:
                cls.logger.info(f"Log file {log_path} is finished")
                return True
        cls.logger.info(f"Log file {log_path} is not finished")
        return False


    @classmethod
    def _analyze(cls, lines):
        raise NotImplementedError("_analyze not implemented in sub class {cls.__name__}")
    
    @classmethod
    def pre_analyze(cls, prog_conf):
        veri_ans = None
        veri_time = None

        with open(prog_conf['log_path'], "r") as fp:
            lines = fp.readlines()

        # TODO: fix this with resmonitor -lli
        # for now just get the timeout value from the config
        for l in lines:
            if "Timeout (terminating process)" in l:
                veri_ans = "timeout"
                # veri_time = float(l.strip().split()[-1])
                veri_time = prog_conf['timeout']
            elif "Out of Memory" in l:
                veri_ans = "memout"
                # veri_time = float(l.strip().split()[-1])
                veri_time = prog_conf['timeout']

        return veri_ans, veri_time

    @classmethod
    def get_exec_time(cls, lines):
        for l in lines:
            if " (resmonitor) Process finished successfully" in l:
                return float(l.split()[-3].split("s")[0])
        cls.logger.warning(f"Execution time not found in {log_path}")
        return None


    @classmethod
    def analyze(cls, prog_conf):
        log_path = prog_conf.get('log_path')

        with open(log_path, "r") as fp:
            lines = fp.readlines()

        veri_ans, veri_time = cls.pre_analyze(prog_conf)
        if veri_ans is not None:
            return veri_ans, veri_time
        else:
            return cls._analyze(lines)

        if veri_ans in [FalsifiedResult, VerifiedResult, UnknownResult]:
            pass
        elif veri_ans == "timeout":
            veri_ans = TimeoutError
        elif veri_ans == "out_of_memory":
            veri_ans = OutOfMemoryError
        elif veri_ans == "model_not_exist":
            veri_ans = ModelNotFoundError
        elif veri_ans == "vnnlib_not_exist":
            veri_ans = VnnlibNotFoundError
        elif veri_ans == "error":
            veri_ans = UnknownError
        elif veri_ans is None:
            cls.logger.error(f"veri_ans is None in {log_path}")
            veri_ans = UnknownError
        else:
            cls.logger.error(f"unknown veri_ans: {veri_ans} in {log_path}")
            veri_ans = UnknownError

        
        if veri_time is None:
            cls.logger.error(f"veri_time is None in {log_path}")
            print(veri_ans)

        if veri_ans != 'timeout' and veri_time > prog_conf['timeout']:
            veri_ans = 'timeout'
            veri_time = prog_conf['timeout']

        return veri_ans, veri_time   

