import os
import sys
import random
import subprocess
import csv
from pathlib import Path
from multiprocessing import Pool, cpu_count
import sys


if not Path(".envs/ReluSplitter").exists():
    subprocess.run(["./scripts/install.sh"])


_TOOL_ROOT = Path(__file__).resolve().parent
_LIB_PATH  = _TOOL_ROOT / "libs"
os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, [str(_LIB_PATH), str(_TOOL_ROOT), os.environ.get("PYTHONPATH", "")])
)
sys.path.insert(0, str(_LIB_PATH))



seed = sys.argv[1] if len(sys.argv) > 1 else "42"
subprocess.run([".envs/ReluSplitter/bin/python", "./scripts/gen_props.py", seed])