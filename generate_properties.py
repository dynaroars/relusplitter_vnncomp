import os
import sys
import subprocess
from pathlib import Path

_TOOL_ROOT = Path(__file__).resolve().parent
_LIB_PATH  = _TOOL_ROOT / "libs"
_UV        = _TOOL_ROOT / ".envs" / "uv" / "bin" / "uv"
_VENV_DIR  = _TOOL_ROOT / ".envs" / "ReluSplitter"
_PYTHON    = _VENV_DIR / "bin" / "python"

if not _UV.exists() or not _VENV_DIR.exists():
    subprocess.run([str(_TOOL_ROOT / "scripts" / "install.sh")], check=True)

os.environ["PYTHONPATH"] = os.pathsep.join(
    filter(None, [
        str(_LIB_PATH / "auto_LiRPA"),  # must precede _LIB_PATH to avoid namespace-package shadowing
        str(_LIB_PATH),
        str(_TOOL_ROOT),
        os.environ.get("PYTHONPATH", ""),
    ])
)

seed = sys.argv[1] if len(sys.argv) > 1 else "42"
subprocess.run([str(_PYTHON), str(_TOOL_ROOT / "scripts" / "gen_props.py"), seed], check=True)
