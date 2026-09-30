"""Run an upstream entry point while importing this project's slime first."""
import os
from pathlib import Path
import runpy
import sys

project = Path(__file__).resolve().parents[1]
upstream = Path(os.environ.get("SLIME_UPSTREAM_DIR", "../slime"))
if not upstream.is_absolute():
    upstream = project / upstream
upstream = upstream.resolve()
entry = upstream / sys.argv.pop(1)
sys.path.insert(0, str(upstream))
sys.path.insert(0, str(project))
import slime

if Path(slime.__file__).resolve().parent != project / "slime":
    raise RuntimeError("The project slime package must take precedence over upstream")
sys.argv[0] = str(entry)
runpy.run_path(str(entry), run_name="__main__")
