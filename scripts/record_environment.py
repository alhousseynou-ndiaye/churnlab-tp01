
import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

record = {
    "python": sys.version,
    "platform": platform.platform(),
    "uv": subprocess.check_output(["uv", "--version"], text=True).strip(),
    "packages": {
        name: version(name)
        for name in ["churnlab", "pandas", "numpy", "scikit-learn"]
    },
}

out = Path("evidence/tp01/environment.json")
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(record, indent=2), encoding="utf-8")

print(json.dumps(record, indent=2))
