"""Root app.py entrypoint for CardioTwin X."""
import os
import sys
from pathlib import Path

# Ensure root and cardiotwin directory are in sys.path
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

# Execute cardiotwin/app.py
target_app = root_dir / "cardiotwin" / "app.py"
with open(target_app, "rb") as f:
    code = compile(f.read(), str(target_app), "exec")
    exec(code, globals())
