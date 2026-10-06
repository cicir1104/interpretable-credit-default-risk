"""Run the complete reproducible analysis in the required order."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STEPS = [
    "download_data.py",
    "audit_data.py",
    "create_modelling_table.py",
    "train_evaluate.py",
]


def main() -> None:
    for script in STEPS:
        print(f"\n=== Running {script} ===", flush=True)
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / script)],
            cwd=PROJECT_ROOT,
            check=True,
        )
    print("\nComplete analysis finished successfully.")


if __name__ == "__main__":
    main()
