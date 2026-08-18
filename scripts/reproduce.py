#!/usr/bin/env python3
"""Here we regenerate every result file that is distributed with the release."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
GENERATED = (
    "chunks.csv",
    "compare.json",
    "fit.json",
    "gamera.csv",
    "gamera.json",
    "literature.csv",
    "methods.csv",
    "mpmath.json",
    "pair_injection.csv",
    "pair_speed.csv",
    "pair_stress.csv",
    "pairs.csv",
    "pairs.json",
    "spectra.csv",
    "spectra.json",
    "speed.csv",
    "speed.json",
    "summary.txt",
    "universal.csv",
    "universal_celli.csv",
    "universal.json",
)
SCRIPTS = (
    "fit.py",
    "universal.py",
    "pairs.py",
    "spectra.py",
    "compare.py",
    "mpcheck.py",
    "speed.py",
)


def main() -> None:
    """Clear old outputs and run each reproduction step in a fresh process."""
    RESULTS.mkdir(exist_ok=True)
    for name in GENERATED:
        path = RESULTS / name
        if path.exists():
            path.unlink()

    for name in SCRIPTS:
        print(f"\n==> {name}", flush=True)
        subprocess.run(
            [sys.executable, str(ROOT / "scripts" / name)],
            cwd=ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
