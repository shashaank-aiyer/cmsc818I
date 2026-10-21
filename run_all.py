#!/usr/bin/env python3
"""Run the tests, synthetic experiment, and results analysis.

Install dependencies first: python -m pip install -r requirements.txt
Run everything: python run_all.py
"""
from pathlib import Path
import subprocess
import sys


def main():
    root = Path(__file__).resolve().parent
    steps = [
        ["-m", "unittest", "-v", "test_experiment.py", "test_analysis.py"],
        ["experiment.py"],
        ["analyze_fixed.py"],
    ]
    for step in steps:
        subprocess.run([sys.executable, *step], cwd=root, check=True)


if __name__ == "__main__":
    main()
