from __future__ import annotations

import argparse
import runpy
import sys

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("result")
    parser.add_argument("--logging", default="INFO")
    args = parser.parse_args()

    if not hasattr(np, "float_"):
        np.float_ = np.float64

    sys.argv = [
        "ImmuneMLApp.py",
        "--logging",
        args.logging,
        args.spec,
        args.result,
    ]
    runpy.run_module("immuneML.app.ImmuneMLApp", run_name="__main__")


if __name__ == "__main__":
    main()
