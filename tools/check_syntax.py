#!/usr/bin/env python3
"""Parse Python and YAML files without importing or executing analyses."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    failures: list[str] = []
    python_files = list(ROOT.rglob("*.py"))
    yaml_files = list((ROOT / "immuneml").rglob("*.yml")) + list((ROOT / "immuneml").rglob("*.yaml"))

    for path in python_files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")

    for path in yaml_files:
        try:
            yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.relative_to(ROOT)}: {exc}")

    print(f"Python files parsed: {len(python_files)}")
    print(f"YAML files parsed: {len(yaml_files)}")
    if failures:
        print("Syntax checks failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("Syntax checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

