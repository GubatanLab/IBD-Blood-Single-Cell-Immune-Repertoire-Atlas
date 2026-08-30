#!/usr/bin/env python3
"""Replace neutral path placeholders with authorized local roots.

This utility only edits text files inside the cloned repository. Use --apply to
write changes; without it, the command reports the files that would change.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


TEXT_SUFFIXES = {".py", ".r", ".R", ".yml", ".yaml", ".json", ".md", ".txt", ".tsv"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--apply", action="store_true", help="write replacements in place")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    replacements = config.get("replacements", {})
    if not replacements:
        raise SystemExit("No replacements were defined in the configuration file.")

    changed: list[Path] = []
    for path in args.repo.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if ".git" in path.parts or path.resolve() == args.config.resolve():
            continue
        try:
            original = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        revised = original
        for placeholder, local_root in replacements.items():
            revised = revised.replace(placeholder, str(local_root).replace("\\", "/"))
        if revised != original:
            changed.append(path)
            if args.apply:
                path.write_text(revised, encoding="utf-8", newline="\n")

    action = "updated" if args.apply else "would update"
    print(f"{action} {len(changed)} files")
    for path in changed:
        print(path.relative_to(args.repo))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

