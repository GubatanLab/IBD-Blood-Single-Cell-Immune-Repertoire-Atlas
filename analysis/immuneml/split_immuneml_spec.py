from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def prune_definitions(definitions: dict, instructions: dict) -> dict:
    datasets = {value["dataset"] for value in instructions.values()}
    encodings = {
        setting["encoding"]
        for value in instructions.values()
        for setting in value.get("settings", [])
        if setting.get("encoding") is not None
    }
    ml_methods = {
        setting["ml_method"]
        for value in instructions.values()
        for setting in value.get("settings", [])
        if setting.get("ml_method") is not None
    }

    pruned = dict(definitions)
    pruned["datasets"] = {
        key: value
        for key, value in definitions.get("datasets", {}).items()
        if key in datasets
    }
    pruned["encodings"] = {
        key: value
        for key, value in definitions.get("encodings", {}).items()
        if key in encodings
    }
    pruned["ml_methods"] = {
        key: value
        for key, value in definitions.get("ml_methods", {}).items()
        if key in ml_methods
    }
    return pruned


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["instruction", "chain"], default="chain")
    args = parser.parse_args()

    spec_path = Path(args.spec)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with spec_path.open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    instructions = spec["instructions"]
    grouped: dict[str, dict] = {}

    if args.mode == "instruction":
        grouped = {name: {name: value} for name, value in instructions.items()}
    else:
        for name, value in instructions.items():
            if "_cd_vs_control_" in name:
                chain = name.split("_cd_vs_control_", 1)[0]
            elif "_uc_vs_control_" in name:
                chain = name.split("_uc_vs_control_", 1)[0]
            elif "_cd_vs_uc_" in name:
                chain = name.split("_cd_vs_uc_", 1)[0]
            else:
                chain = "unknown"
            grouped.setdefault(chain, {})[name] = value

    for group, group_instructions in grouped.items():
        subset = {
            "definitions": prune_definitions(spec["definitions"], group_instructions),
            "instructions": group_instructions,
        }
        out_path = output_dir / f"{group}.yaml"
        with out_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(subset, handle, sort_keys=False)
        print(out_path)


if __name__ == "__main__":
    main()
