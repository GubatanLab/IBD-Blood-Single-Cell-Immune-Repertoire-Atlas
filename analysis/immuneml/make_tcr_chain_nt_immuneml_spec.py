from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-spec",
        default="chain_tcr_immuneml/tcr_chain_diagnosis_k3_k4_aa.yaml",
    )
    parser.add_argument(
        "--output-spec",
        default="chain_tcr_immuneml/tcr_chain_diagnosis_k3_k4_nt.yaml",
    )
    args = parser.parse_args()

    with Path(args.input_spec).open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)

    spec["definitions"]["encodings"] = {
        "k3_nt": {
            "KmerFrequency": {
                **spec["definitions"]["encodings"]["k3_aa"]["KmerFrequency"],
                "name": "k3_nt",
                "sequence_type": "NUCLEOTIDE",
            }
        },
        "k4_nt": {
            "KmerFrequency": {
                **spec["definitions"]["encodings"]["k4_aa"]["KmerFrequency"],
                "name": "k4_nt",
                "sequence_type": "NUCLEOTIDE",
            }
        },
    }

    nt_instructions = {}
    for old_name, instruction in spec["instructions"].items():
        new_name = old_name.removesuffix("_aa") + "_nt"
        updated = dict(instruction)
        updated["sequence_type"] = "NUCLEOTIDE"
        updated["settings"] = [
            {
                **setting,
                "encoding": setting["encoding"].removesuffix("_aa") + "_nt",
            }
            for setting in instruction["settings"]
        ]
        nt_instructions[new_name] = updated
    spec["instructions"] = nt_instructions

    output_path = Path(args.output_spec)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(spec, handle, sort_keys=False)
    print(output_path)


if __name__ == "__main__":
    main()
