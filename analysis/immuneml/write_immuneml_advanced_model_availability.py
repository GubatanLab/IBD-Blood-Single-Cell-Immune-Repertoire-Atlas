from __future__ import annotations

from pathlib import Path

import pandas as pd


def main() -> None:
    output_dir = Path("chain_tcr_immuneml/advanced_immuneml_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [
        {
            "requested_model": "SVM",
            "immuneml_class": "immuneML.ml_methods.classifiers.SVM",
            "status": "completed",
            "reason": "immuneML SVM is a scikit-learn SVC wrapper; equivalent linear SVM models were run locally on the same immuneML-style normalized CDR3 k-mer repertoire features for all prior comparisons.",
            "feature_representation": "AA/NT CDR3 3-mer and 4-mer normalized k-mer frequencies",
            "dataset_compatibility": "RepertoireDataset/sample-level repertoire classification",
        },
        {
            "requested_model": "DeepRC",
            "immuneml_class": "immuneML.ml_methods.classifiers.DeepRC",
            "status": "completed_in_tables_and_heatmaps",
            "reason": "DeepRC 0.0.6 and widis-lstm-tools 0.4 were installed from the official GitHub ZIP archives into immuneml_py310_clean. The immuneML v3.0.27 wrapper was patched locally to pass n_sequences_per_bag to the installed DeepRC.forward API, read DeepRC metadata with the correct separator/IDs, and write HTML reports as UTF-8 on Windows. DeepRC was run on CPU with one update and one held-out assessment split for all prior comparisons and TCR chain groups.",
            "feature_representation": "DeepRC encoder / repertoire MIL sequence bags",
            "dataset_compatibility": "Binary RepertoireDataset/sample-level repertoire classification; CPU-only local run",
        },
        {
            "requested_model": "MIL",
            "immuneml_class": "DeepRC is the immuneML repertoire-level attention/MIL classifier available in this install",
            "status": "completed_via_deeprc",
            "reason": "The available immuneML MIL-style repertoire classifier for sample-level repertoire prediction is DeepRC. Completed MIL rows therefore correspond to the DeepRC attention/MIL classifier.",
            "feature_representation": "DeepRC repertoire sequence bags",
            "dataset_compatibility": "Binary RepertoireDataset/sample-level repertoire classification; CPU-only local run",
        },
        {
            "requested_model": "CNN",
            "immuneml_class": "KerasSequenceCNN / ReceptorCNN",
            "status": "blocked_not_run",
            "reason": "KerasSequenceCNN is for SequenceDataset classification, not sample-level RepertoireDataset prediction. ReceptorCNN is for paired ReceptorDataset classification, not sample-level repertoire labels. The current analyses are repertoire-level patient/sample predictions.",
            "feature_representation": "One-hot sequence/receptor encoding",
            "dataset_compatibility": "Not directly compatible with current repertoire-level AIRR inputs/comparisons",
        },
        {
            "requested_model": "neural immuneML",
            "immuneml_class": "DeepRC / KerasSequenceCNN / ReceptorCNN",
            "status": "partially_completed",
            "reason": "DeepRC was completed as the compatible repertoire-level neural/MIL model after local wrapper patches. The available CNN classes are intended for SequenceDataset/ReceptorDataset rather than the current sample-level RepertoireDataset comparisons.",
            "feature_representation": "DeepRC or one-hot sequence/receptor encodings",
            "dataset_compatibility": "DeepRC compatible with Binary RepertoireDataset; CNN classes not directly compatible with current repertoire-level AIRR inputs/comparisons",
        },
    ]

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / "advanced_immuneml_model_availability.csv", index=False)
    with (output_dir / "advanced_immuneml_model_availability.md").open("w", encoding="utf-8") as handle:
        handle.write("# Advanced immuneML Model Availability\n\n")
        handle.write(
            "This table records which requested immuneML model families were runnable "
            "for the current sample-level repertoire prediction analyses.\n\n"
        )
        columns = table.columns.tolist()
        handle.write("| " + " | ".join(columns) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(columns)) + " |\n")
        for row in table.itertuples(index=False):
            values = [str(value).replace("|", "/") for value in row]
            handle.write("| " + " | ".join(values) + " |\n")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
