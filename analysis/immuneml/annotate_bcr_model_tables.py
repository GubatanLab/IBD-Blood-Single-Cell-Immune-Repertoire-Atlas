from __future__ import annotations

from pathlib import Path

import pandas as pd


ML_MODEL = "LogisticRegression"
ML_MODEL_DETAILS = "DictVectorizer + StandardScaler(with_mean=False) + LogisticRegression(solver=liblinear,max_iter=2000)"
EVALUATION_DESIGN = "5 stratified random 70/30 train/test splits; aggregate tables report mean/std/count across splits"

CHAIN_LABELS = {
    "bcr_light": "BCR light chain",
    "bcr_heavy": "BCR heavy chain",
    "bcr_heavy_light": "BCR paired heavy+light chain",
    "bcr_isotype": "BCR isotype proportions",
    "bcr_shm": "BCR somatic hypermutation rates",
}

FEATURE_LABELS = {
    "aa_k3": "amino-acid CDR3 3-mer frequencies",
    "aa_k4": "amino-acid CDR3 4-mer frequencies",
    "nt_k3": "nucleotide CDR3 3-mer frequencies",
    "nt_k4": "nucleotide CDR3 4-mer frequencies",
    "repertoire_metrics": "repertoire diversity/clonality metrics",
    "aa_k3_plus_repertoire_metrics": "amino-acid CDR3 3-mer frequencies + repertoire diversity/clonality metrics",
    "aa_k4_plus_repertoire_metrics": "amino-acid CDR3 4-mer frequencies + repertoire diversity/clonality metrics",
    "nt_k3_plus_repertoire_metrics": "nucleotide CDR3 3-mer frequencies + repertoire diversity/clonality metrics",
    "nt_k4_plus_repertoire_metrics": "nucleotide CDR3 4-mer frequencies + repertoire diversity/clonality metrics",
    "isotype_proportions": "IgA/IgG/IgM/IgD proportions",
    "shm_rates": "unweighted and duplicate-count-weighted SHM rates",
}

TABLES = (
    "chain_bcr_immuneml/bcr_chain_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_chain_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_diagnosis_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_diagnosis_svm_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_diagnosis_lr_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_diagnosis_lr_svm_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_inflammation_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_inflammation_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_inflammation_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_inflammation_svm_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_inflammation_lr_svm_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_therapy_response_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_therapy_response_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_therapy_response_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_therapy_response_svm_feature_model_summary_aggregate.csv",
    "chain_bcr_immuneml/bcr_therapy_response_lr_svm_feature_model_summary.csv",
    "chain_bcr_immuneml/bcr_therapy_response_lr_svm_feature_model_summary_aggregate.csv",
)


def model_label(row: pd.Series) -> str:
    chain = CHAIN_LABELS.get(row["chain_group"], row["chain_group"])
    feature = FEATURE_LABELS.get(row["feature_set"], row["feature_set"])
    return f"{chain}: {feature}"


def ordered_columns(df: pd.DataFrame) -> list[str]:
    preferred = [
        "chain_group",
        "comparison",
        "feature_set",
        "sequence_type",
        "model_label",
        "ml_model",
        "ml_model_details",
        "evaluation_design",
        "n_samples",
        "positive_label",
    ]
    return [column for column in preferred if column in df.columns] + [
        column for column in df.columns if column not in preferred
    ]


def annotate_table(path: Path) -> None:
    df = pd.read_csv(path)
    df["model_label"] = df.apply(model_label, axis=1)
    if "ml_model" not in df.columns:
        df["ml_model"] = ML_MODEL
    else:
        df["ml_model"] = df["ml_model"].fillna(ML_MODEL)
    if "ml_model_details" not in df.columns:
        df["ml_model_details"] = ML_MODEL_DETAILS
    else:
        df["ml_model_details"] = df["ml_model_details"].fillna(ML_MODEL_DETAILS)
    if "evaluation_design" not in df.columns:
        df["evaluation_design"] = EVALUATION_DESIGN
    else:
        df["evaluation_design"] = df["evaluation_design"].fillna(EVALUATION_DESIGN)
    df = df[ordered_columns(df)]
    df.to_csv(path, index=False)
    print(f"Annotated {path}")


def main() -> None:
    for table in TABLES:
        path = Path(table)
        if path.exists():
            annotate_table(path)
        else:
            print(f"Skipped missing table {path}")


if __name__ == "__main__":
    main()
