# TCR Model Key for Tables and Figures

Implementation note: these analyses used the immuneML-prepared AIRR repertoire files and immuneML-style repertoire feature representations, then fit the completed classifiers with local runner scripts in this workspace. SVM metrics are included in the current tables and heatmaps. DeepRC/MIL metrics are also included after patching the immuneML v3.0.27 DeepRC wrapper for the installed DeepRC API and Windows UTF-8 report output. DeepRC was run as a CPU one-update immuneML assessment with one held-out split for each prior comparison and chain group. CNN/neural immuneML models other than DeepRC were not included because the available CNN classes are not compatible with these sample-level RepertoireDataset comparisons.

The tables and heatmaps use model labels that combine the feature representation and the classifier:

| Figure/table label | Feature representation | Classifier |
|---|---|---|
| `LR AA <chain> K3` | Amino-acid CDR3 3-mer normalized frequencies | Logistic regression |
| `LR AA <chain> K4` | Amino-acid CDR3 4-mer normalized frequencies | Logistic regression |
| `LR NT <chain> K3` | CDR3 nucleotide 3-mer normalized frequencies | Logistic regression |
| `LR NT <chain> K4` | CDR3 nucleotide 4-mer normalized frequencies | Logistic regression |
| `SVM AA <chain> K3` | Amino-acid CDR3 3-mer normalized frequencies | Linear SVM |
| `SVM AA <chain> K4` | Amino-acid CDR3 4-mer normalized frequencies | Linear SVM |
| `SVM NT <chain> K3` | CDR3 nucleotide 3-mer normalized frequencies | Linear SVM |
| `SVM NT <chain> K4` | CDR3 nucleotide 4-mer normalized frequencies | Linear SVM |
| `DeepRC <chain>` | DeepRC amino-acid CDR3 repertoire sequence bags | DeepRC attention/MIL classifier |
| `DIV <chain> LR` | Repertoire diversity/clonality summary features | Logistic regression |
| `DIV <chain> RF` | Repertoire diversity/clonality summary features | Random forest |

Chains are shown as `alpha`, `beta`, `alpha+beta`, `gamma`, `delta`, or `gamma+delta`.

For all CDR3 k-mer models, the feature matrix was sample-level normalized k-mer frequencies. Completed k-mer classifiers include logistic regression and linear SVM. For DeepRC/MIL models, immuneML's DeepRC encoder was used on amino-acid CDR3 repertoire sequence bags. For diversity/clonality models, the feature set included clonotype count, total clone count, Shannon entropy, Pielou evenness, clonality, Simpson metrics, Gini index, top clone frequencies, and singleton fraction, with either logistic regression or random forest as the classifier.

The current result tables include explicit columns:

| Column | Meaning |
|---|---|
| `model_label` | Short label used in heatmaps and summary figures |
| `feature_representation` | Input feature representation used by the classifier |
| `classifier` | Classifier name |
| `model_used` | Full feature-plus-classifier description |
