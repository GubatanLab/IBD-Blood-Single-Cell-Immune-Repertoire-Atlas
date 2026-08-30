# Analysis provenance

The code is organized by role rather than by original storage location.

- `core/` contains statistical and biological analysis workflows.
- `figures/` contains panel assembly and publication rendering.
- `external_validation/` contains public-cohort validation.
- `immuneml/` contains repertoire preparation, immuneML execution, and model interpretation.
- `ml_validation/` contains nested-validation and sensitivity figure workflows.

Some scripts are historical execution records rather than a unified software package. They may assume intermediate objects created by an earlier script. Use `workflow_manifest.tsv` and `docs/figure-analysis-map.md` to identify the intended order and the controlled inputs that are not distributed.

Known absolute roots have been neutralized. Use `tools/relocate_paths.py` to insert authorized local paths in a fresh clone. Do not commit the resulting private `config/paths.json`.

