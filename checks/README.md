# Checks

From the repository root:

```text
python -m checks.check_configuration
python -m checks.check_descriptor_math --output outputs/checks/descriptor_math.json
```

The configuration checks use the standard library and cover split overlap, duplicate settings, coverage, invalid parameters, and different experiment sizes. The descriptor checks require NumPy and cover isotropic/rank-one geometry, selected invariances, centered versus uncentered behavior, invalid inputs, and kernel integrity.

`python -m checks.check_preprocessing` uses synthetic WFDB recordings to check the input pipeline. `python -m checks.check_extraction` adds a small deterministic PyTorch encoder for batching, fingerprints, and aligned outputs. See [the pipeline guide](../docs/running_pipeline.md).

`python -m checks.check_evaluation` generates a metadata/embedding cohort and exercises task labels, patient subsets, both probes, analytical metrics, grouped bootstrap coefficients, correction families, complete joins, and protected outputs. It needs `requirements-evaluation.txt` plus PyTorch; no dataset or model downloads are needed.

`python -m checks.check_reporting` generates aggregate tables and checks report selection, display precision, missing values, support flags, pagination, linked fingerprints, and PNG/SVG/PDF exports. It needs `requirements-reporting.txt`.

`python -m checks.check_workflow` connects generated WFDB signals, real preprocessing recipes, fixture encoders, saved embeddings, probes, analysis, and reports in one bounded run. `python -m checks.check_publication` checks the source-and-documentation publication boundary. See [workflow verification](../docs/workflow_checks.md).
