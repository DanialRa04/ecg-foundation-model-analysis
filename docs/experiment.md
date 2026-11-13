# The experiment

**Question:** How does label-free embedding geometry relate to downstream performance across ECG foundation models?

The experiment uses frozen final representations on PTB-XL. It compares spectral diversity, concentration, and collapse descriptors with performance on ECG classification tasks.

- Main comparison: full-training descriptors versus fixed-ANN test AUPRC, using Spearman correlation.
- Additional views: AUROC, linear probes, and matching 50% training subsets.
- Sensitivity checks: pretraining overlap, related model families, duplicate representations, and influential models.
- Additional experiments: label efficiency, morphology-neighborhood descriptors, and spectral ablation.

Scalers and probe preprocessing are fitted on training data. Patient splits stay separate, descriptor and probe cohorts match, and related checkpoints are accounted for in statistical uncertainty. Effect sizes, confidence intervals, and multiple-testing correction are reported together.

The preprocessing guides explain the exact input used for each model, including deterministic choices made for this experiment. See [models](models.md), [dataset](datasets.md), and [preprocessing](preprocessing/README.md).

The [experiment protocol](protocol.md) specifies splits, task definitions, probe settings, descriptor conventions, statistics, and saved outputs. Validate the configuration before execution:

```text
python -m scripts.validate_config
```
