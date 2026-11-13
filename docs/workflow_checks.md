# Verify the workflow

Repository verification uses small, bounded checks. Experiment results and generated figures stay local; a complete research run is not required before publishing the code.

## Check each stage

From the repository root, in an environment with the relevant requirements installed:

```text
python -m checks.check_configuration
python -m checks.check_descriptor_math
python -m checks.check_preprocessing
python -m checks.check_extraction
python -m checks.check_evaluation
python -m checks.check_reporting
```

These cover numerical definitions, model input recipes, training-only fitting, patient separation, alignment, frozen extraction, probes, metrics, statistics, report exports, and failure handling. The [checks guide](../checks/README.md) lists the dependencies for each command.

## Check the connections

```text
python -m checks.check_workflow
```

This creates 24 synthetic ECGs at 100 and 500 Hz and passes them through the production stages in order:

1. Read metadata, create targets, and assign patient subsets.
2. Apply three real model preprocessing recipes and fit scalers on training records.
3. Extract embeddings through the frozen extraction runner using tiny generated encoders.
4. Fit both configured probe types on the saved, aligned embeddings.
5. Calculate descriptors and model-level associations.
6. Create a linked local report and verify its artifacts.

Only the pretrained encoder construction is replaced by a fixture in this connected check. It requires no dataset downloads, author implementations, checkpoints, or GPU. Temporary inputs and outputs are created under ignored `outputs/` and removed afterward. The separate small real-model checks described in [the verification record](verification.md) cover the author integrations.

The connected check needs preprocessing, evaluation, and reporting dependencies plus PyTorch. Use the requirement files for those stages; choose a maintained CPU PyTorch build if no GPU is needed.

```text
python -m pip install -r requirements-preprocessing.txt -r requirements-evaluation.txt -r requirements-reporting.txt
```

Install PyTorch separately in that environment. The recorded dependency versions are in [verification](verification.md).

## Check public contents

```text
python -m checks.check_publication
```

This reviews Git-visible files, including tracked files that could bypass ignore rules. It accepts source code, configurations, guides, and folder instructions, and rejects generated results or local assets. `outputs/`, `results/`, `figures/`, data, weights, external model code, and local verification artifacts remain outside Git; folder instruction files may be tracked.

Finish with a review of download instructions, citations, authorship, and license. The [usage guide](usage.md) explains how users can run their own experiments locally.
