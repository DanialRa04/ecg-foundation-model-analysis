# ECG Foundation Model Analysis

I started this experiment with a question: **does the geometry of an ECG foundation model's embeddings tell us anything about its downstream performance?**

This repository shares the code and guides for exploring that question with different ECG models and datasets. Experiment results and generated figures stay local.

The Python workflow connects ECG preprocessing, frozen embedding extraction, representation geometry, downstream evaluation, and statistical analysis. Shared calculations are separated from the model integrations and the included PTB-XL reader.

## Check the code

Use Python 3.11 or later and run commands from the repository root:

```text
python -m pip install -r requirements-core.txt
python -m scripts.validate_config
python -m checks.check_configuration
python -m checks.check_descriptor_math
```

These commands need no datasets or model weights. The [workflow checks](docs/workflow_checks.md) cover the connected stages with generated inputs; their guide lists the additional dependencies.

## Start here

1. [Set up and run the available code](docs/usage.md).
2. [Download the dataset](docs/datasets.md).
3. [Download the model checkpoints](docs/models.md).
4. [Follow the preprocessing overview and model-specific steps](docs/preprocessing/README.md).
5. [Run preprocessing and embedding extraction](docs/running_pipeline.md).
6. [Prepare labels, evaluate embeddings, and analyze results](docs/evaluation.md).
7. [Create result tables and figures](docs/reporting.md).

See [the experiment](docs/experiment.md) for the question, [the protocol](docs/protocol.md) for settings and methods, and [the folder guide](docs/folders.md) for where everything belongs.

## Included workflow

The code includes patient manifests and nested training subsets, two probe types, training-only scaling, representation descriptors, grouped uncertainty estimates, multiple-testing correction, and local table/figure generation. Verification uses analytical cases, generated-data workflows, and small local model checks.

Integrations cover MERL, ECGFounder, ECGFlow, ECGFM-KED, CLEF, HeartLang, ECG-LFM, DeepECG-SSL, ECG-JEPA, ST-MEM, ECG-CPC, ECG-FM, HuBERT-ECG, and five S4 checkpoints. See [model downloads](docs/models.md) for the exact variants and author implementations.

Follow [workflow verification](docs/workflow_checks.md) to check the stages and their connections before using your own inputs.

See [the verification record](docs/verification.md) for completed checks and their scope.

Shared methods live in `common/`; dataset and model details live in `integrations/`. GitHub contains source code, configurations, and guides. Data, weights, downloaded model code, results, and generated figures stay outside Git.

## Citation and license

Citation metadata is in [CITATION.cff](CITATION.cff). For reproducibility, record the Git commit you use. Dataset, model, method, and software references are collected in [references](docs/references.md).

Repository code and documentation use the [MIT License](LICENSE). Downloaded datasets, model weights, and author implementations retain their providers' terms; see [external resources](THIRD_PARTY_NOTICES.md).
