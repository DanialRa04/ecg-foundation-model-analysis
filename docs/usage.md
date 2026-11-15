# How to use the repository

## 1. Set up Python

From the repository root, create and activate an environment, then install the descriptor dependencies:

```text
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

```text
python -m pip install -r requirements-core.txt
```

## 2. Run the checks

```text
python -m scripts.validate_config --output outputs/checks/config.json
python -m checks.check_configuration
python -m checks.check_descriptor_math --output outputs/checks/descriptor_math.json
```

The first command checks settings and reports the planned workload without reading datasets or loading models. The other commands exercise configuration-integrity and synthetic geometry checks. Choose new output paths for another run; existing files are preserved.

## 3. Analyze your own embeddings

The descriptor function accepts a finite two-dimensional NumPy array: rows are recordings, columns are features.

```python
import numpy as np
from common.descriptor_math import spectral_and_collapse_descriptors

embeddings = np.load("private/embeddings.npy", allow_pickle=False)
descriptors = spectral_and_collapse_descriptors(
    embeddings, variance_tolerance=1e-8, rankme_epsilon=1e-7
)
print(descriptors)
```

Run this from the repository root. Constant embeddings are rejected; some diagnostics are undefined for very low rank. When exporting JSON, represent undefined values as `null`.

## 4. Prepare the ECG experiment

Follow the [dataset download](datasets.md), [model downloads](models.md), and [preprocessing guide](preprocessing/README.md). Copy `configs/paths.example.json` to `configs/paths.local.json` and set your paths. Relative paths resolve from the config directory.

The [preprocessing and extraction guide](running_pipeline.md) gives the stage commands, per-model path settings, and output layout. Settings are specified in the [protocol](protocol.md) and linked JSON configs. Validate another configuration with `python -m scripts.validate_config --config path/to/experiment.json`.

## 5. Evaluate and analyze

The [evaluation guide](evaluation.md) explains shared task manifests, patient subsets, probe fitting, and statistical comparisons, with commands and output layouts. `python -m checks.check_evaluation` exercises these stages with generated data first.

## 6. Create tables and figures

Follow [the reporting guide](reporting.md) to render completed results as aggregate CSV/Markdown tables and PNG/SVG/PDF figures. Report settings control the comparison, labels, styling, and pagination.

## Outputs

Keep each run under `outputs/<run-name>/`, with its config, JSON metadata, CSV tables, plots, and logs. `results/` and `figures/` are optional local storage directories whose contents are also ignored. All experiment outputs stay outside GitHub. See the [folder guide](folders.md).
