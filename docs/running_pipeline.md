# Preprocessing and embedding extraction

## Setup

From the repository root, install the preprocessing dependencies:

```text
python -m pip install -r requirements-preprocessing.txt
```

For extraction, use the model author's environment with a maintained, compatible PyTorch build. `requirements-extraction.txt` includes the shared numerical dependency; install PyTorch and the model's other dependencies according to its environment instructions. Obtain [checkpoints and author implementations](models.md) separately. Copy `configs/paths.example.json` to `configs/paths.local.json` and set the local locations. Relative paths resolve from that configuration's directory.

Default source directories use the repository names in `configs/models.json`; default weights live under `checkpoints/<model-name>/`. Override any model explicitly when your layout differs:

```json
{
  "paths": {
    "dataset_root": "../data/PTB-XL",
    "checkpoint_root": "../checkpoints",
    "external_source_root": "../external"
  },
  "models": {
    "MERL": {
      "source": "../external/MERL-ICML2024",
      "checkpoint": "../checkpoints/MERL/vit_tiny_best_encoder.pth"
    }
  }
}
```

`dependencies` can list additional local import roots. ECGFlow uses its authors' timm fork; ECG-FM and DeepECG-SSL use the appropriate fairseq-signals implementation. ECG-LFM uses the implementation in its own repository. Follow each author's environment instructions when these dependencies differ.

## 1. Preprocess a small selection

```text
python -m scripts.preprocess --paths configs/paths.local.json --models MERL --limit 8 --output outputs/example/preprocessing
```

Use `--models all` or multiple model names for several recipes. Omit `--limit` for the complete cohort. `--ids path/to/ids.txt` selects recording IDs, one per line, in that exact order.

The reader checks patient separation, WFDB pairs, physical units, leads, and native rates. Each [model recipe](preprocessing/README.md) applies its ordered operations and validates the output shape.

CLEF and ECGFM-KED need training-fitted signal scalers. Even a small selected output fits those scalers on the complete training cohort. Save time on subsequent runs with `--scalers path/to/scalers.json`; reuse is accepted only when its training-cohort, metadata, lead, and source-rate fingerprints match. All required local training recordings must be readable when fitting afresh.

## 2. Extract embeddings

Run each model in a separate process/environment:

```text
python -m scripts.extract_embeddings --paths configs/paths.local.json --model MERL --input outputs/example/preprocessing/MERL --output outputs/example/embeddings/MERL --batch-size 4 --trust-checkpoint
```

Pickle-based author weights require `--trust-checkpoint`; use files obtained from a trusted provider. Safetensors integrations do not require that flag. Nothing is downloaded automatically.

The encoder runs frozen in evaluation/inference mode. Fixed-length batches produce one final embedding per recording. Set `--device cpu` or `--device cuda`, and reduce `--batch-size` if needed. `--threads` controls CPU threads (default 1); `--seed` overrides the experiment's extraction seed (default 42). Seeds and cuDNN settings are saved in metadata. Run large models one at a time; they still need sufficient device memory. The fairseq-based integrations stream model tensors without materializing optimizer state.

Use `--checkpoint`, `--source`, and `--model-config` for one-run overrides. ECG-CPC needs `config_last_11597276_ckpt.yaml` beside its weights, or an explicit `--model-config` path. S4 variants use their matching released YAML under the authors' `checkpoints/` directory. The integration registry covers the 18 model IDs in `configs/models.json`.

## Output layout

```text
preprocessing/
  config.resolved.json
  metadata.json
  scalers.json                # when fitted scalers are needed
  run.log
  MERL/
    signals.npy              # float32 [recordings, leads, samples]
    sample_ids.npy
    records.csv              # local recording/patient/split mapping
    recipe.json
    metadata.json

embeddings/MERL/
  embeddings.npy             # float32 [recordings, features]
  sample_ids.npy
  records.csv
  config.resolved.json
  metadata.json
  run.log
```

Each command requires a new output directory. Metadata records selection, shapes, recipe/settings, hashes, dependencies, checkpoint identity, pooling, and completion status. Extraction verifies input fingerprints and ID alignment. Failed/incomplete outputs stay marked and are not accepted as complete inputs. Do not edit paired files independently.

An upstream commit is recorded when the configured model source directory is its own Git repository root. For unpacked source archives, the revision is `null`; checkpoint and integration hashes are still recorded.

Dataset arrays, checkpoints, and sample-level outputs stay local. The default `outputs/` tree is ignored by Git.

## Checks

```text
python -m checks.check_preprocessing
python -m checks.check_extraction
```

These use generated WFDB fixtures and a small deterministic encoder. They exercise every input recipe, training-only scalers, path and split checks, batching, alignment, fingerprint enforcement, and output protection.

See [the verification record](verification.md) for the completed local checks.
