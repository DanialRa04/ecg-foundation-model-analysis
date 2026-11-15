# Downstream evaluation and analysis

Run these commands from the repository root. Install `requirements-evaluation.txt` and a maintained PyTorch build for the ANN probe. Model author implementations are only needed for extraction; these stages use saved embeddings.

## 1. Create shared labels and selections

```text
python -m scripts.prepare_manifest --paths configs/paths.local.json --output outputs/example/manifest
```

This reads PTB-XL metadata and `scp_statements.csv`, without reading waveforms. The experiment configuration determines tasks, folds, training fractions, and seeds. To change them, pass `--config path/to/experiment.json` here; the saved manifest carries those settings into evaluation and analysis.

Binary labels follow SCP code-key membership, including keys with zero confidence. MI and ST/T targets resolve through the supplied taxonomy. QRS-axis classes are retained using full-training support only; missing and unretained labels are blank in `targets.csv` and excluded from that task.

Whole training patients are assigned to balanced bins once. The configured fractions select nested sets of bins, shared by every model. Validation and test stay fixed. Patient ordering, tie resolution, class support, and balance loads are saved.

```text
manifest/
  records.csv             # recording, patient, fold, split, training bin
  targets.csv             # recording ID and encoded task targets
  patients.csv            # training-patient bin assignments
  tasks.json              # label meanings, retained classes, resolved codes
  support.csv             # counts and low-support flags by task/fraction/split
  balance.json
  config.resolved.json
  metadata.json
  run.log
```

## 2. Evaluate embeddings

```text
python -m scripts.evaluate_embeddings --manifest outputs/example/manifest --embeddings outputs/example/embeddings/MERL outputs/example/embeddings/ECGFounder --output outputs/example/evaluation
```

Each embedding directory must contain the complete manifest cohort from [extraction](running_pipeline.md), with its IDs, paired CSV, and completed metadata. A small `--limit` extraction cannot stand in for the complete cohort. ID order may differ: evaluation aligns recordings explicitly and verifies patient/split mapping and fingerprints.

For each model and training fraction, descriptors use all selected training embeddings before standardization. A feature scaler is fitted on that same training selection and reused for every task and held-out split. Task-specific missing labels are then excluded. Class weights use eligible training labels only.

Both configured probes run for each seed. The ANN selects its checkpoint using validation average precision; the linear probe uses fixed logistic-regression settings. Test outcomes never choose a checkpoint. The default probe device is CPU; `--device` and `--threads` change execution settings.

Outputs include `descriptors.csv`, per-seed `metrics.csv`, `class_metrics.csv`, `seed_summary.csv`, and scaler JSON files. `fits.csv` maps each model/task/fraction/probe/seed to a directory with `fit.json` and held-out `predictions.csv`. ANN history and class weights are recorded. Unsupported training classes produce a skipped fit with its reason. Undefined metrics remain blank; a seed summary requires all planned seeds to have valid values. Sample standard deviation is blank for a single planned seed.

## 3. Compare geometry and performance

```text
python -m scripts.analyze_results --evaluation outputs/example/evaluation --output outputs/example/analysis
```

The command verifies saved evaluation files and complete model/task joins. Main and additional correlations, grouped bootstrap intervals, group-aggregated results, leave-one-model-out estimates, documented-unexposed comparisons, and input-contract-adjusted Pearson associations are written to `associations.csv`.

Related checkpoints are resampled together. Bootstrap coefficients account for repeated rows and ties; group-aggregated sensitivity results give each group one row. Leave-one-model-out estimates have no bootstrap interval. Input adjustment uses log embedding width, leads, log sampling rate, and duration; constant controls are dropped, collinear or inestimable fits are flagged, and estimable bootstrap draws refit the controls.

P-values use standard two-sided correlation tests on the rows in that comparison, with Benjamini-Hochberg correction inside the configured families. These conventional p-values have small-sample and independence limitations; grouped intervals do not change their assumptions. Undefined comparisons retain a reason. Missing model outcomes are not silently removed from a task-wise comparison.

`overall_ranks.csv` averages within-task percentile ranks on the common complete model set. `duplicate_representations.csv` lists identical aligned float32 representations. Unknown dataset exposure stays unknown; the restricted comparison requires a documented unexposed status and reference.

Each stage needs a new output directory and saves its resolved configuration, hashes, metadata, and log. Keep all sample-level outputs local.

Next, [create result tables and figures](reporting.md) from the completed evaluation and analysis runs.

## Verification

```text
python -m checks.check_evaluation
```

This generates metadata and embedding fixtures, runs both probes, and checks analytical metrics, cohort isolation, grouped uncertainty, correction families, alignment, skipped fits, and tampering. See [the verification record](verification.md) for completed local checks.

Metric and test definitions follow [scikit-learn average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [one-vs-rest classification](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), and [SciPy correlation tests](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html). Experiment choices are specified in [the protocol](protocol.md).
