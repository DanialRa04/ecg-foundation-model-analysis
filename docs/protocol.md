# Experiment protocol

## Data and representations

Use PTB-XL 1.0.3, with folds 1-8 for training, 9 for validation, and 10 for test. Verify unique recording IDs and patient separation. Download instructions and per-model input recipes are in [datasets](datasets.md) and [preprocessing](preprocessing/README.md).

Keep each encoder frozen and in evaluation mode. Extract its documented final representation and record pooling, input shape, lead order, sample rate, duration, checkpoint hash, and upstream revision. Align saved embeddings by recording ID rather than array position alone.

## Training subsets

Use the full training set and a nested approximately 50% patient subset, with seed 42. Validation and test stay fixed. Assign whole patients to two balanced bins and use bin 1 for the half subset; recording counts need not be exactly halved.

Balance patient-level recording counts, task-positive counts, sex counts, and age-band counts (0-40, 40-60, 60-75, 75+). Order patients by rarity-weighted label load, then recording count and seeded tie-breaks. Greedily minimize the increment in squared bin loads, normalized by target loads, plus the configured patient-count penalty; resolve ties by current patient count and seeded bin order. Save the bin assignment once and reuse it for every model. Only training data contribute to this construction.

## Tasks

Definitions are in [tasks.json](../configs/tasks.json). Binary targets use code-key membership in `scp_codes`; no confidence threshold is applied. MI uses the `MI` diagnostic-class mapping in `scp_statements.csv`; ST/T abnormality is the union of the `STTC` mapping and its explicit form codes.

QRS axis uses nonmissing `heart_axis` classes with at least 50 recordings in the full training cohort. Fix that class set before fitting fraction-specific probes. Exclude missing/unretained labels consistently across models, record support, and skip a fit if its training subset cannot learn the retained classes.

## Descriptors

Compute the [configured panel](../configs/descriptors.json) on every embedding in the matching training fraction, before probe feature standardization. The primary comparison uses 100%; the half-fraction comparison uses its own descriptors. [Definitions](descriptor_definitions.csv) specify centering and formulas.

RankMe uses uncentered singular values; centered spectral descriptors use squared singular values. Variance calculations use population moments. Zero-energy centered matrices fail validation, and undefined diagnostics remain missing. Morphology-neighborhood descriptors are a separate additional experiment.

## Probes and metrics

Fit a feature StandardScaler on the current training subset and reuse it unchanged for validation/test. Derive class weights from that subset. [probes.json](../configs/probes.json) contains the ANN and logistic-regression settings.

The ANN has 256/128 hidden units with LayerNorm, GELU, and dropout 0.2; AdamW uses learning rate 0.0003, weight decay 0.0001, batch size 256, and at most 100 epochs. Select the best validation average precision, with patience 10 and minimum improvement 1e-8. Binary loss weights positives by training negatives/positives; multiclass weights are inverse class frequencies, normalized to mean one.

Run seeds 42 and 43 and retain per-seed results plus their mean and sample standard deviation. The linear comparison uses L2 logistic regression, C=1, balanced class weights, liblinear, and one-vs-rest for multiclass tasks. It has no hyperparameter search.

Here **AUPRC means average precision**, not trapezoidal PR integration. AUROC is secondary. Multiclass metrics are unweighted one-vs-rest means over classes with both outcomes in the evaluated split; save the included classes and per-class values. Single-class ranking metrics are missing and flagged. Test results never select hyperparameters, thresholds, or epochs. Optional binary threshold metrics use 0.5.

## Statistical analysis

Join descriptors and seed-aggregated performance on model, layer, and training fraction; task and probe identify the outcome. Reject duplicate or incomplete joins. The main association is task-wise Spearman between the full-training descriptor panel and ANN test average precision. Pearson, Kendall, linear probes, and half-training results are additional views.

Use 2,000 model-group bootstrap resamples for percentile 95% intervals, with seed 42 and deterministic iteration order. Related S4 checkpoints share one resampling group; other configured groups remain explicit. Keep all rows belonging to a sampled group together. Record independent group counts and valid resample counts. Seeds, tasks, and fractions do not increase the independent model count.

Standard two-sided correlation-test p-values operate on checkpoint rows; grouping applies to the bootstrap intervals, not automatically to those p-values. Include a group-aggregated sensitivity analysis. Correct finite p-values with Benjamini-Hochberg within each section x cohort x probe x metric x method, covering all task-descriptor tests in that family. Task-agnostic targets use their own family. Do not select a top descriptor panel from the same outcomes before testing.

For an overall summary, rank models within each task using average ties, convert to percentiles on the common complete model set, then average across tasks. Do not directly average AUPRC across tasks with different prevalence.

Record dataset exposure as yes, no, or unknown, with a reference. The restricted exposure comparison includes documented unexposed checkpoints; keep unknowns separate. Also examine leave-one-model-out influence, related-group aggregation, exact duplicate representations, and differences in dimensions/leads/rates/duration where the available model count permits adjustment. Skip an inestimable comparison with its reason.

## Saved outputs

Each run gets a new directory with resolved config, hashes, seeds, dependency versions, CSV tables, JSON metadata, figures, and logs. Keep sample-level data local. Export undefined numerical values as JSON null, preserve precision in analysis CSVs, and round display tables separately.

The configuration check can run without downloading assets. Stage-specific data and checkpoint checks run when those stages are implemented.
