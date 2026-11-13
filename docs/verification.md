# Workflow verification

A clean source checkout passed **74 bounded check groups** on 2026-11-13.

| Stage | Check groups | Main coverage |
| --- | --- | --- |
| Configuration | 17 | Valid settings, coverage, invalid values, strict JSON |
| Descriptors | 6 | Analytical geometry, invariances, invalid inputs, kernel integrity |
| Preprocessing | 6 | All 18 input recipes, training scalers, paths, patient separation |
| Extraction | 9 | Frozen batching, IDs, hashes, trusted loading, tensor bounds, model Git revision |
| Evaluation | 13 | Labels, subsets, both probes, metrics, validation selection, statistics, skipped fits |
| Reporting | 12 | Precision, missing values, support flags, pagination, PNG/SVG/PDF, aggregate exports |
| Connected workflow | 7 | Generated ECGs through the linked production stages; input preservation |
| Publication boundary | 4 | Source/configuration/guides accepted; generated outputs and local assets excluded |

The connected check uses 24 generated ECGs at 100 and 500 Hz, three real preprocessing recipes, tiny fixture encoders, both probe types, analysis, and local reporting. Only pretrained encoder construction is replaced in this check. Inputs and outputs are temporary and remain under ignored directories.

## Model integration checks

On 2026-11-11, all 18 local pretrained integrations completed two-record preprocessing/extraction checks. Embedding shapes, finite values, recording IDs, paired CSVs, and output hashes were verified. ECG-JEPA ran on CPU; the other encoders ran on CUDA. ECG-LFM used float16 model parameters on CUDA and saved float32 embeddings.

CLEF and ECGFM-KED reused saved training scalers after verifying their cohort fingerprints. Fresh training-only fitting was checked with generated data, including extreme held-out values. A complete PTB-XL metadata manifest was also checked for 21,799 recordings, 18,869 patients, and all 13 configured tasks.

## Reports and environment

The generated-data reporting example produced six figure sets in PNG, SVG, and PDF. All six PDF pages were rendered and visually checked. The reporting environment's eleven runtime packages were audited with no known advisories reported in that snapshot.

The local reference environment used Python 3.11, NumPy 2.1.2, SciPy 1.16.2, WFDB 4.3.1, resampy 0.4.3, scikit-learn 1.7.1, Matplotlib 3.10.5, Pillow 13.0.0, and fontTools 4.60.2. Model integrations use their authors' additional dependencies and record them with each run.

The local model checks used PyTorch 2.5.1, which has reported security advisories. That version is not prescribed by extraction requirements. Use a maintained, author-compatible build and load pickle checkpoints only from trusted providers.

Follow [workflow checks](workflow_checks.md) to repeat the bounded verification. The repository shares code, configurations, and guides; all experiment results and generated figures stay local.
