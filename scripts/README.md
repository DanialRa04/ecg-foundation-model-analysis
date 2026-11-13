# Stage scripts

Shared methods belong in `common/`; model and dataset details belong in `integrations/`.

Available now: `python -m scripts.validate_config`, which checks the experiment settings and planned workload without loading assets.

`python -m scripts.preprocess` prepares model inputs. `python -m scripts.extract_embeddings` extracts one model's final embeddings. See [the pipeline guide](../docs/running_pipeline.md) for arguments and examples.

Available commands are listed in [usage](../docs/usage.md).

`python -m scripts.prepare_manifest` saves shared targets and patient subsets. `python -m scripts.evaluate_embeddings` runs descriptors and probes on aligned complete cohorts. `python -m scripts.analyze_results` joins results and computes associations. See [evaluation and analysis](../docs/evaluation.md).

`python -m scripts.render_report` creates aggregate tables and scientific figures from linked completed runs. See [reporting](../docs/reporting.md).
