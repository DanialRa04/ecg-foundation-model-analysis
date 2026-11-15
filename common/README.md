# General methods

Reusable calculations live here. `descriptor_math.py` computes geometry descriptors from a NumPy embedding matrix without loading datasets or models.

`configuration.py` loads linked JSON settings relative to the experiment config and checks consistency. It supports configurable model/task lists, patient-bin mappings, probe settings, and statistical parameters.

`signals.py` provides reusable signal operations and streaming scalar fitting. `artifacts.py` handles hashes, IDs, metadata, and separate run directories. `paths.py` resolves local config paths. `checkpoints.py` streams selected tensors from explicitly trusted author checkpoints.

`cohorts.py` builds whole-patient subsets. `experiment_data.py` verifies and aligns saved stages; `tables.py` reads and writes CSVs. `metrics.py` and `probes.py` implement ranking metrics and fixed classifiers. `evaluation.py` connects those calculations, while `associations.py` and `analysis.py` provide grouped uncertainty, correction, and result comparisons.

`plotting.py` provides reusable headless figures. `report_tables.py` handles report selection and display formatting; `reporting.py` verifies aggregate inputs and connects table/figure exports. See [reporting](../docs/reporting.md).

See [usage](../docs/usage.md) for an example and [checks](../checks/README.md) for analytical cases. The calculations form feature-by-feature Gram matrices, so memory grows with embedding width. Undefined diagnostics need explicit missing values when exported.
