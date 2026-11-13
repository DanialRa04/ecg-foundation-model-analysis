# Dataset integration

Dataset readers, labels, units, patient mapping, and split rules live here. Download instructions are in [datasets.md](../../docs/datasets.md).

`ptbxl.py` reads local WFDB records in physical mV, validates metadata and patient partitions, and selects native 100/500 Hz inputs with explicit lead order.

`ptbxl_targets.py` resolves configured tasks from SCP code keys and taxonomy mappings, retaining QRS-axis classes using full-training support. Shared subset calculations live in `common/cohorts.py`.
