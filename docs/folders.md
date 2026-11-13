# Folder guide

| Folder | What belongs here |
| --- | --- |
| `common/` | General descriptors, statistics, evaluation, I/O, and plotting functions |
| `integrations/datasets/` | Dataset readers, labels, patient mapping, and splits |
| `integrations/models/` | Model-specific loading, input contracts, and extraction |
| `scripts/` | Thin entry points that run each experiment stage |
| `configs/` | Experiment parameters and path/model/dataset settings |
| `checks/` | Focused numerical and data-integrity checks |
| `docs/` | Usage, downloads, experiment description, and preprocessing references |
| `data/` | User-downloaded datasets; only instructions are tracked |
| `checkpoints/` | User-downloaded weights; only instructions are tracked |
| `external/` | User-installed upstream code; only instructions are tracked |
| `outputs/` | Generated per-run CSVs, JSON, plots, and logs; ignored |
| `results/` | Optional local result tables; contents ignored |
| `figures/` | Optional local figure exports; contents ignored |
| `private/` | Local working material; ignored |

`common/` calculates, `integrations/` handles model/dataset differences, and `scripts/` connects them. Inputs stay separate from outputs. Use explicit paths and save the configuration with each run.
