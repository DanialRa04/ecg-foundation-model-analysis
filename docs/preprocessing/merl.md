# MERL: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 5000` (leads x samples), 500 Hz.

## Steps

1. Read records500 in physical mV.
2. Order leads I, II, III, aVR, aVF, aVL, V1-V6 (aVF precedes aVL).
3. Keep the full 10 seconds.
4. Apply one min-max scaling to [0, 1] over all leads and samples of each recording.
## Basis

We built this recipe from the authors' implementation and released input settings:

- [finetune_dataset.py](https://github.com/cheliu-computation/MERL-ICML2024/blob/main/finetune/finetune_dataset.py)

[All recipes](README.md) | [Download this model](../models.md)
