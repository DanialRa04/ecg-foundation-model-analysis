# CLEF: preprocessing

**Input:** PTB-XL records100. **Output:** `1 x 5000` (leads x samples), 500 Hz.

## Steps

1. Read native records100 and select lead II (tar_lead=1 in the upstream dataset).
2. Fit one scalar StandardScaler on training-fold lead-II signals.
3. Apply that scaler to all splits.
4. Resample 100 to 500 Hz using the upstream WFDB/SciPy FFT semantics; retain 10 seconds.
## Experiment choice

Fit the scaler on folds 1-8 only.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [lightning_ptbxl.py](https://github.com/Nokia-Bell-Labs/ecg-foundation-model/blob/main/clef/lightning/lightning_ptbxl.py)
- [ptb_utils.py](https://github.com/Nokia-Bell-Labs/ecg-foundation-model/blob/main/clef/utils/ptb_utils.py)
- [downstream_dataloader.py](https://github.com/Nokia-Bell-Labs/ecg-foundation-model/blob/main/clef/data/downstream_dataloader.py)

[All recipes](README.md) | [Download this model](../models.md)
