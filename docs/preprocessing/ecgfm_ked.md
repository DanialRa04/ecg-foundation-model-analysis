# ECGFM-KED: preprocessing

**Input:** PTB-XL records100. **Output:** `12 x 1000` (leads x samples), 100 Hz.

## Steps

1. Read native records100 rather than downsampling records500.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Fit a single scalar StandardScaler over training-fold signals only.
4. Apply the saved mean and scale unchanged to validation and test recordings.
## Experiment choice

The scaler uses folds 1-8 only; save its fitted statistics with the run.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [data_preprocess.py](https://github.com/control-spiderman/ECGFM-KED/blob/main/dataset/ptb-xl/data_preprocess.py)

[All recipes](README.md) | [Download this model](../models.md)
