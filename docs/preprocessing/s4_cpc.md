# S4-CPC: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 600` (leads x samples), 240 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Resample 500 to 240 Hz using resampy.
4. Leave amplitudes unnormalized: the CPC configuration sets base.normalize=false.
5. Select the center 600 samples (2.5 seconds).
## Experiment choice

Use a deterministic center crop instead of training-time random windows.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [checkpoints](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints)
- [ecg.py](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/blob/main/downstream_evaluation/code/clinical_ts/task/ecg.py)
- [main_lite_base.py](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/blob/main/downstream_evaluation/code/main_lite_base.py)
- [config_cpc_ecg_s4.yaml](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/blob/main/checkpoints/config_cpc_ecg_s4.yaml)

[All recipes](README.md) | [Download this model](../models.md)
