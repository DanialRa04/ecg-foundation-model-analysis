# ECG-LFM: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2500` (leads x samples), 500 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Keep the center 5 seconds (2500 samples).
4. Preserve amplitudes without per-record normalization.
## Experiment choice

Use a deterministic center crop instead of a random training crop, giving one representation per recording.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [preprocess_mimic_iv_ecg.py](https://github.com/biomed-AI/ECG-LFM/blob/master/fairseq_signals/data/ecg/preprocess/preprocess_mimic_iv_ecg.py)
- [ECG-LFM](https://github.com/biomed-AI/ECG-LFM)

[All recipes](README.md) | [Download this model](../models.md)
