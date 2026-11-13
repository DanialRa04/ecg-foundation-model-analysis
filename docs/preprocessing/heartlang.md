# HeartLang: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 1000` (leads x samples), 100 Hz.

## Steps

1. Read records500.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. FFT-resample 500 to 100 Hz.
4. Apply a fifth-order Butterworth 0.67-40 Hz bandpass using b/a coefficients and filtfilt.
5. Keep physical amplitudes without an additional z-score.
6. The extraction stage then creates QRS tokens using the authors' tokenizer.
## Experiment choice

Use the amplitude-preserving `is_norm=False` branch of the authors' preprocessing.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [mimic_preprocess.py](https://github.com/PKUDigitalHealth/HeartLang/blob/main/datasets/dataset_preprocess/MIMIC-IV/mimic_preprocess.py)
- [QRSTokenizer.py](https://github.com/PKUDigitalHealth/HeartLang/blob/main/QRSTokenizer.py)

[All recipes](README.md) | [Download this model](../models.md)
