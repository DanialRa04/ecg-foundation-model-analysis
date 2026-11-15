# HuBERT-ECG: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2500` (leads x samples), 500 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Select the center 5 seconds.
4. Preserve amplitudes without an added z-score.
5. Flatten the 12 lead segments in the extraction stage, following the model interface.
## Experiment choice

Use a deterministic center crop. The current upstream dataset also exposes first-window and random-crop paths; record the selected view explicitly.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [dataset.py](https://github.com/Edoar-do/HuBERT-ECG/blob/master/hubert_ecg/dataset.py)
- [hubert-ecg-base](https://huggingface.co/Edoardo-Coppola/hubert-ecg-base)

[All recipes](README.md) | [Download this model](../models.md)
