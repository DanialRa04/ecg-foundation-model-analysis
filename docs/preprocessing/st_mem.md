# ST-MEM: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2250` (leads x samples), 250 Hz.

## Steps

1. Read records500.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. FFT-resample 500 to 250 Hz.
4. Select the center 2250 samples (9 seconds).
5. Apply fifth-order zero-phase 0.67 Hz high-pass, then 40 Hz low-pass filtering.
6. Apply one global per-record z-score across leads and time.
## Experiment choice

Use only the center member of the upstream three-crop evaluation so each recording has one representation.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [dataset.py](https://github.com/vuno/ST-MEM/blob/main/util/dataset.py)
- [transforms.py](https://github.com/vuno/ST-MEM/blob/main/util/transforms.py)

[All recipes](README.md) | [Download this model](../models.md)
