# ECG-JEPA: preprocessing

**Input:** PTB-XL records500. **Output:** `8 x 2500` (leads x samples), 250 Hz.

## Steps

1. Read records500 in physical mV.
2. Select I, II, V1, V2, V3, V4, V5, V6.
3. FFT-resample 500 to 250 Hz, retaining 10 seconds.
4. Preserve amplitudes without additional normalization.
## Basis

We built this recipe from the authors' implementation and released input settings:

- [dataset.py](https://github.com/sehunfromdaegu/ECG_JEPA/blob/master/util/dataset.py)
- [transforms.py](https://github.com/sehunfromdaegu/ECG_JEPA/blob/master/util/transforms.py)

[All recipes](README.md) | [Download this model](../models.md)
