# ECG-CPC: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2400` (leads x samples), 240 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Resample 500 to 240 Hz with resampy anti-aliasing, retaining 10 seconds.
4. Do not normalize amplitudes; the selected checkpoint uses normalize=false.
## Basis

We built this recipe from the authors' implementation and released input settings:

- [README.md](https://github.com/AI4HealthUOL/ecg-fm-benchmarking/blob/main/README.md)
- [ecg-fm-benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking)

[All recipes](README.md) | [Download this model](../models.md)
