# ECGFounder: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 5000` (leads x samples), 500 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Keep the full 10 seconds at 500 Hz.
4. Apply a global per-record z-score over leads and time, with epsilon 1e-8.
## Experiment choice

This recipe follows the PTB-XL evaluation path in ptbxl_eval.py. The repository's general dataset.py path describes additional filtering; keep those paths distinct.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [ptbxl_eval.py](https://github.com/PKUDigitalHealth/ECGFounder/blob/master/ptbxl_eval.py)
- [README.md](https://github.com/PKUDigitalHealth/ECGFounder/blob/master/README.md)

[All recipes](README.md) | [Download this model](../models.md)
