# DeepECG-SSL: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2500` (leads x samples), 250 Hz.

## Steps

1. Read records500 in physical mV.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. FFT-resample 500 to 250 Hz, retaining all 10 seconds.
4. Preserve physical amplitudes; do not apply a per-record z-score.
## Basis

We built this recipe from the authors' implementation and released input settings:

- [README.md](https://github.com/HeartWise-AI/DeepECG-SSL-finetune/blob/main/README.md)
- [preprocess_parquet.py](https://github.com/HeartWise-AI/DeepECG-SSL-finetune/blob/main/fairseq_signals/data/ecg/preprocess/preprocess_parquet.py)

[All recipes](README.md) | [Download this model](../models.md)
