# ECG-FM: preprocessing

**Input:** PTB-XL records500. **Output:** `12 x 2500` (leads x samples), 500 Hz.

## Steps

1. Read records500.
2. Use lead order I, II, III, aVR, aVL, aVF, V1-V6.
3. Use the authors' linear-resampling semantics if the input rate differs from 500 Hz.
4. Select the center 5 seconds.
5. Apply per-lead z-score standardization; map constant leads to zero.
## Experiment choice

Use one deterministic center segment rather than multiple evaluation views.

## Basis

We built this recipe from the authors' implementation and released input settings:

- [README.md](https://github.com/bowang-lab/ecg-fm/blob/main/README.md)
- [fairseq-signals](https://github.com/Jwoo5/fairseq-signals)

[All recipes](README.md) | [Download this model](../models.md)
