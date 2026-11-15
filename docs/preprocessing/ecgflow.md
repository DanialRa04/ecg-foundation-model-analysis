# ECGFlow: preprocessing

**Input:** PTB-XL records500. **Output:** `8 x 5000` (leads x samples), 500 Hz.

## Steps

1. Read records500.
2. Select I, II, V1, V2, V3, V4, V5, V6.
3. Keep all 5000 samples.
4. Use the released PTB-XL eight-channel scaling: (x - (-0.0012090115)) / 0.26843855.
## Basis

We built this recipe from the authors' implementation and released input settings:

- [data.py](https://github.com/4dm-labs/ecgflow/blob/main/ecgflow/data.py)

[All recipes](README.md) | [Download this model](../models.md)
