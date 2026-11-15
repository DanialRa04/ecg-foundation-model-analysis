# Preprocessing

We create a separate preprocessing recipe for each model from its authors' GitHub implementation, article, or released configuration. Each page below gives the ordered steps and links to that evidence. Choices made for this experiment, such as deterministic center crops, are identified separately.

## Shared workflow

1. Read the [dataset](../datasets.md) with WFDB physical-unit conversion and keep recording/patient identifiers for alignment.
2. Select the recipe's native source resolution (`records100` or `records500`).
3. Apply its lead order, resampling, filtering, normalization, and crop steps in the listed order. Do not apply one shared normalization to all models.
4. Fit any learned scaler on training data only and reuse it for validation/test. Save its parameters.
5. Validate finite values, sampling rate, shape, and sample order. Keep one consistent recording-ID sequence across model outputs.
6. Save transformed arrays outside Git, with a JSON record of the recipe, parameters, dataset version, upstream code revision, fitted scaler information, input/output shapes, and recording-order hash.

Canonical lead order is **I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6** unless the model page says otherwise. Filtering and resampling methods are part of the recipe; identical target rates do not make different algorithms interchangeable.

## Recipes and references

Shapes below are per recording, before model-specific batching or tokenization.

| Model | Rate (Hz) | Leads x samples | Main preparation |
| --- | --- | --- | --- |
| [MERL](merl.md) | 500 | 12 x 5000 | Lead reorder; record min-max |
| [ECGFounder](ecgfounder.md) | 500 | 12 x 5000 | PTB-XL evaluation z-score |
| [ECGFlow](ecgflow.md) | 500 | 8 x 5000 | Eight leads; released scalar scaling |
| [ECGFM-KED](ecgfm_ked.md) | 100 | 12 x 1000 | Native 100 Hz; training-fitted scalar scale |
| [CLEF](clef.md) | 500 | 1 x 5000 | Native 100 Hz lead II; scale; FFT to 500 Hz |
| [HeartLang](heartlang.md) | 100 | 12 x 1000 | FFT to 100 Hz; bandpass; QRS tokenization |
| [ECG-LFM](ecg_lfm.md) | 500 | 12 x 2500 | Center 5 s; preserve amplitudes |
| [DeepECG-SSL](deepecg_ssl.md) | 250 | 12 x 2500 | FFT to 250 Hz; preserve amplitudes |
| [ECG-JEPA](ecg_jepa.md) | 250 | 8 x 2500 | Eight leads; FFT to 250 Hz |
| [ST-MEM](st_mem.md) | 250 | 12 x 2250 | FFT; center 9 s; filters; z-score |
| [ECG-CPC](ecg_cpc.md) | 240 | 12 x 2400 | resampy to 240 Hz; no normalization |
| [ECG-FM](ecg_fm.md) | 500 | 12 x 2500 | Center 5 s; per-lead z-score |
| [HuBERT-ECG](hubert_ecg.md) | 500 | 12 x 2500 | Center 5 s; flatten in extractor |
| [S4-CPC](s4_cpc.md) | 240 | 12 x 600 | resampy; no normalization; center 2.5 s |
| [S4-Data2Vec](s4_data2vec.md) | 240 | 12 x 600 | resampy; released per-lead scale; center 2.5 s |
| [S4-DinoSR](s4_dinosr.md) | 240 | 12 x 600 | resampy; released per-lead scale; center 2.5 s |
| [S4-JEPA](s4_jepa.md) | 240 | 12 x 600 | resampy; released per-lead scale; center 2.5 s |
| [S4-HuBERT++](s4_hubertpp.md) | 240 | 12 x 600 | resampy; released per-lead scale; center 2.5 s |

## Code organization

General signal operations are in `common/signals.py`; PTB-XL reading is in `integrations/datasets/ptbxl.py`; each model's recipe and extraction interface are in `integrations/models/`. Configs select paths and settings. See [the execution guide](../running_pipeline.md) for preprocessing and extraction commands.

Each model page links the exact upstream files used for its recipe. In particular, the S4 per-lead constants are published in [clinical_ts/task/ecg.py](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/blob/main/downstream_evaluation/code/clinical_ts/task/ecg.py), and ECGFlow scaling is defined in [ecgflow/data.py](https://github.com/4dm-labs/ecgflow/blob/main/ecgflow/data.py).
