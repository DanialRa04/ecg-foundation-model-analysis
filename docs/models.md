# Model downloads

The experiment includes the 18 checkpoints below. You can start with one model and add others later.

1. Open the provider link and download the listed pretrained variant. On Hugging Face, use **Files and versions**; for Zenodo/Figshare, select the listed file or extract it from the archive; for Google Drive, use Download.
2. Put each model in `checkpoints/<model-name>/`, using the filenames below. For a full Hugging Face model directory, keep its config and weight filenames together.
3. If the model needs its authors' code, obtain that repository separately under `external/` and follow its environment instructions. Record the upstream revision and checkpoint hash with your run.

| Model | Provider | Local file(s) | Implementation |
| --- | --- | --- | --- |
| MERL | [Download](https://drive.google.com/drive/folders/13wb4DppUciMn-Y_qC2JRWTbZdz3xX0w2) | `vit_tiny_best_encoder.pth` | [Code](https://github.com/cheliu-computation/MERL-ICML2024) |
| ECGFounder | [Download](https://huggingface.co/PKUDigitalHealth/ECGFounder/tree/main) | `12_lead_ECGFounder.pth` | [Code](https://github.com/PKUDigitalHealth/ECGFounder) |
| ECGFlow | [Download](https://huggingface.co/4dm-labs/1dvit_base_patch50_5000.ssl_mimic/tree/main) | `model.safetensors` | [Code](https://github.com/4dm-labs/ecgflow) |
| ECGFM-KED | [Download](https://zenodo.org/records/14881564) | `best_valid_all_increase_with_augment_epoch_3.pt` | [Code](https://github.com/control-spiderman/ECGFM-KED) |
| CLEF | [Download](https://zenodo.org/records/17572734) | `clef_medium.ckpt` | [Code](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) |
| HeartLang | [Download](https://huggingface.co/PKUDigitalHealth/HeartLang/tree/main) | `checkpoint-200.pth` | [Code](https://github.com/PKUDigitalHealth/HeartLang) |
| ECG-LFM | [Download](https://zenodo.org/records/20388950) | `pretrain.pt` | [Code](https://github.com/biomed-AI/ECG-LFM) |
| DeepECG-SSL | [Download](https://huggingface.co/heartwise/SSL_Pretrained_model/tree/main) | `SSL_pretrained.pt` | [Code](https://github.com/HeartWise-AI/DeepECG-SSL-finetune) |
| ECG-JEPA | [Download](https://drive.google.com/file/d/1gMOT4xjQQg0GZkY1iE6NuDzua4ALw00l/view) | `multiblock_epoch100.pth` | [Code](https://github.com/sehunfromdaegu/ECG_JEPA) |
| ST-MEM | [Download](https://drive.google.com/file/d/1E7J-A1HqWa2f08T6Sfk5uWk-_CFJhOYQ/view) | `st_mem_vit_base_encoder.pth` | [Code](https://github.com/vuno/ST-MEM) |
| ECG-CPC | [Download](https://figshare.com/articles/dataset/ECG-CPC_Checkpoint_zip/30192604?file=58173919) | `last_11597276.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-fm-benchmarking) |
| ECG-FM | [Download](https://huggingface.co/wanglab/ecg-fm/tree/main) | `mimic_iv_ecg_physionet_pretrained.pt` | [Code](https://github.com/bowang-lab/ECG-FM) |
| HuBERT-ECG | [Download](https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/tree/main) | `config.json + model weights` | [Code](https://github.com/Edoar-do/HuBERT-ECG) |
| S4-CPC | [Download](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints) | `cpc_e10.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |
| S4-Data2Vec | [Download](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints) | `data2vec_e10.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |
| S4-DinoSR | [Download](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints) | `dinosr_e10.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |
| S4-JEPA | [Download](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints) | `jepa_e10.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |
| S4-HuBERT++ | [Download](https://github.com/AI4HealthUOL/ecg-pretraining-strategies/tree/main/checkpoints) | `hubert++_e10.ckpt` | [Code](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |

## Variant notes

- **MERL:** Select the pretrained ViT-Tiny encoder and use the filename in the table.
- **ECGFlow:** Use the SSL-pretrained model, not a task-fine-tuned variant; also keep its config files.
- **CLEF:** Download CLEF-Medium.
- **HeartLang:** Choose the MIMIC-IV pretrained checkpoint under checkpoints/pretrain/MIMIC-IV, rather than a fine-tuned checkpoint.
- **ECG-JEPA:** Download the multi-block masking checkpoint and use the filename in the table.
- **ST-MEM:** Download the encoder checkpoint and use the filename in the table.
- **ECG-CPC:** Extract the archive and keep `last_11597276.ckpt` and `config_last_11597276_ckpt.yaml` together in the model directory.
- **ECG-FM:** Download the MIMIC-IV + PhysioNet pretrained weights and their corresponding model configuration.
- **HuBERT-ECG:** Download the entire pretrained base model directory, including config.json and its weights; keep their filenames unchanged.
- **S4-CPC:** Download the named e10 checkpoint and its matching released configuration.
- **S4-Data2Vec:** Download the named e10 checkpoint and its matching released configuration.
- **S4-DinoSR:** Download the named e10 checkpoint and its matching released configuration.
- **S4-JEPA:** Download the named e10 checkpoint and its matching released configuration.
- **S4-HuBERT++:** Download the named e10 checkpoint and its matching released configuration.

The S4 integrations read these YAML files from the author's repository, under `checkpoints/`:

| Model | Matching configuration |
| --- | --- |
| S4-CPC | `config_cpc_ecg_s4.yaml` |
| S4-Data2Vec | `config_data2vec_ecg_s4.yaml` |
| S4-DinoSR | `config_dinosr_ecg_s4.yaml` |
| S4-JEPA | `config_jepa_ecg_s4.yaml` |
| S4-HuBERT++ | `config_skkmeans_ecg_s4.yaml` |

Use `--model-config` or a model's `config` path override for a different local layout. See [the pipeline guide](running_pipeline.md) for environments and commands.

Example layout:

```text
checkpoints/
  ECGFounder/12_lead_ECGFounder.pth
  ECGFlow/model.safetensors
  HuBERT-ECG/config.json
  HuBERT-ECG/<weight-files>
```

Keep downloaded files local and follow each provider's access and citation instructions. Next: [preprocessing](preprocessing/README.md).
