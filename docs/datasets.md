# Dataset download: PTB-XL

Use **PTB-XL 1.0.3** from the [official PhysioNet page](https://physionet.org/content/ptb-xl/1.0.3/).

1. Open the page and follow its access and citation instructions.
2. Under Files, download the dataset archive and extract it locally, or download the required folders individually.
3. Keep `ptbxl_database.csv`, `scp_statements.csv`, **both** `records100/` and `records500/`. CLEF and ECGFM-KED use native 100 Hz recordings; the other model recipes start from 500 Hz recordings.
4. Arrange the files as below, or point `dataset_root` to an existing download.

```text
data/PTB-XL/
  ptbxl_database.csv
  scp_statements.csv
  records100/
  records500/
```

Keep the WFDB `.hea` and `.dat` pairs together. Paths in `filename_lr` and `filename_hr` are relative to the dataset directory.

For the experiment, folds 1-8 are training, fold 9 is validation, and fold 10 is test. Preserve `ecg_id`, `patient_id`, and `strat_fold` locally for alignment and split checks. Record the dataset version in the run metadata. Dataset files stay ignored by Git.

Next: [model downloads](models.md) and [preprocessing](preprocessing/README.md).
