# References and citation

## This repository

Use [CITATION.cff](../CITATION.cff) for the repository citation. Credit the code to [DanialRa04](https://github.com/DanialRa04), and record the Git commit used in your experiment. Cite the dataset, models, and methods you use as well.

## Dataset

- **PTB-XL 1.0.3:** Wagner et al. (2022), PhysioNet dataset release. [DOI: 10.13026/kfzx-aw45](https://doi.org/10.13026/kfzx-aw45).
- **PTB-XL paper:** Wagner et al. (2020), *PTB-XL, a large publicly available electrocardiography dataset*. Scientific Data. [DOI: 10.1038/s41597-020-0495-6](https://doi.org/10.1038/s41597-020-0495-6).
- **PhysioNet:** Pollard et al. (2026), *PhysioNet as a global platform for biomedical research*. Nature Health. [DOI: 10.1038/s44360-026-00096-z](https://doi.org/10.1038/s44360-026-00096-z).

These follow the citation instructions on the [dataset provider page](https://physionet.org/content/ptb-xl/1.0.3/).

## Models and released implementations

Follow the authors' recommended citation on the linked project page, alongside the referenced paper where available. The checkpoint variants and downloads are specified in [models](models.md); the exact input-processing sources are linked from each [preprocessing recipe](preprocessing/README.md).

| Integration | Author project | Paper or citation source |
| --- | --- | --- |
| MERL | [MERL-ICML2024](https://github.com/cheliu-computation/MERL-ICML2024) | [Zero-Shot ECG Classification with Multimodal Learning and Test-time Clinical Knowledge Enhancement](https://arxiv.org/abs/2403.06659) |
| ECGFounder | [ECGFounder](https://github.com/PKUDigitalHealth/ECGFounder) | [Authors' paper and NEJM AI citation](https://github.com/PKUDigitalHealth/ECGFounder) |
| ECGFlow | [ECGFlow](https://github.com/4dm-labs/ecgflow) | [A foundation transformer model with self-supervised learning for ECG-based assessment of cardiac and coronary function](https://www.medrxiv.org/content/10.1101/2023.10.25.23297552v3) |
| ECGFM-KED | [ECGFM-KED](https://github.com/control-spiderman/ECGFM-KED) | [Foundation model of ECG diagnosis: Diagnostics and explanations of any form and rhythm on ECG](https://doi.org/10.1016/j.xcrm.2024.101875) |
| CLEF | [CLEF](https://github.com/Nokia-Bell-Labs/ecg-foundation-model) | [CLEF: Clinically-Guided Contrastive Learning for Electrocardiogram Foundation Models](https://arxiv.org/abs/2512.02180) |
| HeartLang | [HeartLang](https://github.com/PKUDigitalHealth/HeartLang) | [Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model](https://openreview.net/forum?id=6Hz1Ko087B) |
| ECG-LFM | [ECG-LFM](https://github.com/biomed-AI/ECG-LFM) | [Authors' reference and citation](https://github.com/biomed-AI/ECG-LFM) |
| DeepECG-SSL | [DeepECG-SSL-finetune](https://github.com/HeartWise-AI/DeepECG-SSL-finetune) | [Foundation models for electrocardiogram interpretation: clinical implications](https://doi.org/10.1093/eurheartj/ehaf1119) |
| ECG-JEPA | [ECG-JEPA](https://github.com/sehunfromdaegu/ECG_JEPA) | [Learning General Representation of 12-Lead Electrocardiogram with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2410.08559) |
| ST-MEM | [ST-MEM](https://github.com/vuno/ST-MEM) | [Guiding Masked Representation Learning to Capture Spatio-Temporal Relationship of Electrocardiogram](https://openreview.net/forum?id=WcOohbsF4H) |
| ECG-CPC | [ECG-FM benchmarking](https://github.com/AI4HealthUOL/ecg-fm-benchmarking) | [Benchmarking paper and released-checkpoint citation](https://openreview.net/forum?id=xXRqWpt3Xr) |
| ECG-FM | [ECG-FM](https://github.com/bowang-lab/ECG-FM) | [ECG-FM paper](https://arxiv.org/abs/2408.05178) |
| HuBERT-ECG | [HuBERT-ECG](https://github.com/Edoar-do/HuBERT-ECG) | [HuBERT-ECG as a self-supervised foundation model for broad and scalable cardiac applications](https://www.medrxiv.org/content/10.1101/2024.11.14.24317328) |
| S4-CPC, S4-Data2Vec, S4-DinoSR, S4-JEPA, S4-HuBERT++ | [ECG pretraining strategies](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) | [Authors' paper and project references](https://github.com/AI4HealthUOL/ecg-pretraining-strategies) |

## Geometry, evaluation, and statistics

- Garrido et al. (2023), *RankMe: Assessing the Downstream Performance of Pretrained Self-Supervised Representations by Their Rank*. ICML. [Paper](https://proceedings.mlr.press/v202/garrido23a.html).
- Roy and Vetterli (2007), *The Effective Rank: A Measure of Effective Dimensionality*. [Author-hosted paper](https://infoscience.epfl.ch/bitstreams/9a5f3153-5c5d-4845-ab41-962296ec93ac/download).
- Benjamini and Hochberg (1995), *Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing*. [DOI: 10.1111/j.2517-6161.1995.tb02031.x](https://doi.org/10.1111/j.2517-6161.1995.tb02031.x).

Our centered spectral descriptors use squared singular values; RankMe uses uncentered singular values. Exact formulas and conventions are in [descriptor definitions](descriptor_definitions.csv), and model-group bootstrap, test units, and correction families are specified in [the protocol](protocol.md).

Metric and estimator definitions follow the official documentation for [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html), [AUROC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html), [one-vs-rest classification](https://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html), and [SciPy correlations](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html).

## Software

| Dependency | Citation or project reference |
| --- | --- |
| NumPy | Harris et al. (2020), *Array programming with NumPy*. [Citation instructions](https://numpy.org/citing-numpy/) |
| SciPy | Virtanen et al. (2020), *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. [Citation instructions](https://scipy.org/citing-scipy/) |
| scikit-learn | Pedregosa et al. (2011), *Scikit-learn: Machine Learning in Python*. [JMLR](https://jmlr.org/papers/v12/pedregosa11a.html) |
| PyTorch | [PyTorch project and citation](https://github.com/pytorch/pytorch#citation) |
| Matplotlib | Hunter (2007), *Matplotlib: A 2D Graphics Environment*. [Citation instructions](https://matplotlib.org/stable/project/citing.html) |
| WFDB | [WFDB Python project](https://github.com/MIT-LCP/wfdb-python) |
| resampy | [resampy project](https://github.com/bmcfee/resampy) |

Use the authors' citation instructions for additional model-environment dependencies. Provider licenses and access terms are described in [external resources](../THIRD_PARTY_NOTICES.md).
