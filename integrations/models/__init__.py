"""Explicit model integration registry."""
from importlib import import_module

REGISTRY = {'MERL': 'merl',
 'ECGFounder': 'ecgfounder',
 'ECGFlow': 'ecgflow',
 'ECGFM-KED': 'ecgfm_ked',
 'CLEF': 'clef',
 'HeartLang': 'heartlang',
 'ECG-LFM': 'ecg_lfm',
 'DeepECG-SSL': 'deepecg_ssl',
 'ECG-JEPA': 'ecg_jepa',
 'ST-MEM': 'st_mem',
 'ECG-CPC': 'ecg_cpc',
 'ECG-FM': 'ecg_fm',
 'HuBERT-ECG': 'hubert_ecg',
 'S4-CPC': 's4_cpc',
 'S4-Data2Vec': 's4_data2vec',
 'S4-DinoSR': 's4_dinosr',
 'S4-JEPA': 's4_jepa',
 'S4-HuBERT++': 's4_hubertpp'}

def get_integration(name):
    if name not in REGISTRY:
        raise ValueError(f"Unknown model integration: {name}")
    return import_module(f"{__name__}.{REGISTRY[name]}")
