"""ECG-JEPA input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ECG-JEPA',
 'source_hz': 500,
 'leads': ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 250,
 'shape': [8, 2500],
 'operations': [{'operation': 'resample', 'hz': 250, 'method': 'fft'}],
 'guide': 'docs/preprocessing/ecg_jepa.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import ecg_jepa
    return ecg_jepa(context)
