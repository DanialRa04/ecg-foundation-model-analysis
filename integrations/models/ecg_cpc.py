"""ECG-CPC input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ECG-CPC',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 240,
 'shape': [12, 2400],
 'operations': [{'operation': 'resample', 'hz': 240, 'method': 'resampy'}],
 'guide': 'docs/preprocessing/ecg_cpc.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import ecg_cpc
    return ecg_cpc(context)
