"""ECG-LFM input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ECG-LFM',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 500,
 'shape': [12, 2500],
 'operations': [{'operation': 'center_crop', 'samples': 2500}],
 'guide': 'docs/preprocessing/ecg_lfm.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import ecg_lfm
    return ecg_lfm(context)
