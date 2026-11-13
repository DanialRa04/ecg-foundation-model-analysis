"""HeartLang input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'HeartLang',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 100,
 'shape': [12, 1000],
 'operations': [{'operation': 'resample', 'hz': 100, 'method': 'fft'},
                {'operation': 'bandpass', 'low': 0.67, 'high': 40, 'order': 5, 'method': 'ba'}],
 'guide': 'docs/preprocessing/heartlang.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import heartlang
    return heartlang(context)
