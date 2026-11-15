"""ST-MEM input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ST-MEM',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 250,
 'shape': [12, 2250],
 'operations': [{'operation': 'resample', 'hz': 250, 'method': 'fft'},
                {'operation': 'center_crop', 'samples': 2250},
                {'operation': 'bandpass',
                 'low': 0.67,
                 'high': 40,
                 'order': 5,
                 'method': 'sequential_sos'},
                {'operation': 'zscore'}],
 'guide': 'docs/preprocessing/st_mem.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import st_mem
    return st_mem(context)
