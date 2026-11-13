"""DeepECG-SSL input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'DeepECG-SSL',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 250,
 'shape': [12, 2500],
 'operations': [{'operation': 'resample', 'hz': 250, 'method': 'fft'}],
 'guide': 'docs/preprocessing/deepecg_ssl.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import deepecg_ssl
    return deepecg_ssl(context)
