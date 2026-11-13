"""ECGFlow input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ECGFlow',
 'source_hz': 500,
 'leads': ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 500,
 'shape': [8, 5000],
 'operations': [{'operation': 'fixed_scale', 'mean': -0.0012090115, 'scale': 0.26843855}],
 'guide': 'docs/preprocessing/ecgflow.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import ecgflow
    return ecgflow(context)
