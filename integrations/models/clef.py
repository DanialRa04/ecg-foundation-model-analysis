"""CLEF input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'CLEF',
 'source_hz': 100,
 'leads': ['II'],
 'hz': 500,
 'shape': [1, 5000],
 'operations': [{'operation': 'training_scalar'},
                {'operation': 'resample', 'hz': 500, 'method': 'fft'}],
 'guide': 'docs/preprocessing/clef.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import clef
    return clef(context)
