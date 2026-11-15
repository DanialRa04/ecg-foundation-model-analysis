"""S4-CPC input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'S4-CPC',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 240,
 'shape': [12, 600],
 'operations': [{'operation': 'resample', 'hz': 240, 'method': 'resampy'},
                {'operation': 'center_crop', 'samples': 600}],
 'guide': 'docs/preprocessing/s4_cpc.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import s4
    return s4(context, variant='cpc')
