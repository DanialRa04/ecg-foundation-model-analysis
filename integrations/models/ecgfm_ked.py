"""ECGFM-KED input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'ECGFM-KED',
 'source_hz': 100,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 100,
 'shape': [12, 1000],
 'operations': [{'operation': 'training_scalar'}],
 'guide': 'docs/preprocessing/ecgfm_ked.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import ecgfm_ked
    return ecgfm_ked(context)
