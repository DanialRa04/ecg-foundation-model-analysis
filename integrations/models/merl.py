"""MERL input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'MERL',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVF', 'aVL', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 500,
 'shape': [12, 5000],
 'operations': [{'operation': 'minmax'}],
 'guide': 'docs/preprocessing/merl.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import merl
    return merl(context)
