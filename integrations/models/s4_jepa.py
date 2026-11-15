"""S4-JEPA input recipe and final-embedding integration."""
from common.signals import execute_recipe

RECIPE = {'model': 'S4-JEPA',
 'source_hz': 500,
 'leads': ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'],
 'hz': 240,
 'shape': [12, 600],
 'operations': [{'operation': 'resample', 'hz': 240, 'method': 'resampy'},
                {'operation': 'fixed_scale',
                 'mean': [-0.00184586,
                          -0.00130277,
                          0.00017031,
                          -0.00091313,
                          -0.00148835,
                          -0.00174687,
                          -0.00077071,
                          -0.00207407,
                          0.00054329,
                          0.00155546,
                          -0.00114379,
                          -0.00035649],
                 'scale': [0.16401004,
                           0.1647168,
                           0.23374124,
                           0.33767231,
                           0.33362807,
                           0.30583013,
                           0.2731171,
                           0.27554379,
                           0.17128962,
                           0.14030828,
                           0.14606956,
                           0.14656108]},
                {'operation': 'center_crop', 'samples': 600}],
 'guide': 'docs/preprocessing/s4_jepa.md'}

def preprocess(values, source_hz, scaler=None):
    return execute_recipe(values, source_hz, RECIPE, scaler)

def build_extractor(context):
    from integrations.models.builders import s4
    return s4(context, variant='jepa')
