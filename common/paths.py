"""Config-relative local paths; no asset download side effects."""
from pathlib import Path
from common.configuration import read_json

def load_paths(path):
    path = Path(path).resolve()
    config = read_json(path)
    paths = {name: (path.parent / value).resolve() for name, value in config['paths'].items()}
    overrides = {}
    for name, values in config.get('models', {}).items():
        overrides[name] = {}
        for key, value in values.items():
            if key == 'dependencies':
                overrides[name][key] = [(path.parent / item).resolve() for item in value]
            else:
                overrides[name][key] = (path.parent / value).resolve()
    return paths, overrides

def model_assets(model, paths, overrides):
    values = overrides.get(model['name'], {})
    checkpoint = values.get('checkpoint', paths['checkpoint_root'] / model['name'] / model['checkpoint_file'])
    source = values.get('source', paths['external_source_root'] / model['source_directory'])
    config = values.get('config')
    if config is None and model.get('model_config'):
        config = source / model['model_config']
    if config is None and model.get('checkpoint_config'):
        config = checkpoint.parent / model['checkpoint_config']
    dependencies = values.get('dependencies', [])
    return {'checkpoint': checkpoint.resolve(), 'source': source.resolve(), 'config': config, 'dependencies': dependencies}
