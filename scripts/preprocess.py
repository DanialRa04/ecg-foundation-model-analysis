"""Prepare selected PTB-XL recordings for one or more model input recipes."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common.artifacts import file_hash, log_event, new_run, save_records, sequence_hash, versions, write_json
from common.configuration import load_experiment, read_json
from common.paths import load_paths
from common.signals import fit_scalar
from integrations.datasets import ptbxl
from integrations.models import get_integration

def run(args):
    bundle = load_experiment(args.config)
    config = bundle['experiment']
    if config['dataset']['name'] != 'PTB-XL':
        raise ValueError('This entry point uses the PTB-XL reader; general signal methods accept other datasets')
    paths, _ = load_paths(args.paths)
    dataset = paths['dataset_root']
    rows = ptbxl.metadata(dataset, config['dataset']['splits'])
    selected = rows
    if args.ids:
        wanted = [line.strip() for line in args.ids.read_text().splitlines() if line.strip()]
        if not wanted or len(set(wanted)) != len(wanted):
            raise ValueError('Requested IDs must be nonempty and unique')
        indexed = {row['sample_id']: row for row in rows}
        if any(value not in indexed for value in wanted):
            raise ValueError('Some requested recording IDs are absent')
        selected = [indexed[value] for value in wanted]
    if args.limit is not None:
        if args.limit < 1: raise ValueError('Limit must be positive')
        selected = selected[:args.limit]
    catalog = {model['name']: model for model in bundle['models']['models']}
    names = config['model_ids'] if args.models == ['all'] else args.models
    if not names or len(set(names)) != len(names) or any(name not in config['model_ids'] for name in names):
        raise ValueError('Choose unique models from the experiment configuration')
    modules = {name: get_integration(name) for name in names}
    for name, module in modules.items():
        if module.RECIPE['shape'] != catalog[name]['shape'] or module.RECIPE['hz'] != catalog[name]['sampling_hz']:
            raise ValueError('Catalog and input recipe disagree')
    scalers = read_json(args.scalers) if args.scalers else {}
    output = new_run(args.output, [dataset])
    metadata = {'status': 'RUNNING', 'created_utc': datetime.now(timezone.utc).isoformat(), 'dataset': config['dataset'], 'models': names,
                'selected_records': len(selected), 'complete_cohort': len(selected) == len(rows), 'record_order_sha256': sequence_hash(row['sample_id'] for row in selected),
                'dataset_metadata_sha256': file_hash(dataset / 'ptbxl_database.csv'), 'versions': versions(['numpy','scipy','wfdb','resampy'])}
    write_json(output / 'metadata.json', metadata)
    log_event(output, f'START preprocessing: {len(selected)} recordings, {len(names)} models')
    write_json(output / 'config.resolved.json', {'experiment': config, 'paths': paths})
    try:
        training = [row for row in rows if row['split'] == 'train']
        for name, module in modules.items():
            recipe = module.RECIPE
            scaler = None
            if any(step['operation'] == 'training_scalar' for step in recipe['operations']):
                fit_ids = sequence_hash(row['sample_id'] for row in training)
                scaler = scalers.get(name)
                expected = {'fit_ids_sha256': fit_ids, 'dataset_metadata_sha256': metadata['dataset_metadata_sha256'], 'fit_records': len(training), 'leads': recipe['leads'], 'source_hz': recipe['source_hz']}
                if scaler:
                    if any(scaler.get(key) != value for key, value in expected.items()):
                        raise ValueError('Saved scaler does not match the training cohort and recipe')
                else:
                    print(f'{name}: fitting scalar on {len(training)} training recordings', flush=True)
                    def training_signals():
                        for fit_index, row in enumerate(training, 1):
                            yield ptbxl.read_record(dataset, row, recipe['source_hz'], recipe['leads'])
                            if fit_index % 1000 == 0: print(f'{name}: fitted {fit_index}/{len(training)}', flush=True)
                    scaler = {**fit_scalar(training_signals()), **expected}
                    scalers[name] = scaler
                write_json(output / 'scalers.json', scalers)
            directory = output / name
            directory.mkdir()
            array = np.lib.format.open_memmap(directory / 'signals.npy', mode='w+', dtype=np.float32, shape=(len(selected), *recipe['shape']))
            for index, row in enumerate(selected):
                native = ptbxl.read_record(dataset, row, recipe['source_hz'], recipe['leads'])
                array[index] = module.preprocess(native, recipe['source_hz'], scaler)
                if (index + 1) % 500 == 0: print(f'{name}: {index + 1}/{len(selected)}', flush=True)
            array.flush(); del array
            ids = np.asarray([row['sample_id'] for row in selected], dtype=str)
            np.save(directory / 'sample_ids.npy', ids, allow_pickle=False)
            save_records(directory / 'records.csv', [{'sample_id': row['sample_id'], 'patient_id': row['patient_id'], 'fold': row['fold'], 'split': row['split']} for row in selected])
            write_json(directory / 'recipe.json', recipe)
            write_json(directory / 'metadata.json', {**metadata, 'status': 'COMPLETE', 'model': name, 'shape': [len(selected), *recipe['shape']], 'dtype': 'float32',
                'recipe_sha256': file_hash(directory / 'recipe.json'), 'signals_sha256': file_hash(directory / 'signals.npy'), 'ids_file_sha256': file_hash(directory / 'sample_ids.npy'), 'records_sha256': file_hash(directory / 'records.csv'),
                'implementation_sha256': file_hash(module.__file__), 'signal_operations_sha256': file_hash(Path(__file__).resolve().parents[1] / 'common/signals.py'), 'scaler': scaler})
            print(f'{name}: prepared {len(selected)} recordings', flush=True)
            log_event(output, f'{name}: COMPLETE, shape={recipe["shape"]}')
        metadata['status'] = 'COMPLETE'
    except Exception as error:
        metadata.update(status='FAILED', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / 'metadata.json', metadata)
        log_event(output, metadata['status'])
    return metadata

def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=root / 'configs/experiment.json')
    parser.add_argument('--paths', type=Path, default=root / 'configs/paths.local.json')
    parser.add_argument('--models', nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--ids', type=Path, help='Text file with one recording ID per line, in the requested order')
    parser.add_argument('--scalers', type=Path, help='Reuse a saved scaler bank only if training fingerprints match')
    args = parser.parse_args()
    try: run(args)
    except (ValueError, OSError, KeyError, ImportError) as error: parser.exit(1, f'Preprocessing failed: {error}\n')

if __name__ == '__main__': main()
