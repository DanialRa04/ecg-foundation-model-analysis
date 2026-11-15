"""Read PTB-XL metadata and local WFDB recordings without changing inputs."""
import csv
from pathlib import Path
import numpy as np

CANONICAL_LEADS = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def metadata(root, splits):
    root = Path(root).resolve()
    with (root / 'ptbxl_database.csv').open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids, patient_splits = set(), {}
    reverse = {str(fold): name for name, folds in splits.items() for fold in folds}
    if len(reverse) != sum(len(folds) for folds in splits.values()):
        raise ValueError('Split fold assignments overlap')
    result = []
    for row in rows:
        sample_id, patient_id = row['ecg_id'].strip(), row['patient_id'].strip()
        if not sample_id or sample_id in ids or not patient_id:
            raise ValueError('Empty or duplicate recording ID, or empty patient ID')
        ids.add(sample_id)
        fold = str(int(float(row['strat_fold'])))
        if fold not in reverse:
            raise ValueError(f'Unassigned fold: {fold}')
        split = reverse[fold]
        previous = patient_splits.setdefault(patient_id, split)
        if previous != split:
            raise ValueError('A patient crosses dataset partitions')
        result.append({**row, 'sample_id': sample_id, 'split': split, 'fold': fold})
    if not result:
        raise ValueError('Dataset metadata is empty')
    return result

def record_path(root, row, source_hz):
    root = Path(root).resolve()
    column, folder = ('filename_lr', 'records100') if source_hz == 100 else ('filename_hr', 'records500') if source_hz == 500 else (None, None)
    if column is None:
        raise ValueError('PTB-XL native rate must be 100 or 500 Hz')
    relative = Path(row[column])
    path = (root / relative).resolve()
    if relative.is_absolute() or root not in path.parents or relative.parts[0] != folder:
        raise ValueError('Invalid dataset recording path')
    if not path.with_suffix('.hea').is_file() or not path.with_suffix('.dat').is_file():
        raise FileNotFoundError('Required local WFDB pair is missing')
    return path

def read_record(root, row, source_hz, leads):
    import wfdb
    values, fields = wfdb.rdsamp(str(record_path(root, row, source_hz)))
    names = [name.strip().lower() for name in fields['sig_name']]
    if len(set(names)) != len(names) or fields['fs'] != source_hz:
        raise ValueError('Duplicate leads or incompatible native rate')
    selected = []
    for lead in leads:
        if lead.lower() not in names:
            raise ValueError(f'Missing required lead: {lead}')
        index = names.index(lead.lower())
        unit = fields['units'][index].strip().lower().replace('µ', 'u').replace('μ', 'u')
        scale = {'mv': 1.0, 'uv': 0.001, 'v': 1000.0}.get(unit)
        if scale is None:
            raise ValueError(f'Unknown physical unit: {unit}')
        selected.append(values[:, index] * scale)
    output = np.asarray(selected, dtype=np.float32)
    if output.shape != (len(leads), int(10 * source_hz)) or not np.isfinite(output).all():
        raise ValueError('Expected a finite ten-second recording')
    return output
