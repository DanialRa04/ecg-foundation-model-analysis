"""Explicit run directories, hashes, JSON, and array validation."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()

def sequence_hash(values):
    result = hashlib.sha256()
    for value in values:
        encoded = str(value).encode('utf-8')
        result.update(len(encoded).to_bytes(8, 'little'))
        result.update(encoded)
    return result.hexdigest()

def json_values(value):
    if isinstance(value, dict):
        return {str(key): json_values(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_values(item) for item in value]
    if isinstance(value, np.generic):
        return json_values(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value

def write_json(path, value):
    path = Path(path)
    rendered = json.dumps(json_values(value), indent=2, allow_nan=False) + '\n'
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent, prefix=f'.{path.name}.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists(): temporary.unlink()

def new_run(path, inputs=()):
    path = Path(path).resolve()
    for source in inputs:
        source = Path(source).resolve()
        if path == source or source in path.parents or path in source.parents:
            raise ValueError('Output and input locations must be separate')
    path.mkdir(parents=True, exist_ok=False)
    return path

def save_records(path, rows):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def log_event(directory, message):
    with (Path(directory) / 'run.log').open('a', encoding='utf-8') as stream:
        stream.write(f'{datetime.now(timezone.utc).isoformat()} {message}\n')

def load_ids(path):
    ids = np.load(path, allow_pickle=False)
    if ids.ndim != 1 or ids.dtype.kind not in 'USiu' or not len(ids):
        raise ValueError('Recording IDs must be a nonempty one-dimensional string/integer array')
    ids = ids.astype(str)
    if any(not item.strip() for item in ids) or len(set(ids)) != len(ids):
        raise ValueError('Recording IDs must be nonempty and unique')
    return ids

def array_info(path, rank):
    values = np.load(path, mmap_mode='r', allow_pickle=False)
    if values.ndim != rank or min(values.shape) <= 0 or values.dtype.kind not in 'fiu':
        raise ValueError(f'Expected nonempty rank-{rank} numeric array')
    return values

def versions(names):
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result
