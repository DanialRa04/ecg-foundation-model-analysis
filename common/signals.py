"""Channel-first signal operations independent of model and dataset names."""
from __future__ import annotations
import numpy as np
from scipy import signal as scipy_signal

def matrix(values):
    values = np.asarray(values)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError('Expected finite, nonempty [leads, samples] values')
    return values

def resample(values, source_hz, target_hz, method='fft'):
    values = matrix(values)
    if not np.isfinite([source_hz, target_hz]).all() or min(source_hz, target_hz) <= 0:
        raise ValueError('Sampling rates must be finite and positive')
    if source_hz == target_hz:
        return values.astype(np.float32, copy=False)
    length = int(round(values.shape[-1] * target_hz / source_hz))
    if length < 1:
        raise ValueError('Resampled recording would be empty')
    if method == 'fft':
        output = scipy_signal.resample(values, length, axis=-1)
    elif method == 'resampy':
        import resampy
        output = resampy.resample(values, source_hz, target_hz, axis=-1)
    elif method == 'linear':
        old = np.arange(values.shape[-1], dtype=float) / source_hz
        new = np.arange(length, dtype=float) / target_hz
        output = np.stack([np.interp(new, old, lead) for lead in values])
    else:
        raise ValueError(f'Unknown resampling method: {method}')
    return output.astype(np.float32)

def center_crop(values, samples):
    values = matrix(values)
    if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
        raise ValueError('Sample count must be a positive integer')
    size = values.shape[-1]
    if size >= samples:
        start = (size - samples) // 2
        return values[:, start:start + samples]
    left = (samples - size) // 2
    return np.pad(values, ((0, 0), (left, samples - size - left)))

def zscore(values, per_lead=False, epsilon=1e-8):
    values = matrix(values).astype(np.float64)
    axes = -1 if per_lead else None
    mean = values.mean(axis=axes, keepdims=True)
    std = values.std(axis=axes, keepdims=True)
    return ((values - mean) / (std + epsilon)).astype(np.float32)

def minmax(values):
    values = matrix(values)
    low, high = values.min(), values.max()
    return ((values - low) / (high - low) if high > low else np.zeros_like(values)).astype(np.float32)

def zero_phase_sos(sos, values):
    """Odd-padded forward/backward SOS filtering, with explicit steady states."""
    values = matrix(values).astype(np.float64)
    sos = np.asarray(sos, dtype=np.float64)
    edge = 3 * (2 * len(sos) + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    if values.shape[-1] <= edge:
        raise ValueError('Recording is shorter than the filter padding')
    extended = np.concatenate((2 * values[:, :1] - values[:, 1:edge + 1][:, ::-1], values, 2 * values[:, -1:] - values[:, -edge - 1:-1][:, ::-1]), axis=-1)
    states, scale = [], 1.0
    for section in sos:
        b, a = section[:3] / section[3], section[3:] / section[3]
        first = ((b[1] - a[1] * b[0]) + (b[2] - a[2] * b[0])) / a.sum()
        second = (1 + a[1]) * first - (b[1] - a[1] * b[0])
        states.append(scale * np.array([first, second]))
        scale *= b.sum() / a.sum()
    zi = np.asarray(states)[:, None, :]
    forward, _ = scipy_signal.sosfilt(sos, extended, axis=-1, zi=zi * extended[:, 0][None, :, None])
    backward, _ = scipy_signal.sosfilt(sos, forward[:, ::-1], axis=-1, zi=zi * forward[:, -1][None, :, None])
    return backward[:, ::-1][:, edge:-edge]

def bandpass(values, hz, low, high, order=5, method='sequential_sos'):
    if not 0 < low < high < hz / 2:
        raise ValueError('Filter cutoffs must lie inside Nyquist')
    if method == 'ba':
        b, a = scipy_signal.butter(order, [low / (hz / 2), high / (hz / 2)], btype='band')
        return scipy_signal.filtfilt(b, a, matrix(values), axis=-1).astype(np.float32)
    if method != 'sequential_sos':
        raise ValueError('Unknown filter method')
    for cutoff, kind in ((low, 'highpass'), (high, 'lowpass')):
        values = zero_phase_sos(scipy_signal.butter(order, cutoff, btype=kind, fs=hz, output='sos'), values)
    return values.astype(np.float32)

def fit_scalar(recordings):
    """Streaming population moments over all supplied lead/time values."""
    count, mean, m2 = 0, 0.0, 0.0
    for values in recordings:
        values = matrix(values).astype(np.float64)
        n, block_mean = values.size, float(values.mean())
        block_m2 = float(np.square(values - block_mean).sum())
        delta, total = block_mean - mean, count + n
        m2 += block_m2 + delta * delta * count * n / total
        mean += delta * n / total
        count = total
    if not count:
        raise ValueError('No training values for scalar fitting')
    std = float(np.sqrt(m2 / count))
    return {'mean': mean, 'scale': std if std > 0 else 1.0, 'value_count': count}

def execute_recipe(values, source_hz, recipe, scaler=None):
    values = matrix(values).astype(np.float32)
    if source_hz != recipe['source_hz']:
        raise ValueError(f"Recipe requires native {recipe['source_hz']} Hz recordings")
    hz = source_hz
    for step in recipe['operations']:
        kind = step['operation']
        if kind == 'resample':
            values = resample(values, hz, step['hz'], step['method']); hz = step['hz']
        elif kind == 'center_crop':
            values = center_crop(values, step['samples'])
        elif kind == 'zscore':
            values = zscore(values, step.get('per_lead', False))
        elif kind == 'minmax':
            values = minmax(values)
        elif kind == 'bandpass':
            values = bandpass(values, hz, step['low'], step['high'], step.get('order', 5), step['method'])
        elif kind == 'fixed_scale':
            mean, scale = np.asarray(step['mean']), np.asarray(step['scale'])
            if mean.ndim == 1: mean, scale = mean[:, None], scale[:, None]
            values = ((values - mean) / scale).astype(np.float32)
        elif kind == 'training_scalar':
            if scaler is None or not np.isfinite([scaler['mean'], scaler['scale']]).all() or scaler['scale'] <= 0:
                raise ValueError('A valid training-only scalar is required')
            values = ((values - scaler['mean']) / scaler['scale']).astype(np.float32)
        else:
            raise ValueError(f'Unknown operation: {kind}')
    values = matrix(values)
    if list(values.shape) != recipe['shape'] or hz != recipe['hz']:
        raise ValueError(f"Recipe produced incompatible shape/rate: {values.shape}, {hz}")
    return values.astype(np.float32)
