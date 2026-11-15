"""Exact, label-free descriptors evaluated on complete embedding cohorts."""

from __future__ import annotations

from typing import Any

import numpy as np


def shannon_entropy(probabilities: np.ndarray) -> float:
    positive = probabilities[probabilities > 0.0]
    return -float(np.sum(positive * np.log(positive)))


def mean_pairwise_cosine(values: np.ndarray) -> float:
    """Exact off-diagonal mean cosine without materializing an N by N matrix."""
    norms = np.linalg.norm(values, axis=1)
    valid = norms > 0.0
    normalized = values[valid] / norms[valid, None]
    count = len(normalized)
    if count < 2:
        return float("nan")
    summed = normalized.sum(axis=0, dtype=np.float64)
    result = (float(summed @ summed) - count) / (count * (count - 1))
    return float(np.clip(result, -1.0, 1.0))


def _spectral_concentration(probabilities: np.ndarray, fraction: float) -> float:
    count = max(1, int(np.ceil(fraction * len(probabilities))))
    return float(probabilities[:count].sum())


def _self_cluster(values: np.ndarray) -> float:
    """Exact SelfCluster without materializing the sample-by-sample Gram matrix."""
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms == 0.0):
        return float("nan")
    normalized = values / norms[:, None]
    sample_count, dimension = normalized.shape
    if sample_count < 2 or dimension < 2:
        return float("nan")
    # ||W W.T||_F^2 == ||W.T W||_F^2.  The right-hand side is d by d,
    # so this remains exact and RAM-safe for the complete ECG cohort.
    feature_gram = normalized.T @ normalized
    pairwise_squared_dot_sum = float(np.square(feature_gram).sum())
    expected_isotropic = sample_count + sample_count * (sample_count - 1) / dimension
    collapsed = float(sample_count**2)
    return (pairwise_squared_dot_sum - expected_isotropic) / (
        collapsed - expected_isotropic
    )


def _standard_incoherence(
    values: np.ndarray,
    right_vectors: np.ndarray,
    singular_values: np.ndarray,
    numerical_rank_mask: np.ndarray,
) -> float:
    """Compute standard mu_0-incoherence for the numerical-rank SVD basis."""
    rank = int(numerical_rank_mask.sum())
    if rank == 0:
        return float("nan")
    retained_vectors = right_vectors[:, numerical_rank_mask]
    retained_singular = singular_values[numerical_rank_mask]
    # U = M V Sigma^-1. This avoids storing an n by d U matrix beyond the
    # temporary product already needed for its row leverage scores.
    left_vectors = (values @ retained_vectors) / retained_singular
    left_leverage_max = float(np.square(left_vectors).sum(axis=1).max())
    right_leverage_max = float(np.square(retained_vectors).sum(axis=1).max())
    return max(
        values.shape[0] * left_leverage_max / rank,
        values.shape[1] * right_leverage_max / rank,
    )


def spectral_and_collapse_descriptors(
    values: np.ndarray,
    *,
    variance_tolerance: float,
    rankme_epsilon: float,
) -> dict[str, Any]:
    """Calculate exact global geometry from every row of ``values``.

    Spectral energy descriptors use the feature-centered matrix and squared
    singular values. RankMe follows its singular-value probability convention
    on the uncentered matrix. No ECG rows are sampled.
    """
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or min(values.shape) == 0:
        raise ValueError(f"Expected a nonempty [ECG, feature] matrix, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("Embedding contains NaN or infinite values")

    sample_count = len(values)
    centroid = values.mean(axis=0)
    centered = values - centroid
    centered_gram = centered.T @ centered
    centered_energy = np.linalg.eigvalsh(centered_gram)
    centered_energy = np.clip(centered_energy, 0.0, None)[::-1]
    uncentered_gram = centered_gram + sample_count * np.outer(centroid, centroid)
    uncentered_energy, uncentered_vectors = np.linalg.eigh(uncentered_gram)
    uncentered_energy = np.clip(uncentered_energy, 0.0, None)[::-1]
    uncentered_vectors = uncentered_vectors[:, ::-1]
    uncentered_singular = np.sqrt(uncentered_energy)
    numerical_tolerance = (
        np.finfo(np.float64).eps * max(values.shape) * uncentered_singular[0]
    )
    numerical_rank_mask = uncentered_singular > numerical_tolerance
    numerical_singular = uncentered_singular[numerical_rank_mask]
    rankme_p = uncentered_singular / uncentered_singular.sum() + rankme_epsilon
    rankme_p = rankme_p / rankme_p.sum()
    rankme = float(np.exp(shannon_entropy(rankme_p)))

    feature_variance = np.mean(np.square(centered), axis=0, dtype=np.float64)
    energy = centered_energy
    singular = np.sqrt(energy)
    total_energy = float(energy.sum())
    if total_energy <= 0.0:
        raise ValueError("Centered embedding has zero spectral energy")
    probabilities = energy / total_energy
    spectral_entropy = shannon_entropy(probabilities)
    maximum_rank = min(values.shape)

    keep = singular > singular[0] * 1e-12
    ranks = np.arange(1, len(singular) + 1, dtype=np.float64)[keep]
    log_rank = np.log(ranks)
    log_singular = np.log(singular[keep])
    if len(log_rank) >= 2:
        decay_slope, decay_intercept = np.polyfit(log_rank, log_singular, 1)
        prediction = decay_slope * log_rank + decay_intercept
        residual = float(np.square(log_singular - prediction).sum())
        total = float(np.square(log_singular - log_singular.mean()).sum())
        decay_r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    else:
        decay_slope = float("nan")
        decay_r_squared = float("nan")

    variance_mean = float(feature_variance.mean())
    variance_std = float(feature_variance.std())
    absolute_near_zero = int((feature_variance <= variance_tolerance).sum())
    relative_threshold = float(feature_variance.max()) * variance_tolerance
    near_zero = int((feature_variance <= relative_threshold).sum())

    return {
        "rankme": rankme,
        "normalized_rankme": rankme / maximum_rank,
        "pseudo_condition_number": float(
            numerical_singular[0] / numerical_singular[-1]
        ),
        "standard_incoherence": _standard_incoherence(
            values,
            uncentered_vectors,
            uncentered_singular,
            numerical_rank_mask,
        ),
        "selfcluster": _self_cluster(values),
        "effective_rank": float(np.exp(spectral_entropy)),
        "participation_ratio": float(total_energy**2 / np.square(energy).sum()),
        "stable_rank": float(total_energy / energy.max()),
        "spectral_entropy": spectral_entropy,
        "normalized_spectral_entropy": (
            spectral_entropy / np.log(maximum_rank) if maximum_rank > 1 else 0.0
        ),
        "singular_value_decay_exponent": float(-decay_slope),
        "singular_value_decay_r_squared": float(decay_r_squared),
        "largest_spectral_component_fraction": float(probabilities[0]),
        "top_5_percent_spectral_energy": _spectral_concentration(probabilities, 0.05),
        "top_10_percent_spectral_energy": _spectral_concentration(probabilities, 0.10),
        "top_25_percent_spectral_energy": _spectral_concentration(probabilities, 0.25),
        "feature_variance_minimum": float(feature_variance.min()),
        "feature_variance_mean": variance_mean,
        "feature_variance_median": float(np.median(feature_variance)),
        "feature_variance_maximum": float(feature_variance.max()),
        "feature_variance_coefficient_of_variation": (
            variance_std / variance_mean if variance_mean > 0.0 else float("nan")
        ),
        "near_zero_variance_dimension_count": near_zero,
        "near_zero_variance_dimension_fraction": near_zero / values.shape[1],
        "near_zero_variance_absolute_dimension_count": absolute_near_zero,
        "near_zero_variance_absolute_dimension_fraction": absolute_near_zero / values.shape[1],
        "feature_variance_relative_tolerance": float(variance_tolerance),
        "feature_variance_relative_threshold": relative_threshold,
        "centered_mean_pairwise_cosine_similarity": mean_pairwise_cosine(centered),
        "centroid_to_sample_norm_ratio": float(
            np.linalg.norm(centroid) / np.linalg.norm(values, axis=1).mean()
        ),
        "uncentered_mean_pairwise_cosine_similarity": mean_pairwise_cosine(values),
    }
