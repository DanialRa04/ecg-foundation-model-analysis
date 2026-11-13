"""Model-level correlations, grouped uncertainty, and false-discovery control."""

from collections import defaultdict
import numpy as np
from scipy import stats


def correlation(x, y, method):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        len(x) < 3
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
        or np.ptp(x) == 0
        or np.ptp(y) == 0
    ):
        return None, None
    function = {
        "spearman": stats.spearmanr,
        "pearson": stats.pearsonr,
        "kendall": stats.kendalltau,
    }[method]
    result = function(x, y, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def bootstrap_weights(groups, replicates, seed):
    names = sorted(set(groups))
    if not names:
        return np.empty((replicates, 0))
    group_counts = np.random.default_rng(seed).multinomial(
        len(names), np.full(len(names), 1 / len(names)), size=replicates
    )
    lookup = {name: i for i, name in enumerate(names)}
    return group_counts[:, [lookup[name] for name in groups]].astype(float)


def weighted_correlations(x, y, weights, method):
    """Exact coefficients for repeated rows represented by integer weights."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if method == "kendall":
        left, right = np.triu_indices(len(x), 1)
        pair_weights = weights[:, left] * weights[:, right]
        dx, dy = np.sign(x[right] - x[left]), np.sign(y[right] - y[left])
        numerator = pair_weights @ (dx * dy)
        denominator = np.sqrt((pair_weights @ (dx != 0)) * (pair_weights @ (dy != 0)))
    else:
        if method == "spearman":
            # Average ranks of the expanded multiset, including repeated ties.
            rx = (
                weights @ (x[:, None] < x[None, :])
                + (weights @ (x[:, None] == x[None, :]) + 1) / 2
            )
            ry = (
                weights @ (y[:, None] < y[None, :])
                + (weights @ (y[:, None] == y[None, :]) + 1) / 2
            )
        elif method == "pearson":
            rx, ry = x[None, :], y[None, :]
        else:
            raise ValueError("Unknown correlation method")
        total = weights.sum(1)
        rx = rx - (weights * rx).sum(1)[:, None] / total[:, None]
        ry = ry - (weights * ry).sum(1)[:, None] / total[:, None]
        numerator = (weights * rx * ry).sum(1)
        denominator = np.sqrt((weights * rx**2).sum(1) * (weights * ry**2).sum(1))
    result = np.full(len(weights), np.nan)
    selected = weights > 0
    varied_x = np.max(np.where(selected, x[None, :], -np.inf), axis=1) > np.min(
        np.where(selected, x[None, :], np.inf), axis=1
    )
    varied_y = np.max(np.where(selected, y[None, :], -np.inf), axis=1) > np.min(
        np.where(selected, y[None, :], np.inf), axis=1
    )
    valid = (denominator > 0) & varied_x & varied_y & (weights.sum(1) >= 3)
    result[valid] = np.clip(numerator[valid] / denominator[valid], -1, 1)
    return result


def association(x, y, groups, method, settings, bootstrap=True):
    estimate, p = correlation(x, y, method)
    independent = len(set(groups))
    result = {
        "estimate": estimate,
        "p_value": p,
        "ci_low": None,
        "ci_high": None,
        "models": len(x),
        "independent_groups": independent,
        "bootstrap_requested": settings["bootstrap_replicates"] if bootstrap else 0,
        "bootstrap_valid": 0,
        "status": "OK" if estimate is not None else "UNDEFINED",
        "reason": ""
        if estimate is not None
        else "insufficient, missing, or constant values",
    }
    if estimate is not None and bootstrap and independent >= 3:
        values = weighted_correlations(
            x,
            y,
            bootstrap_weights(
                groups, settings["bootstrap_replicates"], settings["bootstrap_seed"]
            ),
            method,
        )
        values = values[np.isfinite(values)]
        result["bootstrap_valid"] = len(values)
        if len(values) >= 2:
            tail = (1 - settings["confidence_level"]) / 2
            result["ci_low"], result["ci_high"] = np.quantile(
                values, [tail, 1 - tail]
            ).tolist()
        else:
            result["reason"] = "fewer than two valid group-bootstrap resamples"
    elif estimate is not None and bootstrap:
        result["reason"] = "fewer than three independent groups for an interval"
    return result


def adjust_fdr(rows, family_keys):
    families = defaultdict(list)
    for i, row in enumerate(rows):
        row["q_value"] = None
        row["fdr_tests"] = 0
        if row["p_value"] is not None and np.isfinite(row["p_value"]):
            families[tuple(row[key] for key in family_keys)].append(i)
    for indices in families.values():
        ordered = sorted(indices, key=lambda i: rows[i]["p_value"])
        adjusted = np.asarray(
            [
                rows[i]["p_value"] * len(indices) / (rank + 1)
                for rank, i in enumerate(ordered)
            ]
        )
        adjusted = np.minimum.accumulate(adjusted[::-1])[::-1].clip(0, 1)
        for i, value in zip(ordered, adjusted):
            rows[i]["q_value"] = float(value)
            rows[i]["fdr_tests"] = len(indices)


def overall_percentiles(outcomes, model_order, tasks):
    """Average within-task percentile ranks on one common complete model set."""
    complete = [
        model
        for model in model_order
        if all(outcomes[task][model] is not None for task in tasks)
    ]
    if len(complete) < 2:
        return {}, complete
    values = np.array([[outcomes[task][model] for task in tasks] for model in complete])
    ranks = (stats.rankdata(values, method="average", axis=0) - 1) / (len(complete) - 1)
    return dict(zip(complete, ranks.mean(1).tolist())), complete


def partial_pearson(x, y, controls):
    controls = np.asarray(controls, dtype=float)
    design = np.column_stack([np.ones(len(x)), controls])
    rank = np.linalg.matrix_rank(design)
    if rank != design.shape[1] or len(x) - rank < 3:
        return None, None
    residual_x = np.asarray(x) - design @ np.linalg.lstsq(design, x, rcond=None)[0]
    residual_y = np.asarray(y) - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    if any(
        np.linalg.norm(residual)
        <= 1e-10
        * max(
            np.linalg.norm(np.asarray(original) - np.mean(original)),
            np.finfo(float).tiny,
        )
        for residual, original in ((residual_x, x), (residual_y, y))
    ):
        return None, None
    estimate, _ = correlation(residual_x, residual_y, "pearson")
    if estimate is None:
        return None, None
    degrees = len(x) - rank - 1
    p = (
        0.0
        if abs(estimate) >= 1
        else float(
            2
            * stats.t.sf(abs(estimate) * np.sqrt(degrees / (1 - estimate**2)), degrees)
        )
    )
    return estimate, p


def partial_bootstrap(x, y, controls, groups, settings):
    values = []
    for weights in bootstrap_weights(
        groups, settings["bootstrap_replicates"], settings["bootstrap_seed"]
    ).astype(int):
        selected = np.repeat(np.arange(len(x)), weights)
        value, _ = partial_pearson(
            np.asarray(x)[selected], np.asarray(y)[selected], controls[selected]
        )
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None, None, len(values)
    tail = (1 - settings["confidence_level"]) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(low), float(high), len(values)
