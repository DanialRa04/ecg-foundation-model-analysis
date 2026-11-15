"""Ranking metrics with explicit class support and missing-value handling."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def ranking_metrics(targets, probabilities, class_count, binary=False):
    targets = np.asarray(targets)
    probabilities = np.asarray(probabilities, dtype=float)
    if targets.ndim != 1 or probabilities.shape != (len(targets), class_count):
        raise ValueError("Targets and class probabilities have incompatible shapes")
    if (
        np.any((targets < 0) | (targets >= class_count))
        or not np.isfinite(probabilities).all()
    ):
        raise ValueError("Invalid class target or nonfinite probability")
    if np.any((probabilities < -1e-7) | (probabilities > 1 + 1e-7)) or not np.allclose(
        probabilities.sum(1), 1, atol=1e-5
    ):
        raise ValueError("Expected normalized class probabilities")
    per_class = []
    for label in [1] if binary else range(class_count):
        positive = targets == label
        support = int(positive.sum())
        valid = support > 0 and support < len(targets)
        per_class.append(
            {
                "class": label,
                "positive": support,
                "negative": len(targets) - support,
                "average_precision": float(
                    average_precision_score(positive, probabilities[:, label])
                )
                if valid
                else None,
                "auroc": float(roc_auc_score(positive, probabilities[:, label]))
                if valid
                else None,
                "status": "OK" if valid else "UNDEFINED",
                "reason": "" if valid else "class lacks positive or negative outcomes",
            }
        )
    included = [row["class"] for row in per_class if row["status"] == "OK"]
    result = {
        metric: float(
            np.mean([row[metric] for row in per_class if row[metric] is not None])
        )
        if included
        else None
        for metric in ("average_precision", "auroc")
    }
    return {
        **result,
        "included_classes": included,
        "per_class": per_class,
        "records": len(targets),
    }
