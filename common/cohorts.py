"""Deterministic whole-patient training subsets, independent of ECG models."""

import numpy as np


def fraction_key(value):
    return str(float(value))


def validate_records(rows):
    ids, patients = set(), {}
    for row in rows:
        sample, patient, split = row["sample_id"], row["patient_id"], row["split"]
        if (
            not sample
            or sample in ids
            or not patient
            or split not in {"train", "validation", "test"}
        ):
            raise ValueError("Invalid recording ID, patient ID, or split")
        ids.add(sample)
        if patients.setdefault(patient, split) != split:
            raise ValueError("A patient crosses dataset partitions")
    if not rows or set(patients.values()) != {"train", "validation", "test"}:
        raise ValueError("All three dataset partitions must be nonempty")


def balanced_patient_bins(rows, targets, tasks, settings, seed):
    """Fit bin assignments using training labels/demographics only.

    Patient order is independent of incoming recording order. Rarity is the
    sum of patient task-class counts divided by total training class counts.
    The objective is squared normalized feature load plus patient-count cost.
    """
    validate_records(rows)
    if settings["method"] != "greedy_label_balanced_bins":
        raise ValueError("Unknown patient subset method")
    if (
        not np.isfinite(settings["patient_count_penalty"])
        or settings["patient_count_penalty"] < 0
    ):
        raise ValueError("Patient-count penalty must be finite and nonnegative")
    supported = {
        "record_count",
        "task_positive_counts",
        "sex_counts",
        "age_band_counts",
    }
    if set(settings["balance_features"]) - supported:
        raise ValueError("Unknown patient balancing feature")
    training = [i for i, row in enumerate(rows) if row["split"] == "train"]
    patients = sorted({rows[i]["patient_id"] for i in training})
    bins = settings["bins"]
    if len(patients) < bins:
        raise ValueError("Fewer training patients than subset bins")
    columns, blocks, label_columns = [], [], []
    requested = settings["balance_features"]
    if "record_count" in requested:
        columns.append("record_count")
        blocks.append(np.ones((len(training), 1)))
    if "task_positive_counts" in requested:
        for task, definition in tasks.items():
            classes = (
                range(len(definition["classes"]))
                if definition["kind"] == "multiclass"
                else [1]
            )
            for label in classes:
                label_columns.append(len(columns))
                columns.append(f"{task}:class:{label}")
                blocks.append((targets[task][training] == label).astype(float)[:, None])
    if "sex_counts" in requested:
        sexes = sorted(
            {str(rows[i].get("sex", "")).strip() or "missing" for i in training}
        )
        for sex in sexes:
            columns.append(f"sex:{sex}")
            blocks.append(
                np.array(
                    [
                        (str(rows[i].get("sex", "")).strip() or "missing") == sex
                        for i in training
                    ]
                )[:, None]
            )
    if "age_band_counts" in requested:
        edges = settings["age_band_edges"]
        if (
            len(edges) < 2
            or edges[-1] is not None
            or any(b <= a for a, b in zip(edges[:-2], edges[1:-1]))
        ):
            raise ValueError("Age bands require ascending edges and an open final edge")
        ages = []
        for i in training:
            try:
                age = float(rows[i].get("age", ""))
            except (ValueError, TypeError):
                age = float("nan")
            ages.append(age)
        ages = np.asarray(ages)
        for index, low in enumerate(edges[:-1]):
            high = edges[index + 1]
            columns.append(f"age:{low}:{high}")
            blocks.append(
                ((ages >= low) & (ages < (np.inf if high is None else high)))[:, None]
            )
        columns.append("age:missing")
        blocks.append((~np.isfinite(ages) | (ages < edges[0]))[:, None])
    if not blocks:
        raise ValueError("At least one balancing feature is required")
    per_record = np.concatenate(blocks, axis=1).astype(float)
    index = {patient: i for i, patient in enumerate(patients)}
    features = np.zeros((len(patients), len(columns)))
    counts = np.zeros(len(patients), dtype=int)
    for row_index, record in enumerate(training):
        patient = index[rows[record]["patient_id"]]
        features[patient] += per_record[row_index]
        counts[patient] += 1
    totals = features.sum(0)
    target = np.where(totals > 0, totals / bins, 1.0)
    rarity = (
        (features[:, label_columns] / np.maximum(totals[label_columns], 1.0)).sum(1)
        if label_columns
        else np.zeros(len(patients))
    )
    rng = np.random.default_rng(seed)
    tie = rng.random(len(patients))
    order = sorted(
        range(len(patients)),
        key=lambda i: (-rarity[i], -counts[i], tie[i], patients[i]),
    )
    bin_order = rng.permutation(bins).tolist()
    priority = {value: i for i, value in enumerate(bin_order)}
    loads, sizes = np.zeros((bins, len(columns))), np.zeros(bins, dtype=int)
    assignments = {}
    patient_target = len(patients) / bins
    for i in order:
        cost = (((loads + features[i]) / target) ** 2 - (loads / target) ** 2).sum(1)
        cost += settings["patient_count_penalty"] * (
            ((sizes + 1) / patient_target) ** 2 - (sizes / patient_target) ** 2
        )
        best = min(range(bins), key=lambda b: (cost[b], sizes[b], priority[b]))
        assignments[patients[i]] = best + 1
        loads[best] += features[i]
        sizes[best] += 1
    return assignments, {
        "features": columns,
        "target_load": target.tolist(),
        "bin_loads": loads.tolist(),
        "bin_patients": sizes.tolist(),
    }


def training_mask(rows, fraction, settings):
    allowed = set(settings["fraction_bins"][fraction_key(fraction)])
    return np.asarray(
        [row["split"] == "train" and int(row["train_bin"]) in allowed for row in rows]
    )
