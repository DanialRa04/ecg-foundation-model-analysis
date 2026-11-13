"""Run aligned, training-only embedding evaluation with auditable fit outputs."""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from common.artifacts import (
    file_hash,
    log_event,
    new_run,
    sequence_hash,
    versions,
    write_json,
)
from common.cohorts import training_mask
from common.descriptor_math import spectral_and_collapse_descriptors
from common.experiment_data import load_embeddings, load_manifest
from common.metrics import ranking_metrics
from common.probes import linear_probe, neural_probe
from common.tables import write_table

METRIC_COLUMNS = [
    "model",
    "layer",
    "fraction",
    "task",
    "probe",
    "seed",
    "split",
    "metric",
    "value",
    "included_classes",
    "status",
    "reason",
    "low_support",
]


def aggregate_seeds(rows, expected_seeds):
    groups = defaultdict(list)
    keys = ["model", "layer", "fraction", "task", "probe", "split", "metric"]
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    result = []
    for key, values in sorted(groups.items()):
        seeds = [row["seed"] for row in values]
        if len(seeds) != len(set(seeds)) or set(seeds) != set(expected_seeds):
            raise ValueError("Missing or duplicate probe seed results")
        finite = [
            float(row["value"])
            for row in values
            if row["value"] is not None and np.isfinite(row["value"])
        ]
        complete = len(finite) == len(expected_seeds)
        result.append(
            {
                **dict(zip(keys, key)),
                "mean": float(np.mean(finite)) if complete else None,
                "sample_std": float(np.std(finite, ddof=1))
                if complete and len(finite) > 1
                else None,
                "valid_seeds": len(finite),
                "planned_seeds": len(expected_seeds),
                "status": "OK" if complete else "UNDEFINED",
                "reason": "" if complete else "at least one seed has no valid metric",
            }
        )
    return result


def _support(values, count):
    return np.bincount(values, minlength=count).tolist()


def run_evaluation(
    manifest_path, embedding_paths, output_path, device="cpu", threads=1
):
    if threads < 1:
        raise ValueError("Thread count must be positive")
    import torch

    torch.set_num_threads(threads)
    manifest = load_manifest(manifest_path)
    bundle = manifest["bundle"]
    config = bundle["experiment"]
    supported = {
        "scaler": "training_subset_standard_scaler",
        "multiclass_average": "macro_one_vs_rest_evaluable_classes",
        "single_class_metric": "null_with_support_flag",
        "invalid_training_cohort": "skip_with_reason",
        "aggregate_seeds": "mean_and_sample_standard_deviation",
    }
    if any(config["evaluation"][key] != value for key, value in supported.items()):
        raise ValueError("Unsupported evaluation policy")
    if (
        config["representation"]["layer"] != "final"
        or not config["representation"]["frozen"]
    ):
        raise ValueError("Evaluation requires configured frozen final representations")
    catalog = {row["name"]: row for row in bundle["models"]["models"]}
    paths = [Path(path).resolve() for path in embedding_paths]
    if not paths:
        raise ValueError("Provide at least one embedding directory")
    output = new_run(output_path, [manifest["path"], *paths])
    metadata = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_metadata_sha256": file_hash(manifest["path"] / "metadata.json"),
        "manifest_record_order_sha256": manifest["metadata"]["record_order_sha256"],
        "models": [],
        "device": device,
        "cpu_threads": threads,
        "versions": versions(["numpy", "scipy", "scikit-learn", "torch"]),
        "embedding_inputs": [],
    }
    write_json(output / "metadata.json", metadata)
    log_event(output, "START embedding evaluation")
    write_json(output / "config.resolved.json", bundle)
    metrics, per_class, descriptors, fit_index, identities = [], [], [], [], []
    generated = ["config.resolved.json"]
    seen = set()
    records = manifest["records"]
    split_values = np.asarray([row["split"] for row in records])
    try:
        for source in paths:
            embedding_meta, values = load_embeddings(source, manifest)
            model = embedding_meta["model"]
            if model in seen or model not in config["model_ids"]:
                raise ValueError("Duplicate or unselected embedding model")
            seen.add(model)
            metadata["models"].append(model)
            metadata["embedding_inputs"].append(
                {
                    "model": model,
                    "metadata_sha256": file_hash(source / "metadata.json"),
                    "embeddings_sha256": embedding_meta["embeddings_sha256"],
                }
            )
            digest = hashlib.sha256()
            for start in range(0, len(values), 256):
                digest.update(
                    np.ascontiguousarray(
                        values[start : start + 256], dtype="<f4"
                    ).tobytes()
                )
            identities.append(
                {
                    "model": model,
                    "aligned_embeddings_sha256": digest.hexdigest(),
                    "features": values.shape[1],
                    "analysis_group": catalog[model]["analysis_group"],
                    "evaluation_dataset_exposure": catalog[model][
                        "evaluation_dataset_exposure"
                    ],
                }
            )
            for fraction in config["training"]["fractions"]:
                train = training_mask(
                    records, fraction, config["training"]["patient_subset"]
                )
                if not train.any():
                    raise ValueError("Training selection is empty")
                descriptor_values = spectral_and_collapse_descriptors(
                    values[train], **bundle["descriptors"]["parameters"]
                )
                descriptors.append(
                    {
                        "model": model,
                        "layer": "final",
                        "fraction": fraction,
                        "records": int(train.sum()),
                        "features": values.shape[1],
                        "record_order_sha256": sequence_hash(manifest["ids"][train]),
                        **{
                            item["id"]: descriptor_values[item["id"]]
                            for item in bundle["descriptors"]["panel"]
                        },
                    }
                )
                scaler = StandardScaler().fit(values[train])
                standardized = scaler.transform(values).astype(np.float32)
                scaler_name = f"scaler_{len(descriptors):03d}.json"
                write_json(
                    output / scaler_name,
                    {
                        "model": model,
                        "fraction": fraction,
                        "mean": scaler.mean_.tolist(),
                        "scale": scaler.scale_.tolist(),
                        "variance": scaler.var_.tolist(),
                        "fit_records": int(train.sum()),
                        "fit_ids_sha256": sequence_hash(manifest["ids"][train]),
                    },
                )
                generated.append(scaler_name)
                for task in config["task_ids"]:
                    definition = manifest["tasks"][task]
                    targets = manifest["targets"][task]
                    count = len(definition["classes"])
                    binary = definition["kind"] == "binary"
                    masks = {
                        split: (
                            (train if split == "train" else split_values == split)
                            & (targets >= 0)
                        )
                        for split in ("train", "validation", "test")
                    }
                    support = {
                        split: _support(targets[mask], count)
                        for split, mask in masks.items()
                    }
                    missing = count < 2 or any(value == 0 for value in support["train"])
                    for probe in config["evaluation"]["probe_ids"]:
                        for seed in config["training"]["probe_seeds"]:
                            identity = {
                                "model": model,
                                "layer": "final",
                                "fraction": fraction,
                                "task": task,
                                "probe": probe,
                                "seed": seed,
                            }
                            fit_dir = output / "fits" / f"fit_{len(fit_index) + 1:05d}"
                            fit_dir.mkdir(parents=True)
                            reason = (
                                "training subset lacks a retained class"
                                if missing
                                else ""
                            )
                            if not reason and probe == "ann":
                                dummy = np.full(
                                    (int(masks["validation"].sum()), count), 1 / count
                                )
                                if (
                                    ranking_metrics(
                                        targets[masks["validation"]],
                                        dummy,
                                        count,
                                        binary,
                                    )["average_precision"]
                                    is None
                                ):
                                    reason = "validation average precision is undefined"
                            info = {
                                **identity,
                                "classes": definition["classes"],
                                "support": support,
                                "status": "SKIPPED" if reason else "COMPLETE",
                                "reason": reason,
                            }
                            predictions = {}
                            if not reason:
                                evaluate = {
                                    split: standardized[masks[split]]
                                    for split in ("validation", "test")
                                }
                                if probe == "linear":
                                    predictions, details = linear_probe(
                                        standardized[masks["train"]],
                                        targets[masks["train"]],
                                        evaluate,
                                        bundle["probes"]["probes"][probe],
                                        seed,
                                        count,
                                    )
                                elif probe == "ann":
                                    predictions, details = neural_probe(
                                        standardized[masks["train"]],
                                        targets[masks["train"]],
                                        targets[masks["validation"]],
                                        evaluate,
                                        bundle["probes"]["probes"][probe],
                                        seed,
                                        count,
                                        binary,
                                        device,
                                    )
                                else:
                                    raise ValueError("Unknown probe implementation")
                                info.update(details)
                                prediction_rows = []
                                for split in ("validation", "test"):
                                    report = ranking_metrics(
                                        targets[masks[split]],
                                        predictions[split],
                                        count,
                                        binary,
                                    )
                                    indices = np.flatnonzero(masks[split])
                                    for position, index in enumerate(indices):
                                        prediction_rows.append(
                                            {
                                                "sample_id": records[index][
                                                    "sample_id"
                                                ],
                                                "split": split,
                                                "target": int(targets[index]),
                                                **{
                                                    f"probability_{label}": float(
                                                        predictions[split][
                                                            position, label
                                                        ]
                                                    )
                                                    for label in range(count)
                                                },
                                            }
                                        )
                                    for row in report["per_class"]:
                                        per_class.append(
                                            {
                                                **identity,
                                                "split": split,
                                                "class_label": definition["classes"][
                                                    row["class"]
                                                ],
                                                **row,
                                            }
                                        )
                                    low = (
                                        min(support[split], default=0)
                                        < config["reliability"][
                                            "minimum_evaluation_minority_warning"
                                        ]
                                        or min(support["train"], default=0)
                                        < config["reliability"][
                                            "minimum_training_minority_warning"
                                        ]
                                    )
                                    for metric in ("average_precision", "auroc"):
                                        metrics.append(
                                            {
                                                **identity,
                                                "split": split,
                                                "metric": metric,
                                                "value": report[metric],
                                                "included_classes": json.dumps(
                                                    report["included_classes"]
                                                ),
                                                "status": "OK"
                                                if report[metric] is not None
                                                else "UNDEFINED",
                                                "reason": ""
                                                if report[metric] is not None
                                                else "no evaluable class",
                                                "low_support": low,
                                            }
                                        )
                                prediction_path = fit_dir / "predictions.csv"
                                write_table(
                                    prediction_path,
                                    [
                                        "sample_id",
                                        "split",
                                        "target",
                                        *[
                                            f"probability_{label}"
                                            for label in range(count)
                                        ],
                                    ],
                                    prediction_rows,
                                )
                                generated.append(
                                    prediction_path.relative_to(output).as_posix()
                                )
                            else:
                                for split in ("validation", "test"):
                                    for metric in ("average_precision", "auroc"):
                                        metrics.append(
                                            {
                                                **identity,
                                                "split": split,
                                                "metric": metric,
                                                "value": None,
                                                "included_classes": "[]",
                                                "status": "SKIPPED",
                                                "reason": reason,
                                                "low_support": True,
                                            }
                                        )
                            write_json(fit_dir / "fit.json", info)
                            generated.append(
                                (fit_dir / "fit.json").relative_to(output).as_posix()
                            )
                            fit_index.append(
                                {
                                    **identity,
                                    "directory": fit_dir.relative_to(output).as_posix(),
                                    "status": info["status"],
                                    "reason": reason,
                                }
                            )
                    print(f"{model}: fraction={fraction}, task={task}", flush=True)
                log_event(output, f"{model} fraction={fraction}: COMPLETE")
            del values, standardized
        aggregate = aggregate_seeds(metrics, config["training"]["probe_seeds"])
        write_table(output / "metrics.csv", METRIC_COLUMNS, metrics)
        class_columns = [
            "model",
            "layer",
            "fraction",
            "task",
            "probe",
            "seed",
            "split",
            "class_label",
            "class",
            "positive",
            "negative",
            "average_precision",
            "auroc",
            "status",
            "reason",
        ]
        write_table(output / "class_metrics.csv", class_columns, per_class)
        write_table(output / "seed_summary.csv", list(aggregate[0]), aggregate)
        write_table(output / "descriptors.csv", list(descriptors[0]), descriptors)
        write_table(output / "fits.csv", list(fit_index[0]), fit_index)
        write_table(output / "representations.csv", list(identities[0]), identities)
        generated.extend(
            [
                "metrics.csv",
                "class_metrics.csv",
                "seed_summary.csv",
                "descriptors.csv",
                "fits.csv",
                "representations.csv",
            ]
        )
        implementation = [
            "common/evaluation.py",
            "common/probes.py",
            "common/metrics.py",
            "common/experiment_data.py",
            "common/cohorts.py",
            "common/descriptor_math.py",
            "common/tables.py",
            "common/artifacts.py",
        ]
        metadata.update(
            status="COMPLETE",
            fits=len(fit_index),
            skipped_fits=sum(row["status"] == "SKIPPED" for row in fit_index),
            files_sha256={name: file_hash(output / name) for name in generated},
            implementation_sha256={
                name: file_hash(Path(__file__).resolve().parents[1] / name)
                for name in implementation
            },
        )
    except Exception as error:
        metadata.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        write_json(output / "metadata.json", metadata)
        log_event(output, metadata["status"])
    return metadata
