"""Save task labels and shared patient-level selections without reading ECG signals."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common.artifacts import (
    file_hash,
    log_event,
    new_run,
    sequence_hash,
    versions,
    write_json,
)
from common.cohorts import balanced_patient_bins, training_mask
from common.configuration import load_experiment
from common.paths import load_paths
from common.tables import write_table
from integrations.datasets import ptbxl, ptbxl_targets


def run(args):
    bundle = load_experiment(args.config)
    config = bundle["experiment"]
    if config["dataset"]["name"] != "PTB-XL":
        raise ValueError("This manifest entry point uses PTB-XL metadata")
    paths, _ = load_paths(args.paths)
    dataset = paths["dataset_root"]
    rows = ptbxl.metadata(dataset, config["dataset"]["splits"])
    taxonomy = dataset / bundle["tasks"]["taxonomy_file"]
    if dataset.resolve() not in taxonomy.resolve().parents:
        raise ValueError("Taxonomy must remain inside the dataset directory")
    targets, tasks = ptbxl_targets.build_targets(
        rows, bundle["tasks"], config["task_ids"], taxonomy
    )
    assignments, balance = balanced_patient_bins(
        rows,
        targets,
        tasks,
        config["training"]["patient_subset"],
        config["training"]["manifest_seed"],
    )
    records = [
        {
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "fold": row["fold"],
            "split": row["split"],
            "train_bin": assignments[row["patient_id"]]
            if row["split"] == "train"
            else "",
        }
        for row in rows
    ]
    output = new_run(args.output, [dataset, args.config, args.paths])
    metadata = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": config["dataset"],
        "records": len(rows),
        "patients": len({row["patient_id"] for row in rows}),
        "record_order_sha256": sequence_hash(row["sample_id"] for row in rows),
        "dataset_metadata_sha256": file_hash(dataset / "ptbxl_database.csv"),
        "taxonomy_sha256": file_hash(taxonomy),
        "versions": versions(["numpy"]),
        "scope": "complete metadata cohort; no signal files read",
    }
    write_json(output / "metadata.json", metadata)
    log_event(output, "START manifest preparation")
    try:
        write_json(output / "config.resolved.json", bundle)
        write_json(
            output / "tasks.json",
            {
                "tasks": tasks,
                "missing_target": -1,
                "class_selection": "full training cohort only",
            },
        )
        write_json(output / "balance.json", balance)
        write_table(output / "records.csv", list(records[0]), records)
        write_table(
            output / "targets.csv",
            ["sample_id", *tasks],
            (
                {
                    "sample_id": row["sample_id"],
                    **{
                        name: int(values[i]) if values[i] >= 0 else None
                        for name, values in targets.items()
                    },
                }
                for i, row in enumerate(rows)
            ),
        )
        patient_counts = Counter(
            row["patient_id"] for row in rows if row["split"] == "train"
        )
        write_table(
            output / "patients.csv",
            ["patient_id", "train_bin", "recordings"],
            (
                {
                    "patient_id": patient,
                    "train_bin": assignments[patient],
                    "recordings": patient_counts[patient],
                }
                for patient in sorted(assignments)
            ),
        )
        support = []
        for fraction in config["training"]["fractions"]:
            for split in ("train", "validation", "test"):
                mask = (
                    training_mask(
                        records, fraction, config["training"]["patient_subset"]
                    )
                    if split == "train"
                    else np.array([row["split"] == split for row in records])
                )
                for name, definition in tasks.items():
                    values = targets[name][mask]
                    for label, text in enumerate(definition["classes"]):
                        positives, negatives = (
                            int((values == label).sum()),
                            int(((values >= 0) & (values != label)).sum()),
                        )
                        threshold = config["reliability"][
                            "minimum_training_minority_warning"
                            if split == "train"
                            else "minimum_evaluation_minority_warning"
                        ]
                        support.append(
                            {
                                "task": name,
                                "fraction": fraction,
                                "split": split,
                                "class": label,
                                "class_label": text,
                                "positive": positives,
                                "negative": negatives,
                                "eligible": int((values >= 0).sum()),
                                "excluded": int((values < 0).sum()),
                                "low_support": min(positives, negatives) < threshold,
                            }
                        )
        if support:
            write_table(output / "support.csv", list(support[0]), support)
        else:
            write_table(
                output / "support.csv",
                [
                    "task",
                    "fraction",
                    "split",
                    "class",
                    "class_label",
                    "positive",
                    "negative",
                    "eligible",
                    "excluded",
                    "low_support",
                ],
                [],
            )
        files = [
            "config.resolved.json",
            "tasks.json",
            "balance.json",
            "records.csv",
            "targets.csv",
            "patients.csv",
            "support.csv",
        ]
        metadata.update(
            status="COMPLETE",
            files_sha256={name: file_hash(output / name) for name in files},
            implementation_sha256={
                name: file_hash(Path(__file__).resolve().parents[1] / name)
                for name in [
                    "scripts/prepare_manifest.py",
                    "common/cohorts.py",
                    "integrations/datasets/ptbxl_targets.py",
                    "integrations/datasets/ptbxl.py",
                ]
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
    print(
        f"Manifest: {len(rows)} recordings, {metadata['patients']} patients, {len(tasks)} tasks",
        flush=True,
    )
    return metadata


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiment.json")
    parser.add_argument("--paths", type=Path, default=root / "configs/paths.local.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run(args)
    except (ValueError, KeyError, OSError, TypeError) as error:
        parser.exit(1, f"Manifest failed: {error}\n")


if __name__ == "__main__":
    main()
