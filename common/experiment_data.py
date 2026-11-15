"""Verify saved stages and align model outputs to a shared complete cohort."""

from pathlib import Path
import numpy as np
from common.artifacts import array_info, file_hash, load_ids, sequence_hash
from common.cohorts import validate_records
from common.configuration import read_json, validate_experiment
from common.tables import read_table, unique_index


def verified_stage(path, required):
    path = Path(path).resolve()
    metadata = read_json(path / "metadata.json")
    if metadata.get("status") != "COMPLETE":
        raise ValueError("Input stage is not complete")
    hashes = metadata.get("files_sha256", {})
    if not set(required) <= set(hashes):
        raise ValueError("Input stage is missing required fingerprints")
    for name, expected in hashes.items():
        source = (path / name).resolve()
        if path not in source.parents or file_hash(source) != expected:
            raise ValueError(f"Stage fingerprint mismatch: {name}")
    return metadata


def load_manifest(path):
    path = Path(path).resolve()
    meta = verified_stage(
        path,
        [
            "config.resolved.json",
            "records.csv",
            "targets.csv",
            "tasks.json",
            "patients.csv",
        ],
    )
    bundle = read_json(path / "config.resolved.json")
    validate_experiment(bundle)
    records = read_table(path / "records.csv")
    validate_records(records)
    ids = [row["sample_id"] for row in records]
    if (
        len(records) != meta["records"]
        or sequence_hash(ids) != meta["record_order_sha256"]
    ):
        raise ValueError("Manifest recording order differs from metadata")
    config = bundle["experiment"]
    folds = {
        str(fold): split
        for split, values in config["dataset"]["splits"].items()
        for fold in values
    }
    patient_bins = {}
    for row in records:
        if folds.get(row["fold"]) != row["split"]:
            raise ValueError("Manifest fold differs from configured split")
        if row["split"] == "train":
            value = int(row["train_bin"])
            if not 1 <= value <= config["training"]["patient_subset"]["bins"]:
                raise ValueError("Invalid training bin")
            if patient_bins.setdefault(row["patient_id"], value) != value:
                raise ValueError("A patient crosses training bins")
        elif row["train_bin"]:
            raise ValueError("Held-out recording has a training bin")
    task_rows = read_table(path / "targets.csv")
    if [row["sample_id"] for row in task_rows] != ids:
        raise ValueError("Target table is misaligned")
    definitions = read_json(path / "tasks.json")["tasks"]
    if set(definitions) != set(config["task_ids"]):
        raise ValueError("Manifest tasks differ from configuration")
    targets = {}
    for name, definition in definitions.items():
        values = np.array(
            [int(row[name]) if row[name] else -1 for row in task_rows], dtype=np.int64
        )
        if np.any((values < -1) | (values >= len(definition["classes"]))):
            raise ValueError("Target class is out of range")
        targets[name] = values
    return {
        "path": path,
        "metadata": meta,
        "bundle": bundle,
        "records": records,
        "ids": np.asarray(ids),
        "tasks": definitions,
        "targets": targets,
    }


def load_embeddings(path, manifest):
    path = Path(path).resolve()
    meta = read_json(path / "metadata.json")
    if meta.get("status") != "COMPLETE" or meta.get("representation") != "final":
        raise ValueError("Expected complete final embeddings")
    for name, key in [
        ("embeddings.npy", "embeddings_sha256"),
        ("sample_ids.npy", "ids_file_sha256"),
        ("records.csv", "records_sha256"),
    ]:
        if file_hash(path / name) != meta.get(key):
            raise ValueError(f"Embedding fingerprint mismatch: {name}")
    values = array_info(path / "embeddings.npy", 2)
    ids = load_ids(path / "sample_ids.npy")
    if (
        len(ids) != len(values)
        or list(values.shape) != meta["shape"]
        or sequence_hash(ids) != meta["record_order_sha256"]
    ):
        raise ValueError("Embedding shape or recording order mismatch")
    if set(ids) != set(manifest["ids"]):
        raise ValueError("Embeddings must cover the complete manifest cohort exactly")
    paired = read_table(path / "records.csv")
    if [row["sample_id"] for row in paired] != ids.tolist():
        raise ValueError("Embedding record table is misaligned")
    expected = unique_index(manifest["records"], ["sample_id"])
    for row in paired:
        reference = expected[(row["sample_id"],)]
        if any(row[key] != reference[key] for key in ("patient_id", "fold", "split")):
            raise ValueError("Embedding patient or partition differs from manifest")
    lookup = {value: i for i, value in enumerate(ids)}
    aligned = np.asarray(
        values[[lookup[value] for value in manifest["ids"]]], dtype=np.float32
    )
    if not np.isfinite(aligned).all():
        raise ValueError("Embeddings contain nonfinite values")
    return meta, aligned
