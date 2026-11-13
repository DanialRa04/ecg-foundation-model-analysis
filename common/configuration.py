"""Read and validate experiment settings without loading data or models."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Nonstandard JSON number: {value}")


def _finite_float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"JSON number exceeds finite precision: {value}")
    return number


def read_json(path: Path) -> dict:
    value = json.loads(Path(path).read_text(encoding="utf-8-sig"), object_pairs_hook=_unique_object, parse_constant=_invalid_constant, parse_float=_finite_float)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, *, positive=False):
    valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    return valid and (value > 0 if positive else value >= 0)


def _integer(value, *, minimum=1):
    return isinstance(value, int) and not isinstance(value, bool) and value >= minimum


def _unique(values, label):
    _require(isinstance(values, list) and bool(values), f"{label} must be a nonempty list")
    _require(all(isinstance(value, (str, int, float)) and not isinstance(value, bool) for value in values), f"{label} has an invalid item")
    _require(len(set(values)) == len(values), f"{label} contains duplicates")
    return values


def _index(rows, key, label):
    _require(isinstance(rows, list) and all(isinstance(row, dict) for row in rows), f"{label} must contain objects")
    names = [row.get(key) for row in rows]
    _require(all(isinstance(name, str) and name.strip() for name in names), f"{label} needs nonempty {key} values")
    _unique(names, label)
    return dict(zip(names, rows))


def load_experiment(path: Path) -> dict:
    path = Path(path).resolve()
    experiment = read_json(path)
    resources = experiment.get("files", {})
    result = {"experiment": experiment}
    for name in ("models", "tasks", "descriptors", "probes"):
        relative = resources.get(name)
        _require(isinstance(relative, str) and relative.strip(), f"files.{name} must name a JSON file")
        result[name] = read_json(path.parent / relative)
    validate_experiment(result)
    return result


def validate_experiment(bundle: dict) -> None:
    config = bundle["experiment"]
    dataset = config["dataset"]
    for field in ("name", "record_column", "patient_column", "fold_column"):
        _require(isinstance(dataset.get(field), str) and bool(dataset[field].strip()), f"dataset.{field} is required")
    splits = dataset["splits"]
    folds = [_unique(splits[name], f"split {name}") for name in ("train", "validation", "test")]
    _require(all(not set(left) & set(right) for i, left in enumerate(folds) for right in folds[i + 1:]), "Dataset split folds overlap")
    training = config["training"]
    fractions = _unique(training["fractions"], "training fractions")
    _require(all(_number(value, positive=True) and value <= 1 for value in fractions), "Training fractions must be in (0, 1]")
    _require(fractions == sorted(fractions), "Training fractions must be sorted")
    _require(training["primary_fraction"] in fractions, "Primary fraction is not configured")
    seeds = _unique(training["probe_seeds"], "probe seeds")
    _require(all(_integer(value, minimum=0) for value in seeds), "Probe seeds must be nonnegative integers")
    _require(_integer(training["manifest_seed"], minimum=0), "Manifest seed must be a nonnegative integer")
    subset = training["patient_subset"]
    _require(_integer(subset["bins"]), "Patient-bin count must be a positive integer")
    fraction_bins = subset["fraction_bins"]
    for fraction in fractions:
        selected = _unique(fraction_bins[str(float(fraction))], f"bins for {fraction}")
        _require(all(_integer(value) and value <= subset["bins"] for value in selected), "Patient bin is outside the configured range")
        _require(math.isclose(len(selected) / subset["bins"], fraction), "Patient-bin selection does not match its training fraction")
    for low, high in zip(fractions, fractions[1:]):
        _require(set(fraction_bins[str(float(low))]) <= set(fraction_bins[str(float(high))]), "Training fractions are not nested")

    models = _index(bundle["models"]["models"], "name", "models")
    model_ids = _unique(config["model_ids"], "selected models")
    _require(all(name in models for name in model_ids), "A selected model is absent from the catalog")
    for name in model_ids:
        model = models[name]
        _require(_number(model["sampling_hz"], positive=True), f"Invalid sampling rate: {name}")
        _require(isinstance(model["shape"], list) and len(model["shape"]) == 2 and all(_integer(value) for value in model["shape"]), f"Invalid input shape: {name}")
        _require(isinstance(model.get("analysis_group"), str) and bool(model["analysis_group"].strip()), f"Missing analysis group: {name}")
        exposure = model.get("evaluation_dataset_exposure")
        _require(exposure is None or isinstance(exposure, bool), f"Invalid exposure value: {name}")
        if exposure is not None:
            _require(isinstance(model.get("exposure_reference"), str) and bool(model["exposure_reference"].strip()), f"Exposure needs a reference: {name}")

    tasks = _index(bundle["tasks"]["tasks"], "id", "tasks")
    task_ids = _unique(config["task_ids"], "selected tasks")
    _require(all(name in tasks for name in task_ids), "A selected task is absent from the catalog")
    for name in task_ids:
        task = tasks[name]
        kind = task["target_type"]
        _require(kind in {"binary_codes", "binary_taxonomy", "binary_union", "multiclass_column"}, f"Unknown target type: {kind}")
        if kind in {"binary_codes", "binary_union"}:
            _unique(task["codes"], f"codes for {name}")
        if kind in {"binary_taxonomy", "binary_union"}:
            _require(isinstance(task["taxonomy_column"], str) and bool(task["taxonomy_column"]), f"Missing taxonomy column: {name}")
            _unique(task["taxonomy_values"], f"taxonomy values for {name}")
        if kind == "multiclass_column":
            _require(isinstance(task["column"], str) and bool(task["column"]), f"Missing label column: {name}")
            _require(_integer(task["minimum_full_training_support"]), f"Invalid class support: {name}")

    descriptors = bundle["descriptors"]
    _index(descriptors["panel"], "id", "descriptors")
    for row in descriptors["panel"]:
        _require(row["matrix"] in {"centered", "uncentered"}, f"Unknown matrix convention: {row['id']}")
    _require(_number(descriptors["parameters"]["variance_tolerance"]), "Invalid variance tolerance")
    _require(_number(descriptors["parameters"]["rankme_epsilon"], positive=True), "Invalid RankMe epsilon")
    evaluation = config["evaluation"]
    probe_ids = _unique(evaluation["probe_ids"], "selected probes")
    _require(evaluation["primary_probe"] in probe_ids, "Primary probe is not selected")
    probes = bundle["probes"]["probes"]
    _require(all(name in probes for name in probe_ids), "A selected probe has no settings")
    for name in probe_ids:
        probe = probes[name]
        if "hidden_dims" in probe:
            _require(isinstance(probe["hidden_dims"], list) and bool(probe["hidden_dims"]) and all(_integer(value) for value in probe["hidden_dims"]), "Invalid hidden widths")
            _require(_number(probe["dropout"]) and probe["dropout"] < 1, "Dropout must be in [0, 1)")
            for key in ("learning_rate",):
                _require(_number(probe[key], positive=True), f"Invalid probe {key}")
            for key in ("maximum_epochs", "patience", "batch_size"):
                _require(_integer(probe[key]), f"Invalid probe {key}")
        if "C" in probe:
            _require(_number(probe["C"], positive=True), "Linear C must be positive")
            _require(_integer(probe["maximum_iterations"]), "Invalid linear iteration limit")
    analysis = config["analysis"]
    methods = [analysis["primary_method"], *analysis["secondary_methods"]]
    _unique(methods, "association methods")
    _require(all(method in {"spearman", "pearson", "kendall"} for method in methods), "Unknown correlation method")
    _require(_integer(analysis["bootstrap_replicates"]), "Bootstrap replicates must be positive")
    _require(_integer(analysis["bootstrap_seed"], minimum=0), "Bootstrap seed must be a nonnegative integer")
    _require(_number(analysis["confidence_level"], positive=True) and analysis["confidence_level"] < 1, "Confidence level must be in (0, 1)")
    _unique(analysis["fdr_family"], "FDR family keys")


def summarize_experiment(bundle: dict) -> dict:
    config = bundle["experiment"]
    models = {model["name"]: model for model in bundle["models"]["models"]}
    count = len(config["model_ids"]) * len(config["task_ids"]) * len(config["training"]["fractions"]) * len(config["training"]["probe_seeds"])
    return {"status": "VALID", "models": len(config["model_ids"]), "tasks": len(config["task_ids"]),
            "descriptors": len(bundle["descriptors"]["panel"]),
            "analysis_groups": len({models[name]["analysis_group"] for name in config["model_ids"]}),
            "planned_probe_fits": {name: count for name in config["evaluation"]["probe_ids"]},
            "scope": "configuration consistency; assets are checked by their execution stages"}
