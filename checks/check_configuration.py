"""Exercise configuration checks for experiment-integrity errors."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from common.configuration import load_experiment, read_json, summarize_experiment, validate_experiment


def run_checks():
    root = Path(__file__).resolve().parents[1]
    base = load_experiment(root / "configs/experiment.json")
    passed = ["configured experiment"]

    def rejects(name, change):
        candidate = deepcopy(base)
        change(candidate)
        try:
            validate_experiment(candidate)
        except (ValueError, KeyError, TypeError):
            passed.append(name)
        else:
            raise AssertionError(f"Accepted invalid configuration: {name}")

    rejects("overlapping splits", lambda b: b["experiment"]["dataset"]["splits"]["test"].append(b["experiment"]["dataset"]["splits"]["train"][0]))
    rejects("duplicate seeds", lambda b: b["experiment"]["training"]["probe_seeds"].append(b["experiment"]["training"]["probe_seeds"][0]))
    rejects("out-of-range fraction", lambda b: b["experiment"]["training"]["fractions"].append(1.5))
    rejects("unknown selected model", lambda b: b["experiment"]["model_ids"].append("absent-model"))
    rejects("duplicate task", lambda b: b["tasks"]["tasks"].append(deepcopy(b["tasks"]["tasks"][0])))
    rejects("invalid shape", lambda b: b["models"]["models"][0].update(shape=[0, 100]))
    rejects("unreferenced exposure assertion", lambda b: b["models"]["models"][0].update(evaluation_dataset_exposure=False, exposure_reference=None))
    rejects("invalid dropout", lambda b: b["probes"]["probes"]["ann"].update(dropout=1.0))
    rejects("empty resampling", lambda b: b["experiment"]["analysis"].update(bootstrap_replicates=0))
    rejects("duplicate descriptor", lambda b: b["descriptors"]["panel"].append(deepcopy(b["descriptors"]["panel"][0])))
    rejects("wrong fraction-bin mapping", lambda b: b["experiment"]["training"]["patient_subset"]["fraction_bins"].update({"0.5": [1, 2]}))

    def nonnested(candidate):
        candidate["experiment"]["training"]["fractions"] = [0.25, 0.5, 1.0]
        candidate["experiment"]["training"]["patient_subset"].update(bins=4, fraction_bins={"0.25": [1], "0.5": [2, 3], "1.0": [1, 2, 3, 4]})
    rejects("nonnested patient subsets", nonnested)

    small = deepcopy(base)
    small["experiment"]["model_ids"] = small["experiment"]["model_ids"][:2]
    small["experiment"]["task_ids"] = small["experiment"]["task_ids"][:2]
    validate_experiment(small)
    expected = 2 * 2 * len(small["experiment"]["training"]["fractions"]) * len(small["experiment"]["training"]["probe_seeds"])
    if not all(value == expected for value in summarize_experiment(small)["planned_probe_fits"].values()):
        raise AssertionError("Workload depends on fixed model/task counts")
    passed.append("different model and task counts")

    with TemporaryDirectory(prefix="ecg_config_checks_") as directory:
        path = Path(directory) / "config.json"
        for name, text in (("duplicate JSON keys", '{"value":1,"value":2}'), ("nonstandard JSON", '{"value":NaN}'), ("overflowing JSON number", '{"value":1e999}')):
            path.write_text(text, encoding="utf-8")
            try:
                read_json(path)
            except ValueError:
                passed.append(name)
            else:
                raise AssertionError(f"Accepted {name}")
    return {"status": "PASS", "checks": passed}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2, allow_nan=False))
