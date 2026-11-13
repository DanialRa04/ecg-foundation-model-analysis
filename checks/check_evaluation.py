"""Generated-cohort checks for labels, manifests, probes, and saved-result analysis."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import numpy as np
from scipy import stats
from common.analysis import run_analysis
from common.artifacts import file_hash, sequence_hash, write_json
from common.associations import (
    adjust_fdr,
    bootstrap_weights,
    overall_percentiles,
    partial_pearson,
    weighted_correlations,
)
from common.cohorts import balanced_patient_bins, training_mask
from common.configuration import load_experiment
from common.evaluation import run_evaluation
from common.experiment_data import load_embeddings, load_manifest
from common.metrics import ranking_metrics
from common.probes import neural_probe
from common.tables import read_table, write_table
from integrations.datasets import ptbxl_targets
from scripts.prepare_manifest import run as prepare_manifest


def reject(function):
    try:
        function()
    except (ValueError, FileExistsError):
        return
    raise AssertionError("Invalid input was accepted")


def fixture(base):
    root = Path(__file__).resolve().parents[1]
    bundle = load_experiment(root / "configs/experiment.json")
    config = bundle["experiment"]
    config["model_ids"] = config["model_ids"][:4]
    config["task_ids"] = ["atrial_fibrillation", "qrs_axis"]
    config["analysis"]["bootstrap_replicates"] = 31
    bundle["probes"]["probes"]["ann"].update(
        hidden_dims=[8, 4], maximum_epochs=3, patience=2, batch_size=32
    )
    dataset = base / "dataset"
    dataset.mkdir()
    rows = []
    for i in range(192):
        split = "train" if i < 128 else "validation" if i < 160 else "test"
        rows.append(
            {
                "ecg_id": str(i + 1),
                "patient_id": str(i // 4 + 1),
                "strat_fold": 1
                if split == "train"
                else 9
                if split == "validation"
                else 10,
                "scp_codes": repr({"AFIB": 0.0} if i % 3 == 0 else {"SR": 100.0}),
                "heart_axis": "AXIS_A" if i % 2 else "AXIS_B",
                "sex": str(i % 2),
                "age": str(20 + i % 70),
                "filename_lr": f"records100/{i + 1}",
                "filename_hr": f"records500/{i + 1}",
            }
        )
    write_table(dataset / "ptbxl_database.csv", list(rows[0]), rows)
    codes = {
        code for task in bundle["tasks"]["tasks"] for code in task.get("codes", [])
    } | {"MI_TEST", "STTC_TEST"}
    write_table(
        dataset / "scp_statements.csv",
        ["code", "diagnostic_class"],
        (
            {
                "code": code,
                "diagnostic_class": "MI"
                if code == "MI_TEST"
                else "STTC"
                if code == "STTC_TEST"
                else "",
            }
            for code in sorted(codes)
        ),
    )
    settings = base / "configs"
    settings.mkdir()
    for name in ("models", "tasks", "descriptors", "probes"):
        write_json(settings / f"{name}.json", bundle[name])
    write_json(settings / "experiment.json", config)
    write_json(settings / "paths.json", {"paths": {"dataset_root": str(dataset)}})
    return bundle, settings, rows


def embedding_fixture(directory, manifest, model, seed):
    directory.mkdir()
    rng = np.random.default_rng(seed)
    y = manifest["targets"]["atrial_fibrillation"]
    axis = manifest["targets"]["qrs_axis"]
    values = rng.normal(size=(len(y), 6)).astype(np.float32)
    values[:, 0] += (2 * y - 1) * 0.6
    values[:, 1] += (2 * axis - 1) * 0.6
    # Held-out values make train-only scaler verification sensitive to leakage.
    held = np.array([row["split"] != "train" for row in manifest["records"]])
    values[held, 2] += 50
    order = rng.permutation(len(values))
    np.save(directory / "embeddings.npy", values[order], allow_pickle=False)
    np.save(directory / "sample_ids.npy", manifest["ids"][order], allow_pickle=False)
    write_table(
        directory / "records.csv",
        ["sample_id", "patient_id", "fold", "split"],
        (
            {
                key: manifest["records"][i][key]
                for key in ("sample_id", "patient_id", "fold", "split")
            }
            for i in order
        ),
    )
    write_json(
        directory / "metadata.json",
        {
            "status": "COMPLETE",
            "model": model,
            "representation": "final",
            "records": len(values),
            "shape": list(values.shape),
            "record_order_sha256": sequence_hash(manifest["ids"][order]),
            "embeddings_sha256": file_hash(directory / "embeddings.npy"),
            "ids_file_sha256": file_hash(directory / "sample_ids.npy"),
            "records_sha256": file_hash(directory / "records.csv"),
        },
    )
    return values


def run_checks():
    import torch

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    (root / "outputs").mkdir(exist_ok=True)
    passed = []
    assert ptbxl_targets.code_keys("{'AFIB': 0.0}") == {"AFIB"}
    reject(lambda: ptbxl_targets.code_keys("__import__('os').getcwd()"))
    reject(lambda: ptbxl_targets.code_keys("['AFIB']"))
    passed.append("literal-only SCP parsing and zero-confidence key membership")
    report = ranking_metrics(
        [0, 1, 0, 1], [[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]], 2, True
    )
    assert report["average_precision"] == 1 and report["auroc"] == 1
    assert (
        ranking_metrics([0, 0], [[0.8, 0.2], [0.7, 0.3]], 2, True)["average_precision"]
        is None
    )
    report = ranking_metrics([0, 1], [[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]], 3, False)
    assert report["included_classes"] == [0, 1]
    passed.append("analytical binary and evaluable-class multiclass ranking metrics")
    x = np.array([1.0, 2.0, 2.0, 4.0, 5.0])
    y = np.array([5.0, 2.0, 3.0, 2.0, 1.0])
    weights = np.array([[1, 1, 1, 1, 1], [2, 0, 1, 3, 0], [0, 2, 1, 0, 3]], dtype=float)
    for method, function in [
        ("spearman", stats.spearmanr),
        ("pearson", stats.pearsonr),
        ("kendall", stats.kendalltau),
    ]:
        actual = weighted_correlations(x, y, weights, method)
        for i, w in enumerate(weights.astype(int)):
            expanded = np.repeat(np.arange(len(x)), w)
            assert np.isclose(actual[i], function(x[expanded], y[expanded]).statistic)
    grouped = bootstrap_weights(["A", "A", "B", "C"], 50, 42)
    assert np.array_equal(grouped[:, 0], grouped[:, 1])
    assert np.array_equal(grouped, bootstrap_weights(["A", "A", "B", "C"], 50, 42))
    assert np.isnan(
        weighted_correlations(x, y, np.array([[3, 0, 0, 0, 0.0]]), "pearson")[0]
    )
    assert partial_pearson(
        np.arange(8.0), np.arange(8.0) ** 2, np.arange(8.0)[:, None]
    ) == (None, None)
    passed.append(
        "group resampling and exact weighted Pearson Spearman Kendall coefficients"
    )
    fdr = [{"section": "a", "p_value": value} for value in [0.01, 0.04, 0.03, None]] + [
        {"section": "b", "p_value": 0.04}
    ]
    adjust_fdr(fdr, ["section"])
    assert (
        np.allclose([row["q_value"] for row in fdr[:3]], [0.03, 0.04, 0.04])
        and fdr[3]["q_value"] is None
        and fdr[4]["q_value"] == 0.04
    )
    overall, complete = overall_percentiles(
        {"x": {"A": 0.1, "B": 0.2, "C": None}, "y": {"A": 0.9, "B": 0.7, "C": 0.5}},
        ["A", "B", "C"],
        ["x", "y"],
    )
    assert complete == ["A", "B"] and overall == {"A": 0.5, "B": 0.5}
    passed.append("family-specific BH correction and common-model task percentiles")
    with TemporaryDirectory(prefix="evaluation_checks_", dir=root / "outputs") as temp:
        base = Path(temp)
        bundle, settings, raw = fixture(base)
        args = argparse.Namespace(
            config=settings / "experiment.json",
            paths=settings / "paths.json",
            output=base / "manifest",
        )
        prepare_manifest(args)
        manifest = load_manifest(args.output)
        assert manifest["tasks"]["qrs_axis"]["classes"] == ["AXIS_A", "AXIS_B"]
        rows = [
            {
                **row,
                "sample_id": row["ecg_id"],
                "split": "train"
                if int(row["strat_fold"]) == 1
                else "validation"
                if int(row["strat_fold"]) == 9
                else "test",
            }
            for row in raw
        ]
        altered = deepcopy(rows)
        for row in altered:
            if row["split"] != "train":
                row.update(heart_axis="HELD_OUT_CLASS", sex="new", age="999")
        catalog = bundle["tasks"]
        names = bundle["experiment"]["task_ids"]
        taxonomy = base / "dataset/scp_statements.csv"
        targets, tasks = ptbxl_targets.build_targets(rows, catalog, names, taxonomy)
        alt_targets, alt_tasks = ptbxl_targets.build_targets(
            altered, catalog, names, taxonomy
        )
        assert tasks == alt_tasks
        all_targets, all_tasks = ptbxl_targets.build_targets(
            rows, catalog, [task["id"] for task in catalog["tasks"]], taxonomy
        )
        assert len(all_targets) == 13 and all_targets["atrial_fibrillation"][0] == 1
        changed = deepcopy(rows)
        changed[0]["scp_codes"] = "{'MI_TEST': 0, 'STTC_TEST': 0}"
        mapped, _ = ptbxl_targets.build_targets(
            changed, catalog, ["myocardial_infarction", "st_t_abnormality"], taxonomy
        )
        assert mapped["myocardial_infarction"][0] == mapped["st_t_abnormality"][0] == 1
        passed.append("all configured targets and taxonomy-resolved MI and ST-T unions")
        subset = bundle["experiment"]["training"]["patient_subset"]
        assigned, _ = balanced_patient_bins(rows, targets, tasks, subset, 42)
        reverse, _ = balanced_patient_bins(
            rows[::-1],
            {key: value[::-1] for key, value in targets.items()},
            tasks,
            subset,
            42,
        )
        heldout_changed, _ = balanced_patient_bins(
            altered, alt_targets, alt_tasks, subset, 42
        )
        assert assigned == reverse == heldout_changed
        half = training_mask(manifest["records"], 0.5, subset)
        full = training_mask(manifest["records"], 1.0, subset)
        assert np.all(~half | full) and half.any() and full.sum() == 128
        passed.append(
            "training-only class retention and order-invariant whole-patient subsets"
        )
        embedding_paths = []
        for i, model in enumerate(bundle["experiment"]["model_ids"]):
            path = base / f"embedding{i}"
            expected = embedding_fixture(path, manifest, model, 100 + i)
            _, aligned = load_embeddings(path, manifest)
            assert np.array_equal(aligned, expected)
            embedding_paths.append(path)
        passed.append(
            "record-ID alignment across independently permuted complete embeddings"
        )
        # Changing test features cannot change validation-selected parameters.
        rng = np.random.default_rng(5)
        train = rng.normal(size=(32, 4)).astype(np.float32)
        labels = np.arange(32) % 2
        evaluate = {"validation": train[:8], "test": train[8:16]}
        settings_ann = bundle["probes"]["probes"]["ann"]
        predictions_a, details_a = neural_probe(
            train, labels, labels[:8], evaluate, settings_ann, 42, 2, True
        )
        changed_evaluate = {**evaluate, "test": evaluate["test"] + 100}
        predictions_b, details_b = neural_probe(
            train, labels, labels[:8], changed_evaluate, settings_ann, 42, 2, True
        )
        assert details_a == details_b and np.array_equal(
            predictions_a["validation"], predictions_b["validation"]
        )
        passed.append(
            "seeded neural fitting and checkpoint selection are independent of test features"
        )
        evaluation = base / "evaluation"
        result = run_evaluation(args.output, embedding_paths, evaluation)
        assert result["status"] == "COMPLETE" and result["fits"] == 64
        scaler = json.loads((evaluation / "scaler_001.json").read_text())
        _, first = load_embeddings(embedding_paths[0], manifest)
        assert (
            np.allclose(scaler["mean"], first[half].mean(0), atol=1e-6)
            and scaler["mean"][2] < 10
        )
        fit = json.loads((evaluation / "fits/fit_00001/fit.json").read_text())
        if "history" in fit:
            assert fit["best_validation_average_precision"] == max(
                row["validation_average_precision"] for row in fit["history"]
            )
        summary = read_table(evaluation / "seed_summary.csv")
        assert all(row["valid_seeds"] == "2" for row in summary)
        passed.append(
            "end-to-end binary and multiclass probes training-only scaling and seed summaries"
        )
        analysis = run_analysis(evaluation, base / "analysis")
        assert analysis["status"] == "COMPLETE"
        associations = read_table(base / "analysis/associations.csv")
        assert {
            "primary",
            "additional",
            "group_aggregated",
            "leave_one_model_out",
            "task_agnostic",
            "input_contract_adjusted",
        } <= {row["section"] for row in associations}
        assert any(
            row["cohort"] == "documented_unexposed" and row["status"] == "UNDEFINED"
            for row in associations
        )
        reject(lambda: run_evaluation(args.output, embedding_paths, evaluation))
        reject(lambda: run_analysis(evaluation, base / "analysis"))
        passed.append(
            "configured association families sensitivity outputs and collision protection"
        )
        # Keep binary labels constant to exercise explicit skipped-fit outputs.
        skip_config = deepcopy(bundle["experiment"])
        skip_config["task_ids"] = ["st_elevation"]
        skip_config["model_ids"] = skip_config["model_ids"][:1]
        write_json(settings / "skip.json", skip_config)
        skip_args = argparse.Namespace(
            config=settings / "skip.json",
            paths=settings / "paths.json",
            output=base / "skip_manifest",
        )
        prepare_manifest(skip_args)
        skipped = run_evaluation(
            skip_args.output, embedding_paths[:1], base / "skipped_evaluation"
        )
        assert skipped["skipped_fits"] == skipped["fits"] == 8
        assert all(
            row["mean"] == ""
            for row in read_table(base / "skipped_evaluation/seed_summary.csv")
        )
        passed.append(
            "missing training classes produce explicit skipped fits and missing metrics"
        )
        # A complete hash set does not excuse an incomplete statistical join.
        table = read_table(evaluation / "seed_summary.csv")
        original = (evaluation / "seed_summary.csv").read_bytes()
        original_meta = json.loads((evaluation / "metadata.json").read_text())
        (evaluation / "seed_summary.csv").unlink()
        write_table(evaluation / "seed_summary.csv", list(table[0]), table[1:])
        altered_meta = deepcopy(original_meta)
        altered_meta["files_sha256"]["seed_summary.csv"] = file_hash(
            evaluation / "seed_summary.csv"
        )
        write_json(evaluation / "metadata.json", altered_meta)
        reject(lambda: run_analysis(evaluation, base / "incomplete_join"))
        (evaluation / "seed_summary.csv").write_bytes(original)
        write_json(evaluation / "metadata.json", original_meta)
        passed.append(
            "incomplete model-task statistical joins are rejected independently of hashes"
        )
        # Deliberate tampering is rejected before any fitting or association.
        with (embedding_paths[0] / "records.csv").open("a") as stream:
            stream.write("\n")
        reject(lambda: load_embeddings(embedding_paths[0], manifest))
        with (evaluation / "seed_summary.csv").open("a") as stream:
            stream.write("\n")
        reject(lambda: run_analysis(evaluation, base / "tampered_analysis"))
        passed.append("paired-input and result fingerprint enforcement")
    return {"status": "PASS", "checks": passed}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
