"""Join saved model-level results and produce configured association tables."""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common.artifacts import file_hash, log_event, new_run, versions, write_json
from common.associations import (
    adjust_fdr,
    association,
    overall_percentiles,
    partial_bootstrap,
    partial_pearson,
)
from common.configuration import read_json, validate_experiment
from common.experiment_data import verified_stage
from common.tables import read_table, unique_index, write_table


def _number(value):
    if value in ("", None):
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def run_analysis(evaluation_path, output_path):
    source = Path(evaluation_path).resolve()
    input_meta = verified_stage(
        source,
        [
            "config.resolved.json",
            "descriptors.csv",
            "seed_summary.csv",
            "representations.csv",
        ],
    )
    bundle = read_json(source / "config.resolved.json")
    validate_experiment(bundle)
    config = bundle["experiment"]
    settings = config["analysis"]
    policies = {
        "interval": "percentile_model_group_bootstrap",
        "p_values": "two_sided_standard_correlation_tests",
        "p_value_unit": "checkpoint_rows",
        "fdr_method": "Benjamini-Hochberg",
        "task_agnostic_summary": "mean_within_task_percentile",
    }
    if any(settings[key] != value for key, value in policies.items()):
        raise ValueError("Unsupported analysis policy")
    sensitivities = set(settings["sensitivities"])
    supported = {
        "documented_unexposed_only",
        "leave_one_model_out",
        "analysis_group_aggregation",
        "exact_duplicate_representations",
        "input_contract_adjustment",
    }
    if sensitivities - supported:
        raise ValueError("Unknown analysis sensitivity")
    models = sorted(input_meta["models"])
    if (
        not models
        or len(models) != len(set(models))
        or not set(models) <= set(config["model_ids"])
    ):
        raise ValueError("Invalid evaluation model set")
    catalog = {row["name"]: row for row in bundle["models"]["models"]}
    descriptors = read_table(source / "descriptors.csv")
    performance = read_table(source / "seed_summary.csv")
    identities = read_table(source / "representations.csv")
    dindex = unique_index(descriptors, ["model", "layer", "fraction"])
    pindex = unique_index(
        performance, ["model", "layer", "fraction", "task", "probe", "split", "metric"]
    )
    iindex = unique_index(identities, ["model"])
    if set(iindex) != {(model,) for model in models}:
        raise ValueError("Representation inventory differs from selected models")
    expected_d = {
        (model, "final", str(fraction))
        for model in models
        for fraction in config["training"]["fractions"]
    }
    if set(dindex) != expected_d:
        raise ValueError("Descriptor join is incomplete or contains extra rows")
    expected_p = {
        (model, "final", str(fraction), task, probe, split, metric)
        for model in models
        for fraction in config["training"]["fractions"]
        for task in config["task_ids"]
        for probe in config["evaluation"]["probe_ids"]
        for split in ("validation", "test")
        for metric in ("average_precision", "auroc")
    }
    if set(pindex) != expected_p:
        raise ValueError("Performance join is incomplete or contains extra rows")
    for fraction in config["training"]["fractions"]:
        matched = [dindex[(model, "final", str(fraction))] for model in models]
        if len({(row["records"], row["record_order_sha256"]) for row in matched}) != 1:
            raise ValueError("Models use different descriptor cohorts")
    output = new_run(output_path, [source])
    meta = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evaluation_metadata_sha256": file_hash(source / "metadata.json"),
        "models": models,
        "versions": versions(["numpy", "scipy"]),
        "p_value_note": "standard checkpoint-row tests; group bootstrap does not change their independence assumption; small-model-count approximations",
    }
    write_json(output / "metadata.json", meta)
    log_event(output, "START model-level analysis")
    write_json(output / "config.resolved.json", bundle)
    rows, overall_rows, duplicate_rows = [], [], []
    methods = [settings["primary_method"], *settings["secondary_methods"]]
    groups = {model: catalog[model]["analysis_group"] for model in models}
    cohorts = {"all": models}
    if "documented_unexposed_only" in sensitivities:
        cohorts["documented_unexposed"] = [
            model
            for model in models
            if catalog[model]["evaluation_dataset_exposure"] is False
            and catalog[model]["exposure_reference"]
        ]

    def add_panel(outcome, fraction, task, probe, metric, section_override=None):
        for cohort, selected in cohorts.items():
            for descriptor in bundle["descriptors"]["panel"]:
                name = descriptor["id"]
                x = [
                    _number(dindex[(model, "final", str(fraction))][name])
                    for model in selected
                ]
                y = [outcome[model] for model in selected]
                missing = any(value is None for value in [*x, *y])
                for method in methods:
                    primary = (
                        fraction == config["training"]["primary_fraction"]
                        and probe == config["evaluation"]["primary_probe"]
                        and metric == config["evaluation"]["primary_metric"]
                        and method == settings["primary_method"]
                    )
                    identity = {
                        "section": section_override
                        or ("primary" if primary else "additional"),
                        "cohort": cohort,
                        "probe": probe,
                        "metric": metric,
                        "method": method,
                        "task": task,
                        "descriptor": name,
                        "fraction": fraction,
                        "omitted_model": "",
                        "controls": "",
                    }
                    result = association(
                        x if not missing else [],
                        y if not missing else [],
                        [groups[model] for model in selected] if not missing else [],
                        method,
                        settings,
                    )
                    if missing:
                        result.update(
                            models=len(selected),
                            independent_groups=len(
                                {groups[model] for model in selected}
                            ),
                            reason="undefined descriptor or model performance; no silent model deletion",
                        )
                    if not selected:
                        result["reason"] = (
                            "no checkpoints with documented unexposed status"
                        )
                    rows.append({**identity, **result})
                    if cohort != "all" or section_override or missing:
                        continue
                    unique_groups = sorted({groups[model] for model in selected})
                    grouped_x = [
                        float(
                            np.mean(
                                [
                                    x[i]
                                    for i, model in enumerate(selected)
                                    if groups[model] == group
                                ]
                            )
                        )
                        for group in unique_groups
                    ]
                    grouped_y = [
                        float(
                            np.mean(
                                [
                                    y[i]
                                    for i, model in enumerate(selected)
                                    if groups[model] == group
                                ]
                            )
                        )
                        for group in unique_groups
                    ]
                    if "analysis_group_aggregation" in sensitivities:
                        rows.append(
                            {
                                **identity,
                                "section": "group_aggregated",
                                **association(
                                    grouped_x,
                                    grouped_y,
                                    unique_groups,
                                    method,
                                    settings,
                                ),
                            }
                        )
                    for omitted in (
                        selected if "leave_one_model_out" in sensitivities else []
                    ):
                        keep = [
                            i for i, model in enumerate(selected) if model != omitted
                        ]
                        result = association(
                            np.asarray(x)[keep],
                            np.asarray(y)[keep],
                            [groups[selected[i]] for i in keep],
                            method,
                            settings,
                            bootstrap=False,
                        )
                        rows.append(
                            {
                                **identity,
                                "section": "leave_one_model_out",
                                "omitted_model": omitted,
                                **result,
                            }
                        )
                # A specified Pearson partial association, with refitted controls per bootstrap draw.
                if (
                    cohort == "all"
                    and not section_override
                    and not missing
                    and "pearson" in methods
                    and "input_contract_adjustment" in sensitivities
                ):
                    controls = np.asarray(
                        [
                            [
                                np.log(
                                    float(
                                        dindex[(model, "final", str(fraction))][
                                            "features"
                                        ]
                                    )
                                ),
                                catalog[model]["shape"][0],
                                np.log(catalog[model]["sampling_hz"]),
                                catalog[model]["shape"][1]
                                / catalog[model]["sampling_hz"],
                            ]
                            for model in selected
                        ]
                    )
                    names = np.asarray(
                        ["log_features", "leads", "log_rate", "duration"]
                    )
                    nonconstant = np.ptp(controls, axis=0) > 1e-12
                    controls = controls[:, nonconstant]
                    estimate, p = partial_pearson(x, y, controls)
                    result = {
                        "estimate": estimate,
                        "p_value": p,
                        "ci_low": None,
                        "ci_high": None,
                        "models": len(selected),
                        "independent_groups": len(set(groups.values())),
                        "bootstrap_requested": settings["bootstrap_replicates"],
                        "bootstrap_valid": 0,
                        "status": "OK" if estimate is not None else "UNDEFINED",
                        "reason": ""
                        if estimate is not None
                        else "insufficient residual degrees, constant residuals, or collinear input controls",
                    }
                    if estimate is not None and result["independent_groups"] >= 3:
                        (
                            result["ci_low"],
                            result["ci_high"],
                            result["bootstrap_valid"],
                        ) = partial_bootstrap(
                            x,
                            y,
                            controls,
                            [groups[model] for model in selected],
                            settings,
                        )
                    rows.append(
                        {
                            **identity,
                            "section": "input_contract_adjusted",
                            "method": "pearson",
                            "controls": ",".join(names[nonconstant]),
                            **result,
                        }
                    )

    try:
        duplicate_map = defaultdict(list)
        for row in identities:
            duplicate_map[(row["features"], row["aligned_embeddings_sha256"])].append(
                row["model"]
            )
        for (_, digest), members in duplicate_map.items():
            if len(members) > 1 and "exact_duplicate_representations" in sensitivities:
                duplicate_rows.append(
                    {
                        "models": ",".join(sorted(members)),
                        "aligned_embeddings_sha256": digest,
                    }
                )
        for fraction in config["training"]["fractions"]:
            for probe in config["evaluation"]["probe_ids"]:
                task_outcomes = {}
                for metric in ("average_precision", "auroc"):
                    for task in config["task_ids"]:
                        outcome = {
                            model: _number(
                                pindex[
                                    (
                                        model,
                                        "final",
                                        str(fraction),
                                        task,
                                        probe,
                                        "test",
                                        metric,
                                    )
                                ]["mean"]
                            )
                            for model in models
                        }
                        if metric == config["evaluation"]["primary_metric"]:
                            task_outcomes[task] = outcome
                        add_panel(outcome, fraction, task, probe, metric)
                summary, complete = overall_percentiles(
                    task_outcomes, models, config["task_ids"]
                )
                for model in models:
                    overall_rows.append(
                        {
                            "model": model,
                            "fraction": fraction,
                            "probe": probe,
                            "mean_task_percentile": summary.get(model),
                            "tasks": len(config["task_ids"]),
                            "common_models": len(complete),
                            "included": model in complete,
                        }
                    )
                if summary:
                    # The task-agnostic family uses the same complete model set for every task.
                    original = cohorts.copy()
                    cohorts["all"] = complete
                    if "documented_unexposed" in original:
                        cohorts["documented_unexposed"] = [
                            model
                            for model in original["documented_unexposed"]
                            if model in complete
                        ]
                    add_panel(
                        summary,
                        fraction,
                        "overall_percentile",
                        probe,
                        "mean_task_percentile",
                        "task_agnostic",
                    )
                    cohorts.update(original)
                print(f"Analysis: fraction={fraction}, probe={probe}", flush=True)
        adjust_fdr(rows, settings["fdr_family"])
        write_table(output / "associations.csv", list(rows[0]), rows)
        write_table(output / "overall_ranks.csv", list(overall_rows[0]), overall_rows)
        write_table(
            output / "duplicate_representations.csv",
            ["models", "aligned_embeddings_sha256"],
            duplicate_rows,
        )
        files = [
            "config.resolved.json",
            "associations.csv",
            "overall_ranks.csv",
            "duplicate_representations.csv",
        ]
        implementation = [
            "common/analysis.py",
            "common/associations.py",
            "common/experiment_data.py",
            "common/tables.py",
        ]
        meta.update(
            status="COMPLETE",
            association_rows=len(rows),
            exact_duplicate_groups=len(duplicate_rows),
            unknown_exposure_models=[
                model
                for model in models
                if catalog[model]["evaluation_dataset_exposure"] is None
            ],
            files_sha256={name: file_hash(output / name) for name in files},
            implementation_sha256={
                name: file_hash(Path(__file__).resolve().parents[1] / name)
                for name in implementation
            },
        )
    except Exception as error:
        meta.update(status="FAILED", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        write_json(output / "metadata.json", meta)
        log_event(output, meta["status"])
    return meta
