"""Render verified aggregate results as tables, scientific figures, and an index."""

from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common.artifacts import file_hash, log_event, new_run, versions, write_json
from common.configuration import read_json, validate_experiment
from common.experiment_data import verified_stage
from common.plotting import (
    forest_figure,
    group_styles,
    heatmap_figure,
    plot_style,
    ranking_figure,
    save_figure,
    scatter_figure,
)
from common.report_tables import (
    boolean,
    chunks,
    display_number,
    display_performance,
    display_probability,
    label,
    load_settings,
    markdown_cell,
    markdown_table,
    number,
)
from common.tables import read_table, unique_index, write_table

ASSOCIATION_COLUMNS = [
    "section",
    "cohort",
    "probe",
    "metric",
    "method",
    "task",
    "descriptor",
    "fraction",
    "omitted_model",
    "controls",
    "estimate",
    "p_value",
    "ci_low",
    "ci_high",
    "models",
    "independent_groups",
    "bootstrap_requested",
    "bootstrap_valid",
    "status",
    "reason",
    "q_value",
    "fdr_tests",
]
PERFORMANCE_COLUMNS = [
    "model",
    "analysis_group",
    "task",
    "probe",
    "fraction",
    "metric",
    "mean",
    "sample_std",
    "valid_seeds",
    "planned_seeds",
    "low_support",
    "status",
    "reason",
]


def load_report_data(evaluation_path, analysis_path, settings_path):
    evaluation_path, analysis_path = (
        Path(evaluation_path).resolve(),
        Path(analysis_path).resolve(),
    )
    evaluation_meta = verified_stage(
        evaluation_path,
        ["config.resolved.json", "seed_summary.csv", "metrics.csv", "descriptors.csv"],
    )
    analysis_meta = verified_stage(
        analysis_path, ["config.resolved.json", "associations.csv", "overall_ranks.csv"]
    )
    if analysis_meta["evaluation_metadata_sha256"] != file_hash(
        evaluation_path / "metadata.json"
    ):
        raise ValueError("Analysis belongs to a different evaluation run")
    bundle = read_json(evaluation_path / "config.resolved.json")
    validate_experiment(bundle)
    if bundle != read_json(analysis_path / "config.resolved.json") or set(
        evaluation_meta["models"]
    ) != set(analysis_meta["models"]):
        raise ValueError("Evaluation and analysis configurations or models differ")
    config = bundle["experiment"]
    settings = load_settings(settings_path, bundle)
    selection = settings["selection"]
    model_set = set(evaluation_meta["models"])
    if (
        not model_set
        or len(model_set) != len(evaluation_meta["models"])
        or not model_set <= set(config["model_ids"])
    ):
        raise ValueError("Invalid evaluated model set")
    catalog = {row["name"]: row for row in bundle["models"]["models"]}
    all_models = [model for model in config["model_ids"] if model in model_set]
    models = [
        model
        for model in all_models
        if selection["cohort"] == "all"
        or (
            catalog[model]["evaluation_dataset_exposure"] is False
            and catalog[model]["exposure_reference"]
        )
    ]
    fraction = str(float(selection["fraction"]))
    probe = selection["probe"]
    metric = selection["metric"]
    performance = unique_index(
        read_table(evaluation_path / "seed_summary.csv"),
        ["model", "layer", "fraction", "task", "probe", "split", "metric"],
    )
    seed_metrics = unique_index(
        read_table(evaluation_path / "metrics.csv"),
        ["model", "layer", "fraction", "task", "probe", "seed", "split", "metric"],
    )
    descriptor_index = unique_index(
        read_table(evaluation_path / "descriptors.csv"), ["model", "layer", "fraction"]
    )
    association_rows = read_table(analysis_path / "associations.csv")
    association_index = unique_index(
        association_rows,
        [
            "section",
            "cohort",
            "probe",
            "metric",
            "method",
            "task",
            "descriptor",
            "fraction",
            "omitted_model",
            "controls",
        ],
    )
    ranks = unique_index(
        read_table(analysis_path / "overall_ranks.csv"), ["model", "fraction", "probe"]
    )
    tasks = config["task_ids"]
    descriptor_ids = [row["id"] for row in bundle["descriptors"]["panel"]]
    selected_performance = []
    selected_descriptors = []
    selected_associations = []
    selected_ranks = []
    for model in all_models:
        key = (model, "final", fraction)
        if key not in descriptor_index:
            raise ValueError("Report descriptor grid is incomplete")
        row = descriptor_index[key]
        if int(row["records"]) < 1 or int(row["features"]) < 1:
            raise ValueError("Invalid descriptor support")
        for name in descriptor_ids:
            number(row[name])
        if model in models:
            selected_descriptors.append(
                {
                    "model": model,
                    "analysis_group": catalog[model]["analysis_group"],
                    "fraction": selection["fraction"],
                    "records": row["records"],
                    "features": row["features"],
                    **{name: row[name] for name in descriptor_ids},
                }
            )
        for task in tasks:
            key = (model, "final", fraction, task, probe, "test", metric)
            if key not in performance:
                raise ValueError("Report performance grid is incomplete")
            row = performance[key]
            mean, sd = number(row["mean"]), number(row["sample_std"])
            if mean is not None and not 0 <= mean <= 1 or sd is not None and sd < 0:
                raise ValueError("Invalid performance mean or spread")
            if int(row["planned_seeds"]) != len(
                config["training"]["probe_seeds"]
            ) or not 0 <= int(row["valid_seeds"]) <= int(row["planned_seeds"]):
                raise ValueError("Inconsistent performance seed support")
            if mean is not None and int(row["valid_seeds"]) != int(
                row["planned_seeds"]
            ):
                raise ValueError("Partial seed mean cannot be reported as complete")
            if mean is not None and (
                (int(row["planned_seeds"]) > 1) != (sd is not None)
            ):
                raise ValueError("Seed spread does not match the planned seed count")
            if (
                (mean is None) != (row["status"] != "OK")
                or mean is None
                and sd is not None
            ):
                raise ValueError(
                    "Performance status differs from numerical availability"
                )
            flags = []
            for seed in config["training"]["probe_seeds"]:
                seed_key = (
                    model,
                    "final",
                    fraction,
                    task,
                    probe,
                    str(seed),
                    "test",
                    metric,
                )
                if seed_key not in seed_metrics:
                    raise ValueError("Report seed grid is incomplete")
                flags.append(boolean(seed_metrics[seed_key]["low_support"]))
            if model in models:
                selected_performance.append(
                    {
                        "model": model,
                        "analysis_group": catalog[model]["analysis_group"],
                        "task": task,
                        "probe": probe,
                        "fraction": selection["fraction"],
                        "metric": metric,
                        **{
                            name: row[name]
                            for name in (
                                "mean",
                                "sample_std",
                                "valid_seeds",
                                "planned_seeds",
                                "status",
                                "reason",
                            )
                        },
                        "low_support": any(flags),
                    }
                )
        rank_key = (model, fraction, probe)
        if rank_key not in ranks:
            raise ValueError("Report ranking grid is incomplete")
        rank = ranks[rank_key]
        value = number(rank["mean_task_percentile"])
        if value is not None and (not 0 <= value <= 1 or not boolean(rank["included"])):
            raise ValueError("Invalid task percentile")
        if model in models:
            selected_ranks.append(
                {
                    **{
                        key: rank[key]
                        for key in (
                            "model",
                            "fraction",
                            "probe",
                            "mean_task_percentile",
                            "tasks",
                            "common_models",
                            "included",
                        )
                    },
                    "metric": config["evaluation"]["primary_metric"],
                    "analysis_group": catalog[model]["analysis_group"],
                }
            )
    cohort_hashes = {
        (
            descriptor_index[(model, "final", fraction)]["records"],
            descriptor_index[(model, "final", fraction)]["record_order_sha256"],
        )
        for model in all_models
    }
    if len(cohort_hashes) != 1:
        raise ValueError("Models have different descriptor cohorts")
    for task in tasks:
        for name in descriptor_ids:
            key = (
                selection["section"],
                selection["cohort"],
                probe,
                metric,
                selection["method"],
                task,
                name,
                fraction,
                "",
                "",
            )
            if key not in association_index:
                raise ValueError(
                    "Report association grid is incomplete; this cohort may not have been analyzed"
                )
            row = association_index[key]
            for field in ("estimate", "ci_low", "ci_high"):
                value = number(row[field])
                if value is not None and not -1.0000001 <= value <= 1.0000001:
                    raise ValueError("Invalid correlation or interval")
            for field in ("p_value", "q_value"):
                value = number(row[field])
                if value is not None and not 0 <= value <= 1:
                    raise ValueError("Invalid p-value or q-value")
            low, high = number(row["ci_low"]), number(row["ci_high"])
            if (low is None) != (high is None) or low is not None and low > high:
                raise ValueError("Invalid interval endpoints")
            if int(row["models"]) != len(models):
                raise ValueError(
                    "Association checkpoint count differs from report cohort"
                )
            selected_associations.append(
                {column: row[column] for column in ASSOCIATION_COLUMNS}
            )
    return {
        "bundle": bundle,
        "settings": settings,
        "models": models,
        "all_models": all_models,
        "tasks": tasks,
        "descriptor_ids": descriptor_ids,
        "catalog": catalog,
        "performance": selected_performance,
        "descriptors": selected_descriptors,
        "associations": selected_associations,
        "all_associations": [
            {column: row[column] for column in ASSOCIATION_COLUMNS}
            for row in association_rows
        ],
        "ranks": selected_ranks,
        "evaluation_metadata_sha256": file_hash(evaluation_path / "metadata.json"),
        "analysis_metadata_sha256": file_hash(analysis_path / "metadata.json"),
    }


def run_report(
    evaluation_path, analysis_path, settings_path, output_path, example=False
):
    data = load_report_data(evaluation_path, analysis_path, settings_path)
    settings = data["settings"]
    selection = settings["selection"]
    config = data["bundle"]["experiment"]
    display = settings["display"]
    output = new_run(output_path, [evaluation_path, analysis_path, settings_path])
    metadata = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_kind": "generated_example" if example else "experiment_outputs",
        "selection": selection,
        "models": data["models"],
        "tasks": data["tasks"],
        "evaluation_metadata_sha256": data["evaluation_metadata_sha256"],
        "analysis_metadata_sha256": data["analysis_metadata_sha256"],
        "settings_sha256": file_hash(settings_path),
        "versions": versions(["numpy", "matplotlib", "pillow", "fonttools"]),
        "scope": "aggregate report tables and figures",
    }
    write_json(output / "metadata.json", metadata)
    log_event(output, "START aggregate report")
    table_dir = output / "tables"
    table_dir.mkdir()
    figure_dir = output / "figures"
    figure_dir.mkdir()
    # Method settings are embedded; local configuration-file references are omitted.
    methods = {
        key: value for key, value in data["bundle"].items() if key != "experiment"
    }
    methods["experiment"] = {
        key: value for key, value in config.items() if key != "files"
    }
    write_json(
        output / "config.resolved.json", {"reporting": settings, "methods": methods}
    )
    figures = []
    outputs = ["config.resolved.json"]
    scatter_rows = []
    tasks, models, descriptor_ids = (
        data["tasks"],
        data["models"],
        data["descriptor_ids"],
    )
    pindex = {(row["model"], row["task"]): row for row in data["performance"]}
    aindex = {(row["task"], row["descriptor"]): row for row in data["associations"]}
    dindex = {row["model"]: row for row in data["descriptors"]}
    groups = [data["catalog"][model]["analysis_group"] for model in models]
    styles = group_styles(groups, settings["style"]["palette"])
    cohort_label = {
        "all": "All checkpoints",
        "documented_unexposed": "Documented unexposed checkpoints",
    }[selection["cohort"]]
    scope = f"{label(settings, 'probes', selection['probe'])} · {label(settings, 'metrics', selection['metric'])} · training {selection['fraction']:.0%} · {cohort_label}"
    marker = "GENERATED DATA · workflow example. " if example else ""

    def export_table(name, columns, rows):
        path = table_dir / name
        write_table(path, columns, rows)
        outputs.append(path.relative_to(output).as_posix())

    def export_figure(kind, page, figure, task_ids, descriptors):
        name = f"{kind}_{page:02d}"
        paths = save_figure(
            figure,
            figure_dir / name,
            settings["figures"]["formats"],
            settings["figures"]["dpi"],
        )
        files = {path.suffix[1:]: path.relative_to(output).as_posix() for path in paths}
        figures.append(
            {
                "id": name,
                "kind": kind,
                "page": page,
                "tasks": task_ids,
                "descriptors": descriptors,
                "files": files,
            }
        )
        outputs.extend(files.values())
        print(f"Report: {name}", flush=True)

    try:
        export_table("performance.csv", PERFORMANCE_COLUMNS, data["performance"])
        export_table(
            "descriptors.csv",
            [
                "model",
                "analysis_group",
                "fraction",
                "records",
                "features",
                *descriptor_ids,
            ],
            data["descriptors"],
        )
        export_table("associations.csv", ASSOCIATION_COLUMNS, data["all_associations"])
        export_table(
            "selected_associations.csv", ASSOCIATION_COLUMNS, data["associations"]
        )
        export_table(
            "overall_ranks.csv",
            [
                "model",
                "fraction",
                "probe",
                "metric",
                "mean_task_percentile",
                "tasks",
                "common_models",
                "included",
                "analysis_group",
            ],
            data["ranks"],
        )
        display_rows = [
            {
                "model": model,
                **{
                    task: display_performance(
                        pindex[(model, task)]["mean"],
                        pindex[(model, task)]["sample_std"],
                        display["digits"],
                        display["missing"],
                    )
                    + (
                        " †"
                        if pindex[(model, task)]["low_support"]
                        and number(pindex[(model, task)]["mean"]) is not None
                        else ""
                    )
                    for task in tasks
                },
            }
            for model in models
        ]
        export_table("performance_display.csv", ["model", *tasks], display_rows)
        performance_md = []
        for page, part in enumerate(
            chunks(tasks, settings["figures"]["maximum_columns"]), 1
        ):
            performance_md.append(
                f"## Task panel {page}\n\n"
                + markdown_table(
                    ["Model", *[label(settings, "tasks", task) for task in part]],
                    [
                        [
                            model,
                            *[
                                next(
                                    row for row in display_rows if row["model"] == model
                                )[task]
                                for task in part
                            ],
                        ]
                        for model in models
                    ],
                )
            )
        (table_dir / "performance.md").write_text(
            "# Performance\n\n"
            + marker
            + markdown_cell(scope)
            + "\n\nSeed mean ± sample SD; SD is omitted for one planned seed. † Low training or evaluation support under the configured thresholds.\n\n"
            + "\n".join(performance_md),
            encoding="utf-8",
        )
        outputs.append("tables/performance.md")
        association_md = [
            "# Descriptor associations\n",
            marker + markdown_cell(scope) + "\n",
            "Intervals use the configured model-group bootstrap. P-values and q-values use the tests and families specified in the protocol.\n",
        ]
        for task in tasks:
            rows = []
            for descriptor in descriptor_ids:
                row = aindex[(task, descriptor)]
                low, high = number(row["ci_low"]), number(row["ci_high"])
                interval = (
                    display["missing"]
                    if low is None
                    else f"[{low:.{display['coefficient_digits']}f}, {high:.{display['coefficient_digits']}f}]"
                )
                rows.append(
                    [
                        label(settings, "descriptors", descriptor),
                        display_number(
                            row["estimate"],
                            display["coefficient_digits"],
                            display["missing"],
                        ),
                        interval,
                        display_probability(
                            row["p_value"], display["digits"], display["missing"]
                        ),
                        display_probability(
                            row["q_value"], display["digits"], display["missing"]
                        ),
                        row["models"],
                        row["independent_groups"],
                        row["reason"] or row["status"],
                    ]
                )
            association_md.append(
                "## "
                + markdown_cell(label(settings, "tasks", task))
                + "\n\n"
                + markdown_table(
                    [
                        "Descriptor",
                        "Coefficient",
                        "Interval",
                        "p",
                        "q",
                        "Checkpoints",
                        "Groups",
                        "Status / note",
                    ],
                    rows,
                )
            )
        (table_dir / "associations.md").write_text(
            "\n\n".join(association_md) + "\n", encoding="utf-8"
        )
        outputs.append("tables/associations.md")
        max_rows, max_columns = (
            settings["figures"]["maximum_rows"],
            settings["figures"]["maximum_columns"],
        )
        with plot_style(settings["style"]):
            page = 0
            for row_part in chunks(models, max_rows):
                for task_part in chunks(tasks, max_columns):
                    page += 1
                    matrix = np.array(
                        [
                            [
                                number(pindex[(model, task)]["mean"])
                                for task in task_part
                            ]
                            for model in row_part
                        ],
                        dtype=float,
                    ).reshape(len(row_part), len(task_part))
                    annotations = [
                        [
                            display_number(
                                pindex[(model, task)]["mean"],
                                display["digits"],
                                display["missing"],
                            )
                            + (
                                " †"
                                if pindex[(model, task)]["low_support"]
                                and number(pindex[(model, task)]["mean"]) is not None
                                else ""
                            )
                            for task in task_part
                        ]
                        for model in row_part
                    ]
                    figure = heatmap_figure(
                        matrix,
                        row_part,
                        [label(settings, "tasks", task) for task in task_part],
                        f"Performance · {scope}",
                        marker
                        + "Cell values are seed means. † Low support; a dash marks undefined values. Seed SD is in the tables.",
                        settings["style"],
                        annotations=annotations,
                    )
                    export_figure("performance", page, figure, task_part, [])
            page = 0
            for descriptor_part in chunks(descriptor_ids, max_rows):
                for task_part in chunks(tasks, max_columns):
                    page += 1
                    matrix = np.array(
                        [
                            [
                                number(aindex[(task, name)]["estimate"])
                                for task in task_part
                            ]
                            for name in descriptor_part
                        ],
                        dtype=float,
                    )
                    annotations = [
                        [
                            display_number(
                                aindex[(task, name)]["estimate"],
                                display["coefficient_digits"],
                                display["missing"],
                            )
                            + (
                                " *"
                                if number(aindex[(task, name)]["q_value"]) is not None
                                and number(aindex[(task, name)]["q_value"])
                                <= display["q_threshold"]
                                else ""
                            )
                            for task in task_part
                        ]
                        for name in descriptor_part
                    ]
                    figure = heatmap_figure(
                        matrix,
                        [
                            label(settings, "descriptors", name)
                            for name in descriptor_part
                        ],
                        [label(settings, "tasks", task) for task in task_part],
                        f"{label(settings, 'methods', selection['method'])} associations · {scope}",
                        marker
                        + f"* q ≤ {display['q_threshold']:g} in the configured test family. A dash marks undefined values. Checkpoint-row p-values; grouped intervals are separate.",
                        settings["style"],
                        "association",
                        annotations,
                    )
                    export_figure(
                        "associations", page, figure, task_part, descriptor_part
                    )
            forest_panels = []
            for task in tasks:
                forest_panels.append(
                    {
                        "title": label(settings, "tasks", task),
                        "rows": [
                            {
                                "label": label(settings, "descriptors", name),
                                "estimate": number(aindex[(task, name)]["estimate"]),
                                "low": number(aindex[(task, name)]["ci_low"]),
                                "high": number(aindex[(task, name)]["ci_high"]),
                                "significant": number(aindex[(task, name)]["q_value"])
                                is not None
                                and number(aindex[(task, name)]["q_value"])
                                <= display["q_threshold"],
                            }
                            for name in descriptor_ids
                        ],
                    }
                )
            for page, panels in enumerate(
                chunks(forest_panels, min(2, settings["figures"]["panels_per_page"])), 1
            ):
                figure = forest_figure(
                    panels,
                    f"{label(settings, 'methods', selection['method'])} coefficients · {scope}",
                    marker
                    + f"Lines: {config['analysis']['confidence_level']:.0%} group-bootstrap intervals. Filled points have intervals; hollow points lack one. * q ≤ {display['q_threshold']:g}.",
                    settings["style"],
                )
                export_figure(
                    "intervals",
                    page,
                    figure,
                    tasks[
                        (page - 1)
                        * min(2, settings["figures"]["panels_per_page"]) : page
                        * min(2, settings["figures"]["panels_per_page"])
                    ],
                    descriptor_ids,
                )
            panels = []
            for task in tasks:
                for descriptor in settings["figures"]["scatter_descriptors"]:
                    points = []
                    for model in models:
                        x, y = (
                            number(dindex[model][descriptor]),
                            number(pindex[(model, task)]["mean"]),
                        )
                        if x is not None and y is not None:
                            group = data["catalog"][model]["analysis_group"]
                            points.append({"x": x, "y": y, "group": group})
                            scatter_rows.append(
                                {
                                    "model": model,
                                    "analysis_group": group,
                                    "task": task,
                                    "descriptor": descriptor,
                                    "fraction": selection["fraction"],
                                    "probe": selection["probe"],
                                    "metric": selection["metric"],
                                    "x": dindex[model][descriptor],
                                    "y": pindex[(model, task)]["mean"],
                                }
                            )
                    row = aindex[(task, descriptor)]
                    symbol = {"spearman": "ρ", "pearson": "r", "kendall": "τ"}[
                        selection["method"]
                    ]
                    statistics = f"{symbol} = {display_number(row['estimate'], 2)}; q = {display_probability(row['q_value'], 3)}; paired checkpoints = {len(points)}"
                    panels.append(
                        {
                            "title": label(settings, "tasks", task),
                            "x_label": label(settings, "descriptors", descriptor),
                            "y_label": label(settings, "metrics", selection["metric"]),
                            "points": points,
                            "statistics": statistics,
                            "task": task,
                            "descriptor": descriptor,
                        }
                    )
            for page, part in enumerate(
                chunks(panels, settings["figures"]["panels_per_page"]), 1
            ):
                if not part:
                    continue
                figure = scatter_figure(
                    part,
                    styles,
                    f"{label(settings, 'methods', selection['method'])} paired views · {scope}",
                    marker
                    + "One point per checkpoint; color/shape identify analysis groups. Missing pairs are omitted from points; stored association status is unchanged.",
                    settings["style"],
                )
                export_figure(
                    "scatter",
                    page,
                    figure,
                    [panel["task"] for panel in part],
                    [panel["descriptor"] for panel in part],
                )
            ranks = [
                {
                    "model": row["model"],
                    "group": row["analysis_group"],
                    "value": number(row["mean_task_percentile"]),
                }
                for row in data["ranks"]
            ]
            ranks.sort(
                key=lambda row: (
                    row["value"] is None,
                    -row["value"] if row["value"] is not None else 0,
                    models.index(row["model"]),
                )
            )
            rank_scope = f"{label(settings, 'probes', selection['probe'])} · {label(settings, 'metrics', config['evaluation']['primary_metric'])} · training {selection['fraction']:.0%}"
            figure = ranking_figure(
                ranks,
                styles,
                f"Overall task ranking · {rank_scope}",
                marker
                + "Ranks retain the full analysis common model set; the displayed cohort is filtered. Values are task percentiles, without uncertainty bars.",
                settings["style"],
            )
            export_figure("overall_ranks", 1, figure, tasks, [])
        export_table(
            "scatter_points.csv",
            [
                "model",
                "analysis_group",
                "task",
                "descriptor",
                "fraction",
                "probe",
                "metric",
                "x",
                "y",
            ],
            scatter_rows,
        )
        write_json(
            output / "figures.json", {"figures": figures, "group_styles": styles}
        )
        outputs.append("figures.json")
        preview = []
        for kind in ("performance", "associations", "overall_ranks"):
            item = next((item for item in figures if item["kind"] == kind), None)
            if item and "png" in item["files"]:
                preview.append(
                    f"![{kind.replace('_', ' ').capitalize()}]({item['files']['png']})"
                )
        heading = (
            "Generated-data example" if example else "ECG foundation model analysis"
        )
        description = (
            "Figures and tables use generated embeddings to check the reporting workflow."
            if example
            else "Aggregate results from the linked evaluation and analysis runs."
        )
        index = [
            f"# {heading}",
            description,
            markdown_cell(scope),
            f"{len(models)} checkpoints; {len(tasks)} tasks; {len(descriptor_ids)} descriptors.",
            "[Performance table](tables/performance.md) · [Association tables](tables/associations.md)",
            "Full-precision CSVs: [performance](tables/performance.csv), [descriptors](tables/descriptors.csv), [associations](tables/associations.csv), [ranks](tables/overall_ranks.csv), and [scatter points](tables/scatter_points.csv).",
            "Performance tables show seed mean and sample SD. Association intervals use grouped bootstrap resampling; p-values use the configured checkpoint-row tests. Support flags, undefined outcomes, and their reasons remain in CSVs.",
            *preview,
            "## Figure files",
        ]
        for item in figures:
            links = " · ".join(
                f"[{format_name.upper()}]({filename})"
                for format_name, filename in item["files"].items()
            )
            index.append(f"- {item['id']}: {links}")
        (output / "report.md").write_text("\n\n".join(index) + "\n", encoding="utf-8")
        outputs.append("report.md")
        implementation = [
            "common/reporting.py",
            "common/report_tables.py",
            "common/plotting.py",
            "scripts/render_report.py",
            "common/tables.py",
            "common/artifacts.py",
        ]
        metadata.update(
            status="COMPLETE",
            figure_sets=len(figures),
            figure_files=sum(len(item["files"]) for item in figures),
            low_support_cells=sum(row["low_support"] for row in data["performance"]),
            undefined_performance_cells=sum(
                number(row["mean"]) is None for row in data["performance"]
            ),
            files_sha256={name: file_hash(output / name) for name in outputs},
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
