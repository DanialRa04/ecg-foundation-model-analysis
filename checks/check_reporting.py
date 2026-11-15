"""Generated aggregate fixtures for report selection, precision, privacy, and exports."""

from copy import deepcopy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib as mpl
from matplotlib.image import imread
from common.artifacts import file_hash, write_json
from common.configuration import load_experiment, read_json
from common.experiment_data import verified_stage
from common.plotting import forest_figure, heatmap_figure, plot_style, save_figure
from common.report_tables import (
    display_performance,
    display_probability,
    load_settings,
    markdown_table,
    number,
)
from common.reporting import load_report_data, run_report
from common.tables import read_table, write_table


def reject(function):
    try:
        function()
    except (ValueError, KeyError, TypeError, FileExistsError):
        return
    raise AssertionError("Invalid report input was accepted")


def stage(path, models, extra=None):
    metadata = {
        "status": "COMPLETE",
        "models": models,
        "files_sha256": {
            item.name: file_hash(item)
            for item in path.iterdir()
            if item.is_file() and item.name != "metadata.json"
        },
    }
    metadata.update(extra or {})
    write_json(path / "metadata.json", metadata)


def fixture(base):
    root = Path(__file__).resolve().parents[1]
    bundle = load_experiment(root / "configs/experiment.json")
    models = ["Encoder A", "Encoder | B <tag>", "Encoder C with a longer label"]
    for i, model in enumerate(models):
        bundle["models"]["models"][i]["name"] = model
        bundle["models"]["models"][i]["analysis_group"] = f"Group {i + 1}"
    config = bundle["experiment"]
    config["model_ids"] = models
    tasks = ["atrial_fibrillation", "sinus_rhythm", "tachycardia"]
    config["task_ids"] = tasks
    config["training"]["fractions"] = [1.0]
    # The report embeds method values rather than local config-file references.
    config["files"]["models"] = "C:/private-local-layout/models.json"
    bundle["descriptors"]["panel"] = bundle["descriptors"]["panel"][:5]
    descriptors = [row["id"] for row in bundle["descriptors"]["panel"]]
    evaluation = base / "evaluation"
    analysis = base / "analysis"
    evaluation.mkdir()
    analysis.mkdir()
    for directory in (evaluation, analysis):
        write_json(directory / "config.resolved.json", bundle)
    performance, metrics, geometry, associations, ranks = [], [], [], [], []
    for i, model in enumerate(models):
        geometry.append(
            {
                "model": model,
                "layer": "final",
                "fraction": 1.0,
                "records": 100,
                "features": 8,
                "record_order_sha256": "shared_generated_cohort",
                **{
                    name: f"{2 + i + index / 10:.12f}"
                    for index, name in enumerate(descriptors)
                },
            }
        )
        ranks.append(
            {
                "model": model,
                "fraction": 1.0,
                "probe": "ann",
                "mean_task_percentile": i / 2 if i != 1 else None,
                "tasks": len(tasks),
                "common_models": 2,
                "included": i != 1,
            }
        )
        for j, task in enumerate(tasks):
            missing = i == 1 and j == 2
            for metric in ("average_precision", "auroc"):
                value = (
                    "0.12345678901234567"
                    if i == 0 and j == 0 and metric == "average_precision"
                    else str(0.25 + 0.2 * i + 0.02 * j)
                )
                performance.append(
                    {
                        "model": model,
                        "layer": "final",
                        "fraction": 1.0,
                        "task": task,
                        "probe": "ann",
                        "split": "test",
                        "metric": metric,
                        "mean": None if missing else value,
                        "sample_std": None if missing else ".0123456789012345",
                        "valid_seeds": 0 if missing else 2,
                        "planned_seeds": 2,
                        "status": "UNDEFINED" if missing else "OK",
                        "reason": "missing outcomes" if missing else "",
                        "sample_id": "FORBIDDEN_RECORD",
                    }
                )
                for seed in config["training"]["probe_seeds"]:
                    metrics.append(
                        {
                            "model": model,
                            "layer": "final",
                            "fraction": 1.0,
                            "task": task,
                            "probe": "ann",
                            "seed": seed,
                            "split": "test",
                            "metric": metric,
                            "value": None if missing else value,
                            "low_support": missing or i == 0,
                        }
                    )
    for metric in ("average_precision", "auroc"):
        for cohort in ("all", "documented_unexposed"):
            for j, task in enumerate(tasks):
                for descriptor in descriptors:
                    missing = j == 2 or cohort != "all"
                    associations.append(
                        {
                            "section": "primary"
                            if metric == "average_precision"
                            else "additional",
                            "cohort": cohort,
                            "probe": "ann",
                            "metric": metric,
                            "method": "spearman",
                            "task": task,
                            "descriptor": descriptor,
                            "fraction": 1.0,
                            "omitted_model": "",
                            "controls": "",
                            "estimate": None if missing else 0.8,
                            "p_value": None if missing else 0.0001,
                            "ci_low": None if missing else 0.2,
                            "ci_high": None if missing else 1.0,
                            "models": 3 if cohort == "all" else 0,
                            "independent_groups": 3 if cohort == "all" else 0,
                            "bootstrap_requested": 2000,
                            "bootstrap_valid": 0 if missing else 1900,
                            "status": "UNDEFINED" if missing else "OK",
                            "reason": "undefined input" if missing else "",
                            "q_value": None if missing else 0.0004,
                            "fdr_tests": 0 if missing else 10,
                            "patient_id": "FORBIDDEN_PATIENT",
                        }
                    )
    for name, rows in [
        ("seed_summary.csv", performance),
        ("metrics.csv", metrics),
        ("descriptors.csv", geometry),
    ]:
        write_table(evaluation / name, list(rows[0]), rows)
    write_table(
        evaluation / "predictions.csv",
        ["sample_id", "probability"],
        [{"sample_id": "FORBIDDEN_PREDICTION", "probability": 0.8}],
    )
    stage(evaluation, models)
    write_table(analysis / "associations.csv", list(associations[0]), associations)
    write_table(analysis / "overall_ranks.csv", list(ranks[0]), ranks)
    stage(
        analysis,
        models,
        {"evaluation_metadata_sha256": file_hash(evaluation / "metadata.json")},
    )
    settings = read_json(root / "configs/reporting.json")
    settings["figures"].update(
        formats=["png", "svg"], dpi=72, maximum_rows=2, maximum_columns=2
    )
    settings_path = base / "settings.json"
    write_json(settings_path, settings)
    return evaluation, analysis, settings_path, bundle


def run_checks():
    root = Path(__file__).resolve().parents[1]
    (root / "outputs").mkdir(exist_ok=True)
    passed = []
    assert display_performance(".123456789", ".0123456") == "0.123 ± 0.012"
    assert (
        display_performance(".5", None) == "0.500"
        and display_performance("", None) == "—"
    )
    assert display_probability(0.0004) == "<0.001" and display_probability(None) == "—"
    reject(lambda: number("nan"))
    reject(lambda: number("inf"))
    text = markdown_table(["A | B"], [["<script>[link]`x`"]])
    assert "\\|" in text and "&lt;script&gt;" in text and "<script>" not in text
    passed.append("display-only rounding small p-value notation and escaped Markdown")
    with TemporaryDirectory(
        prefix="reporting_checks_", dir=root / "outputs"
    ) as directory:
        base = Path(directory)
        evaluation, analysis, settings_path, bundle = fixture(base)
        source_hashes = {
            path: file_hash(path)
            for parent in (evaluation, analysis)
            for path in parent.iterdir()
            if path.is_file()
        }
        data = load_report_data(evaluation, analysis, settings_path)
        assert (
            data["models"] == bundle["experiment"]["model_ids"]
            and data["performance"][0]["mean"] == "0.12345678901234567"
        )
        assert data["performance"][0]["low_support"]
        passed.append(
            "configured model order seed support and complete aggregate joins"
        )
        settings = read_json(settings_path)
        for key, value in [("maximum_rows", 0), ("formats", ["../png"])]:
            invalid = deepcopy(settings)
            invalid["figures"][key] = value
            write_json(base / "invalid.json", invalid)
            reject(lambda: load_settings(base / "invalid.json", bundle))
        invalid = deepcopy(settings)
        invalid["selection"]["fraction"] = 0.25
        write_json(base / "invalid.json", invalid)
        reject(lambda: load_settings(base / "invalid.json", bundle))
        passed.append("invalid report selections formats and pagination are rejected")
        alternative = deepcopy(settings)
        alternative["selection"]["metric"] = "auroc"
        write_json(base / "alternative.json", alternative)
        alternative_data = load_report_data(
            evaluation, analysis, base / "alternative.json"
        )
        assert alternative_data["settings"]["selection"]["section"] == "additional"
        assert all(
            row["metric"] == "average_precision" for row in alternative_data["ranks"]
        )
        restricted = deepcopy(settings)
        restricted["selection"]["cohort"] = "documented_unexposed"
        write_json(base / "restricted.json", restricted)
        restricted_data = load_report_data(
            evaluation, analysis, base / "restricted.json"
        )
        assert (
            restricted_data["models"] == []
            and len(restricted_data["associations"]) == 15
        )
        passed.append("secondary-metric selection and explicit empty exposure cohorts")
        before = mpl.rcParams["font.size"]
        with plot_style(settings["style"]):
            heatmap = heatmap_figure(
                [[0.5, np.nan]],
                ["Encoder"],
                ["A", "B"],
                "Example",
                "Generated data",
                settings["style"],
            )
            assert heatmap.axes[0].images[0].get_clim() == (0, 1)
            assert heatmap.axes[0].images[0].get_array().mask.tolist() == [
                [False, True]
            ]
            save_figure(heatmap, base / "direct_heatmap", ["pdf"], 72)
            forest = forest_figure(
                [
                    {
                        "title": "Intervals",
                        "rows": [
                            {
                                "label": "Zero width",
                                "estimate": 1.0,
                                "low": 1.0,
                                "high": 1.0,
                                "significant": False,
                            },
                            {
                                "label": "Missing interval",
                                "estimate": 0.5,
                                "low": None,
                                "high": None,
                                "significant": False,
                            },
                        ],
                    }
                ],
                "Example",
                "Generated data",
                settings["style"],
            )
            points = [line for line in forest.axes[0].lines if line.get_marker() == "o"]
            assert (
                points[0].get_markerfacecolor() != "none"
                and points[1].get_markerfacecolor() == "none"
            )
            forest.clear()
        assert mpl.rcParams["font.size"] == before and (
            base / "direct_heatmap.pdf"
        ).read_bytes().startswith(b"%PDF")
        passed.append(
            "fixed heatmap scales explicit missing cells and zero-width versus absent intervals"
        )
        result = run_report(
            evaluation, analysis, settings_path, base / "report", example=True
        )
        assert (
            result["status"] == "COMPLETE"
            and result["data_kind"] == "generated_example"
            and result["figure_sets"] == 16
        )
        passed.append("paginated end-to-end report with all five figure families")
        output = base / "report"
        verified_stage(output, ["report.md", "figures.json", "tables/performance.csv"])
        raw = read_table(output / "tables/performance.csv")
        assert (
            raw[0]["mean"] == "0.12345678901234567"
            and raw[0]["sample_std"] == ".0123456789012345"
        )
        display_rows = read_table(output / "tables/performance_display.csv")
        assert display_rows[0]["atrial_fibrillation"] == "0.123 ± 0.012 †"
        assert display_rows[1]["tachycardia"] == "—"
        points = read_table(output / "tables/scatter_points.csv")
        assert len(points) == 32 and not any(
            row["model"] == bundle["experiment"]["model_ids"][1]
            and row["task"] == "tachycardia"
            for row in points
        )
        passed.append("full-precision CSV preservation and missing-pair plot data")
        for path in output.rglob("*"):
            if path.suffix in {".md", ".csv", ".json", ".svg"}:
                contents = path.read_text(encoding="utf-8")
                assert not any(
                    value in contents
                    for value in [
                        "FORBIDDEN_RECORD",
                        "FORBIDDEN_PATIENT",
                        "FORBIDDEN_PREDICTION",
                        "C:/private-local-layout",
                    ]
                )
        assert "&lt;tag&gt;" in (output / "tables/performance.md").read_text(
            encoding="utf-8"
        )
        assert "GENERATED DATA" in next((output / "figures").glob("*.svg")).read_text(
            encoding="utf-8"
        )
        passed.append(
            "aggregate-only export local-path exclusion and generated-data labels"
        )
        for path in (output / "figures").glob("*.png"):
            image = imread(path)
            assert (
                image.shape[0] > 100
                and image.shape[1] > 100
                and image[:, :, :3].std() > 0.03
            )
        for path in (output / "figures").glob("*.svg"):
            assert ET.parse(path).getroot().tag.endswith("svg")
        assert source_hashes == {path: file_hash(path) for path in source_hashes}
        reject(
            lambda: run_report(
                evaluation, analysis, settings_path, output, example=True
            )
        )
        passed.append(
            "valid raster vector and PDF exports input preservation and output collision protection"
        )
        # Grid validation is independent of hashes and run identity checks.
        source_bytes = {
            path: path.read_bytes()
            for path in [
                evaluation / "seed_summary.csv",
                evaluation / "metadata.json",
                analysis / "metadata.json",
            ]
        }
        rows = read_table(evaluation / "seed_summary.csv")
        (evaluation / "seed_summary.csv").unlink()
        write_table(evaluation / "seed_summary.csv", list(rows[0]), rows[1:])
        stage(evaluation, bundle["experiment"]["model_ids"])
        stage(
            analysis,
            bundle["experiment"]["model_ids"],
            {"evaluation_metadata_sha256": file_hash(evaluation / "metadata.json")},
        )
        reject(lambda: load_report_data(evaluation, analysis, settings_path))
        for path, contents in source_bytes.items():
            path.write_bytes(contents)
        passed.append(
            "incomplete report grids fail even when file hashes and linked identities match"
        )
        # A bad style creates a failed stage, preserving its error rather than a complete report.
        bad = deepcopy(settings)
        bad["style"]["missing_color"] = "not-a-color"
        write_json(base / "bad_style.json", bad)
        reject(
            lambda: run_report(
                evaluation,
                analysis,
                base / "bad_style.json",
                base / "failed_report",
                example=True,
            )
        )
        assert read_json(base / "failed_report/metadata.json")["status"] == "FAILED"
        passed.append("failed rendering is recorded as an incomplete stage")
        original = (evaluation / "metadata.json").read_bytes()
        changed = read_json(evaluation / "metadata.json")
        changed["new_identity"] = "different run"
        write_json(evaluation / "metadata.json", changed)
        reject(lambda: load_report_data(evaluation, analysis, settings_path))
        (evaluation / "metadata.json").write_bytes(original)
        with (analysis / "associations.csv").open("a") as stream:
            stream.write("\n")
        reject(lambda: load_report_data(evaluation, analysis, settings_path))
        passed.append("linked-run identity and input fingerprint enforcement")
    return {"status": "PASS", "checks": passed}


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
