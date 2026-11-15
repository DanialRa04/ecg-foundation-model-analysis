"""Selection, numerical validation, and display-only formatting for reports."""

import html
import math
from copy import deepcopy
from pathlib import Path
from common.configuration import read_json


def number(value):
    if value is None or value == "":
        return None
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Report tables require finite numbers or blank missing values")
    return result


def boolean(value):
    if str(value).lower() not in {"true", "false"}:
        raise ValueError("Expected an explicit boolean flag")
    return str(value).lower() == "true"


def label(settings, category, name):
    return (
        settings["labels"]
        .get(category, {})
        .get(name, str(name).replace("_", " ").capitalize())
    )


def display_number(value, digits=3, missing="—"):
    value = number(value)
    return missing if value is None else f"{value:.{digits}f}"


def display_performance(mean, sd, digits=3, missing="—"):
    mean, sd = number(mean), number(sd)
    if mean is None:
        return missing
    result = f"{mean:.{digits}f}"
    return result if sd is None else f"{result} ± {sd:.{digits}f}"


def display_probability(value, digits=3, missing="—"):
    value = number(value)
    if value is None:
        return missing
    if not 0 <= value <= 1:
        raise ValueError("Probability must be between zero and one")
    threshold = 10 ** (-digits)
    return f"<{threshold:.{digits}f}" if value < threshold else f"{value:.{digits}f}"


def markdown_cell(value):
    return (
        html.escape(str(value), quote=False)
        .replace("\\", "\\\\")
        .replace("|", "\\|")
        .replace("\n", " ")
        .replace("\r", " ")
        .replace("`", "&#96;")
        .replace("[", "&#91;")
        .replace("]", "&#93;")
    )


def markdown_table(columns, rows):
    lines = [
        "| " + " | ".join(markdown_cell(value) for value in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend(
        "| " + " | ".join(markdown_cell(value) for value in row) + " |" for row in rows
    )
    return "\n".join(lines) + "\n"


def chunks(values, size):
    return [values[start : start + size] for start in range(0, len(values), size)] or [
        []
    ]


def load_settings(path, bundle):
    settings = deepcopy(read_json(Path(path)))
    config = bundle["experiment"]
    selected = settings["selection"]
    defaults = {
        "probe": config["evaluation"]["primary_probe"],
        "metric": config["evaluation"]["primary_metric"],
        "fraction": config["training"]["primary_fraction"],
        "method": config["analysis"]["primary_method"],
    }
    for key, value in defaults.items():
        if selected[key] is None:
            selected[key] = value
    if selected["probe"] not in config["evaluation"]["probe_ids"] or selected[
        "metric"
    ] not in {"average_precision", "auroc"}:
        raise ValueError("Report probe or metric is not configured")
    if selected["fraction"] not in config["training"]["fractions"] or selected[
        "method"
    ] not in [
        config["analysis"]["primary_method"],
        *config["analysis"]["secondary_methods"],
    ]:
        raise ValueError("Report fraction or method is not configured")
    primary = all(selected[key] == value for key, value in defaults.items())
    if selected["section"] == "auto":
        selected["section"] = "primary" if primary else "additional"
    if selected["section"] not in {"primary", "additional"} or selected[
        "cohort"
    ] not in {"all", "documented_unexposed"}:
        raise ValueError(
            "Report figures support checkpoint-level primary/additional comparisons"
        )
    if selected["section"] != ("primary" if primary else "additional"):
        raise ValueError("Report section does not match the configured comparison")
    figures, display, style = (
        settings["figures"],
        settings["display"],
        settings["style"],
    )
    for key in ("maximum_rows", "maximum_columns", "panels_per_page", "dpi"):
        if type(figures[key]) is not int or figures[key] < 1:
            raise ValueError(f"Invalid figure setting: {key}")
    if (
        figures["maximum_rows"] > 40
        or figures["maximum_columns"] > 20
        or figures["panels_per_page"] > 4
        or not 72 <= figures["dpi"] <= 600
    ):
        raise ValueError("Figure pagination or resolution exceeds supported bounds")
    if (
        not figures["formats"]
        or len(set(figures["formats"])) != len(figures["formats"])
        or set(figures["formats"]) - {"png", "svg", "pdf"}
    ):
        raise ValueError("Choose unique PNG, SVG, and/or PDF figure formats")
    descriptor_ids = {item["id"] for item in bundle["descriptors"]["panel"]}
    if len(set(figures["scatter_descriptors"])) != len(figures["scatter_descriptors"]):
        raise ValueError("Duplicate scatter descriptors")
    # A smaller experiment panel can omit any of the default scatter descriptors.
    figures["omitted_scatter_descriptors"] = [
        name for name in figures["scatter_descriptors"] if name not in descriptor_ids
    ]
    figures["scatter_descriptors"] = [
        name for name in figures["scatter_descriptors"] if name in descriptor_ids
    ]
    for key in ("digits", "coefficient_digits"):
        if type(display[key]) is not int or not 0 <= display[key] <= 8:
            raise ValueError("Display precision must be between zero and eight digits")
    if (
        not isinstance(display["missing"], str)
        or not 0 <= number(display["q_threshold"]) <= 1
    ):
        raise ValueError("Invalid missing-value label or q threshold")
    if (
        not isinstance(style["font_family"], str)
        or not style["font_family"]
        or not 6 <= number(style["font_size"]) <= 20
    ):
        raise ValueError("Invalid figure font")
    if not style["palette"] or any(
        not isinstance(value, str) for value in style["palette"]
    ):
        raise ValueError("A color palette is required")
    if any(
        not isinstance(values, dict)
        or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in values.items()
        )
        for values in settings["labels"].values()
    ):
        raise ValueError("Report labels must map strings to strings")
    return settings
