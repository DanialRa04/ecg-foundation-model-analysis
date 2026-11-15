"""Headless scientific figures with shared styling and explicit missing values."""

import math
import textwrap
import numpy as np
import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D


def plot_style(style):
    for color in [style["missing_color"], *style["palette"]]:
        if not mpl.colors.is_color_like(color):
            raise ValueError("Invalid report color")
    for name in ("performance_cmap", "association_cmap"):
        if style[name] not in mpl.colormaps:
            raise ValueError("Unknown report colormap")
    return mpl.rc_context(
        {
            "font.family": style["font_family"],
            "font.size": style["font_size"],
            "axes.titlesize": style["font_size"] + 1,
            "axes.labelsize": style["font_size"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#aab4c0",
            "text.color": "#202c3b",
            "axes.labelcolor": "#202c3b",
            "xtick.color": "#435269",
            "ytick.color": "#435269",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "text.usetex": False,
            "text.parse_math": False,
            "svg.fonttype": "none",
            "svg.hashsalt": "ecg-analysis-report",
            "pdf.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )


def group_styles(groups, palette):
    return {
        name: {
            "color": palette[i % len(palette)],
            "marker": ["o", "s", "D", "^"][i // len(palette) % 4],
        }
        for i, name in enumerate(dict.fromkeys(groups))
    }


def wrapped(value, width=23):
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def _figure(width, height, title, note, bottom=0.12):
    figure = Figure(figsize=(width, height), layout="constrained")
    FigureCanvasAgg(figure)
    figure.set_layout_engine("constrained", rect=(0, bottom, 1, 1 - bottom))
    figure.suptitle(wrapped(title, 85), fontweight="semibold")
    figure.text(
        0.015,
        0.015,
        note,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#526176",
        wrap=True,
    )
    return figure


def heatmap_figure(
    values,
    row_labels,
    column_labels,
    title,
    note,
    style,
    kind="performance",
    annotations=None,
):
    values = np.asarray(values, dtype=float)
    if values.shape != (len(row_labels), len(column_labels)):
        raise ValueError("Heatmap labels do not match its matrix")
    figure = _figure(
        max(7.0, 3.4 + 0.85 * len(column_labels)),
        max(4.0, 2.1 + 0.5 * len(row_labels)),
        title,
        note,
    )
    axis = figure.add_subplot()
    if not len(row_labels) or not len(column_labels):
        axis.text(
            0.5,
            0.5,
            "No eligible checkpoints",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        return figure
    cmap = mpl.colormaps[
        style["performance_cmap" if kind == "performance" else "association_cmap"]
    ].copy()
    cmap.set_bad(style["missing_color"])
    lower, upper = (0, 1) if kind == "performance" else (-1, 1)
    image = axis.imshow(
        np.ma.masked_invalid(values),
        vmin=lower,
        vmax=upper,
        cmap=cmap,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(
        range(len(column_labels)),
        [wrapped(value, 18) for value in column_labels],
        rotation=25,
        ha="right",
        rotation_mode="anchor",
    )
    axis.set_yticks(
        range(len(row_labels)), [wrapped(value, 28) for value in row_labels]
    )
    axis.tick_params(length=0, pad=8)
    for spine in axis.spines.values():
        spine.set_visible(False)
    for i in range(len(row_labels)):
        for j in range(len(column_labels)):
            value = values[i, j]
            text = (
                annotations[i][j]
                if annotations is not None
                else ("—" if not np.isfinite(value) else f"{value:.2f}")
            )
            rgba = (
                cmap((value - lower) / (upper - lower))
                if np.isfinite(value)
                else mpl.colors.to_rgba(style["missing_color"])
            )
            luminance = 0.2126 * rgba[0] + 0.7152 * rgba[1] + 0.0722 * rgba[2]
            axis.text(
                j,
                i,
                text,
                ha="center",
                va="center",
                fontsize=8,
                color="white" if luminance < 0.5 else "#172638",
            )
    colorbar = figure.colorbar(image, ax=axis, pad=0.025, shrink=0.8)
    colorbar.set_label(
        "Seed mean" if kind == "performance" else "Correlation coefficient"
    )
    return figure


def forest_figure(panels, title, note, style):
    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    height = max(5.0, 2.1 + 0.4 * max(len(panel["rows"]) for panel in panels))
    figure = _figure(
        7.0 if columns == 1 else 13.0, height * rows, title, note, bottom=0.1
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis, panel in zip(axes.flat, panels):
        labels = []
        for i, row in enumerate(panel["rows"]):
            labels.append(
                wrapped(row["label"] + (" *" if row["significant"] else ""), 29)
            )
            if row["estimate"] is not None:
                if row["low"] is not None and row["high"] is not None:
                    axis.hlines(
                        i, row["low"], row["high"], color="#8aa3b9", linewidth=2
                    )
                has_interval = row["low"] is not None and row["high"] is not None
                axis.plot(
                    row["estimate"],
                    i,
                    "o",
                    color=style["palette"][0],
                    markerfacecolor=style["palette"][0] if has_interval else "none",
                    markersize=5,
                    zorder=3,
                )
            else:
                axis.text(0, i, "—", ha="center", va="center", color="#7b8795")
        axis.set_yticks(range(len(labels)), labels)
        axis.invert_yaxis()
        axis.set_xlim(-1.07, 1.07)
        axis.axvline(0, color="#c1c9d2", linewidth=0.8, zorder=0)
        axis.set_xlabel("Correlation coefficient")
        axis.set_title(wrapped(panel["title"], 35), pad=12)
        axis.grid(axis="x", color="#edf0f4", linewidth=0.6)
        axis.tick_params(axis="y", length=0)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    return figure


def scatter_figure(panels, styles, title, note, style):
    columns = min(2, len(panels))
    rows = math.ceil(len(panels) / columns)
    legend_rows = math.ceil(len(styles) / min(4, max(1, len(styles))))
    bottom = min(0.32, 0.1 + 0.027 * legend_rows)
    figure = _figure(
        7.0 if columns == 1 else 12.0, 4.2 * rows + 1.0, title, note, bottom=bottom
    )
    axes = figure.subplots(rows, columns, squeeze=False)
    for axis, panel in zip(axes.flat, panels):
        for group, appearance in styles.items():
            points = [point for point in panel["points"] if point["group"] == group]
            if points:
                axis.scatter(
                    [point["x"] for point in points],
                    [point["y"] for point in points],
                    s=45,
                    color=appearance["color"],
                    marker=appearance["marker"],
                    edgecolor="white",
                    linewidth=0.5,
                    alpha=0.9,
                )
        if not panel["points"]:
            axis.text(
                0.5,
                0.5,
                "No finite paired values",
                ha="center",
                transform=axis.transAxes,
            )
        axis.set_xlabel(wrapped(panel["x_label"], 32))
        axis.set_ylabel(panel["y_label"])
        axis.set_ylim(-0.03, 1.03)
        axis.margins(x=0.1)
        axis.set_title(
            wrapped(panel["title"], 34) + "\n" + panel["statistics"],
            fontsize=10,
            pad=10,
        )
        axis.grid(color="#edf0f4", linewidth=0.6, zorder=0)
        axis.set_axisbelow(True)
    for axis in list(axes.flat)[len(panels) :]:
        axis.set_visible(False)
    if styles:
        handles = [
            Line2D(
                [],
                [],
                linestyle="",
                marker=value["marker"],
                color=value["color"],
                label=wrapped(group, 23),
                markersize=6,
            )
            for group, value in styles.items()
        ]
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.045),
            ncols=min(4, len(styles)),
            frameon=False,
            fontsize=8,
        )
    return figure


def ranking_figure(rows, styles, title, note, style):
    figure = _figure(8.0, max(4.0, 2.0 + 0.45 * len(rows)), title, note)
    axis = figure.add_subplot()
    for i, row in enumerate(rows):
        if row["value"] is None:
            axis.text(
                0.5,
                i,
                "Not available",
                ha="center",
                va="center",
                fontsize=9,
                color="#7b8795",
            )
        else:
            appearance = styles[row["group"]]
            axis.plot(
                row["value"],
                i,
                marker=appearance["marker"],
                color=appearance["color"],
                markersize=7,
            )
    axis.set_yticks(range(len(rows)), [wrapped(row["model"], 30) for row in rows])
    axis.invert_yaxis()
    axis.set_xlim(-0.03, 1.03)
    axis.set_xlabel("Mean within-task percentile")
    axis.grid(axis="x", color="#edf0f4")
    axis.tick_params(axis="y", length=0)
    if not rows:
        axis.text(
            0.5, 0.5, "No eligible checkpoints", ha="center", transform=axis.transAxes
        )
    return figure


def save_figure(figure, stem, formats, dpi):
    files = []
    try:
        for format_name in formats:
            target = stem.with_suffix("." + format_name)
            if target.exists():
                raise FileExistsError(target)
            metadata = (
                {"Date": None}
                if format_name == "svg"
                else {"CreationDate": None, "ModDate": None}
                if format_name == "pdf"
                else {"Software": "ECG Foundation Model Analysis"}
            )
            figure.savefig(
                target,
                format=format_name,
                dpi=dpi,
                bbox_inches="tight",
                pad_inches=0.15,
                metadata=metadata,
            )
            files.append(target)
    finally:
        figure.clear()
    return files
