# Result tables and figures

This stage turns completed [evaluation and analysis](evaluation.md) outputs into aggregate tables and scientific figures. It uses NumPy and Matplotlib; model implementations and PyTorch are not needed to render reports.

Use Python 3.11 or later for the reporting environment. From the repository root:

```text
python -m pip install -r requirements-reporting.txt
python -m scripts.render_report --evaluation outputs/example/evaluation --analysis outputs/example/analysis --output outputs/example/report
```

The analysis must belong to the supplied evaluation run. The command checks completion status, file fingerprints, configuration agreement, and the selected result grids. Use a new output directory for every report.

For generated data, add `--example`. This labels the report and every figure as a generated-data example.

## Outputs

```text
report/
  report.md                       # table links, previews, figure index
  config.resolved.json            # report and experiment method settings
  figures.json                    # figure inventory and group styles
  metadata.json                   # inputs, selection, versions, hashes, status
  run.log
  tables/
    performance.csv               # full-precision selected seed summaries
    performance_display.csv       # rounded values for display
    performance.md
    descriptors.csv
    associations.csv              # all saved statistical comparisons
    selected_associations.csv      # comparison shown in the figures
    associations.md
    overall_ranks.csv
    scatter_points.csv            # exact checkpoint-level plotted values
  figures/
    performance_01.png / .svg / .pdf
    associations_01.png / .svg / .pdf
    intervals_01.png / .svg / .pdf
    scatter_01.png / .svg / .pdf
    overall_ranks_01.png / .svg / .pdf
```

Larger tables and figure panels continue onto numbered pages. The report exports aggregate values; recording IDs, patient mappings, waveforms, embeddings, and individual predictions are not copied.

## Reading the figures

- **Performance heatmaps:** seed means on a fixed 0–1 scale. Tables show sample SD separately, and omit SD for a single planned seed. A dagger (†) marks low training or evaluation support under the experiment's thresholds.
- **Association heatmaps:** the complete configured descriptor panel, with a fixed −1 to 1 scale. An asterisk marks the configured q-value threshold. Undefined cells show a dash.
- **Interval plots:** coefficients and configured group-bootstrap intervals. Filled points have an interval, including zero-width intervals; hollow points have no interval. Undefined estimates show a dash.
- **Scatter panels:** one checkpoint per finite descriptor/performance pair, with group colors and shapes. They display stored association statistics; plotting does not recompute a test after omitting missing pairs.
- **Overall rankings:** the saved mean within-task percentiles, using the experiment's primary metric and common complete model set. Filtering the displayed cohort retains that reference set. These figures have no uncertainty bars.

P-values, q-values, and group-bootstrap intervals retain the meaning specified in [the protocol](protocol.md). Display rounding does not alter analysis values. Small p-values and q-values use a format such as `<0.001`; CSVs preserve their original precision.

## Settings

Defaults are in [reporting.json](../configs/reporting.json). Copy that file and edit it, then pass `--settings path/to/reporting.json`.

The selection defaults to the primary probe, metric, fraction, and association method. For another configured comparison, edit the `selection` block of the copied file:

```json
{
  "probe": "linear",
  "metric": "auroc",
  "fraction": 0.5,
  "method": "spearman",
  "section": "auto",
  "cohort": "all"
}
```

`cohort` can also be `documented_unexposed`, when that comparison was included in analysis. Unknown exposure is not treated as unexposed. If the cohort has no eligible models, the report retains its undefined comparisons.

Settings also control label text, font, colors, output formats, display precision, and pagination. Model and task order follow the experiment configuration. Scatter descriptors are listed explicitly in settings; they are not selected using observed correlations. Listed scatter descriptors absent from a smaller configured panel are recorded as omitted in the resolved settings.

Reports stay under ignored `outputs/` by default. All generated tables, figures, and reports remain local, including any files placed in `results/` or `figures/`. GitHub contains the code, configurations, and guides for creating them.

## Checks

```text
python -m checks.check_reporting
```

The checks generate aggregate fixtures and exercise precision, support flags, missing outcomes, linked inputs, pagination, all export formats, and the aggregate-only output boundary. See [the verification record](verification.md).

Plot exports and layout use Matplotlib's [figure export API](https://matplotlib.org/stable/api/_as_gen/matplotlib.figure.Figure.savefig.html), [constrained layout](https://matplotlib.org/stable/users/explain/axes/constrainedlayout_guide.html), and [colormap guidance](https://matplotlib.org/stable/users/explain/colors/colormaps.html).
