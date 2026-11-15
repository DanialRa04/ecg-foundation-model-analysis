# Running the experiment

Start with the [usage guide](docs/usage.md). Dataset preparation is described in [datasets.md](docs/datasets.md), checkpoint acquisition in [models.md](docs/models.md), and signal preparation in [the preprocessing guide](docs/preprocessing/README.md).

For code verification, start with [the bounded workflow checks](docs/workflow_checks.md). All experiment results and generated figures stay local.

Start by checking the experiment settings, then the numerical code:

```text
python -m scripts.validate_config
python -m checks.check_configuration
python -m checks.check_descriptor_math --output outputs/checks/descriptor_math.json
```

Next run [preprocessing and extraction](docs/running_pipeline.md), [task preparation, probe evaluation, and statistical analysis](docs/evaluation.md), then [report tables and figures](docs/reporting.md).
