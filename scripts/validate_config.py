"""Validate experiment settings without downloading datasets or models."""

import argparse
import json
from pathlib import Path

from common.configuration import load_experiment, summarize_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/experiment.json")
    parser.add_argument("--output", type=Path, help="Optional new JSON report")
    args = parser.parse_args()
    try:
        summary = summarize_experiment(load_experiment(args.config))
        rendered = json.dumps(summary, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))
    print(rendered, end="")


if __name__ == "__main__":
    main()
