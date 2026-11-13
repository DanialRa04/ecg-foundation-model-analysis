"""Analyze verified embedding evaluation tables across models."""

import argparse
from pathlib import Path
from common.analysis import run_analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        run_analysis(args.evaluation, args.output)
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")


if __name__ == "__main__":
    main()
