"""Create aggregate result tables and PNG/SVG/PDF scientific figures."""

import argparse
from pathlib import Path
from common.reporting import run_report


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument(
        "--settings", type=Path, default=root / "configs/reporting.json"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--example",
        action="store_true",
        help="Label every figure and the report as a generated-data example",
    )
    args = parser.parse_args()
    try:
        run_report(
            args.evaluation, args.analysis, args.settings, args.output, args.example
        )
    except (ValueError, KeyError, TypeError, OSError, RuntimeError) as error:
        parser.exit(1, f"Report failed: {error}\n")


if __name__ == "__main__":
    main()
