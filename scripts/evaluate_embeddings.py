"""Evaluate complete frozen embedding cohorts using a shared task manifest."""

import argparse
from pathlib import Path
from common.evaluation import run_evaluation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    args = parser.parse_args()
    try:
        run_evaluation(
            args.manifest, args.embeddings, args.output, args.device, args.threads
        )
    except (ValueError, KeyError, OSError, TypeError, RuntimeError) as error:
        parser.exit(1, f"Evaluation failed: {error}\n")


if __name__ == "__main__":
    main()
