"""Check descriptor calculations using analytical synthetic cases."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from common import descriptor_math


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def calculate(values: np.ndarray) -> dict:
    return descriptor_math.spectral_and_collapse_descriptors(
        values, variance_tolerance=1e-8, rankme_epsilon=1e-7
    )


def run_checks() -> dict:
    checks = []
    isotropic = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
    reference = calculate(isotropic)
    for name in ("effective_rank", "participation_ratio", "stable_rank", "rankme"):
        require(math.isclose(reference[name], 2.0, rel_tol=1e-10), name)
    require(math.isclose(reference["spectral_entropy"], math.log(2.0), rel_tol=1e-10), "entropy")
    require(math.isclose(reference["centered_mean_pairwise_cosine_similarity"], -1.0 / 3.0, rel_tol=1e-10), "pairwise cosine")
    checks.append("analytical isotropic geometry")

    rank_one = calculate(np.array([[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]))
    for name in ("effective_rank", "participation_ratio", "stable_rank"):
        require(math.isclose(rank_one[name], 1.0, rel_tol=1e-10), name)
    require(rank_one["near_zero_variance_dimension_count"] == 1, "inactive dimension")
    checks.append("analytical rank-one geometry")

    theta = 0.37
    rotation = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])
    transformed = calculate(3.5 * isotropic @ rotation)
    invariant_names = ("effective_rank", "participation_ratio", "stable_rank", "spectral_entropy", "largest_spectral_component_fraction", "rankme")
    for name in invariant_names:
        require(math.isclose(reference[name], transformed[name], rel_tol=1e-10, abs_tol=1e-12), name)
    checks.append("selected scale and orthogonal invariants")

    translated = calculate(isotropic + np.array([3.0, -2.0]))
    for name in ("effective_rank", "participation_ratio", "stable_rank", "spectral_entropy"):
        require(math.isclose(reference[name], translated[name], rel_tol=1e-10, abs_tol=1e-12), name)
    require(not math.isclose(reference["rankme"], translated["rankme"], rel_tol=1e-6), "uncentered RankMe distinction")
    checks.append("centered geometry versus uncentered RankMe")

    for invalid in (np.empty((0, 2)), np.array([[1.0, float("nan")], [-1.0, 0.0]]), np.ones((4, 2))):
        try:
            calculate(invalid)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid or zero-centered-energy input was accepted")
    checks.append("invalid and degenerate input rejection")

    root = Path(__file__).resolve().parents[1]
    record = json.loads((root / "docs/kernel_manifest.json").read_text(encoding="utf-8"))
    item = next(row for row in record["files"] if row["path"] == "common/descriptor_math.py")
    actual_hash = hashlib.sha256(Path(descriptor_math.__file__).read_bytes()).hexdigest()
    require(actual_hash == item["sha256"], "Descriptor kernel differs from its recorded hash")
    checks.append("kernel integrity")
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "status": "PASS", "checks": checks,
            "python": platform.python_version(), "numpy": np.__version__, "kernel_sha256": actual_hash,
            "scope": "synthetic analytical checks"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional new JSON output; existing files are never overwritten")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    result = run_checks()
    rendered = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
