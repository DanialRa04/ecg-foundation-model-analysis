"""Review Git-visible files for the code-and-documentation publication boundary."""

import json
from pathlib import Path, PurePosixPath
import subprocess


def blocked_files(paths):
    blocked = []
    placeholders = {"data", "checkpoints", "external", "results", "figures"}
    source_dirs = {"common", "integrations", "scripts", "checks"}
    root_files = {
        ".gitignore",
        ".gitattributes",
        "readme.md",
        "reproducing.md",
        "third_party_notices.md",
        "contributing.md",
        "citation.cff",
        "license",
        "license.md",
        "license.txt",
        "notice",
        "notice.md",
    }
    for name in sorted(set(paths)):
        path = PurePosixPath(name.replace("\\", "/"))
        parts = path.parts
        allowed = False
        if path.is_absolute() or ".." in parts or not parts:
            blocked.append(name)
            continue
        if len(parts) == 1:
            allowed = path.name.lower() in root_files or (
                path.name.startswith("requirements-") and path.suffix == ".txt"
            )
        elif parts[0] in placeholders:
            allowed = len(parts) == 2 and parts[1] == "README.md"
        elif parts[0] in source_dirs:
            allowed = path.suffix == ".py" or path.name == "README.md"
        elif parts[0] == "configs":
            allowed = path.suffix == ".json" and "local" not in path.name.split(".")
        elif parts[0] == "docs":
            allowed = path.suffix == ".md" or path.as_posix() in {
                "docs/descriptor_definitions.csv",
                "docs/kernel_manifest.json",
            }
        elif parts[0] == ".github":
            allowed = path.suffix in {".yml", ".yaml", ".md"}
        if not allowed:
            blocked.append(name)
    return blocked


def run_checks():
    assert not blocked_files(
        [
            "README.md",
            "LICENSE",
            "CITATION.cff",
            "common/metrics.py",
            "scripts/render_report.py",
            "configs/experiment.json",
            "docs/descriptor_definitions.csv",
            "results/README.md",
            "figures/README.md",
        ]
    )
    generated = [
        "outputs/run/metadata.json",
        "results/metrics.csv",
        "figures/performance.png",
        "docs/performance.pdf",
        "metrics.csv",
        "common/embeddings.npy",
    ]
    assert set(blocked_files(generated)) == set(generated)
    local = [
        "data/record.dat",
        "checkpoints/model.pt",
        "external/model.py",
        "private/review.md",
        "configs/paths.local.json",
    ]
    assert set(blocked_files(local)) == set(local)
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    paths = [name for name in result.stdout.split("\0") if name]
    blocked = blocked_files(paths)
    return {
        "status": "FAIL" if blocked else "PASS",
        "files_reviewed": len(set(paths)),
        "blocked_files": blocked,
        "checks": [
            "source and documentation paths accepted",
            "generated results rejected including Git-tracked paths",
            "local datasets weights external code and settings rejected",
            "current Git-visible contents reviewed",
        ],
    }


if __name__ == "__main__":
    result = run_checks()
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["status"] == "PASS" else 1)
