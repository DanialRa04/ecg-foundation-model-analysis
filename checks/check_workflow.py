"""Exercise the connected workflow with generated ECGs and tiny frozen encoders."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import torch
import wfdb
from common.analysis import run_analysis
from common.artifacts import file_hash, write_json
from common.configuration import load_experiment, read_json
from common.evaluation import run_evaluation
from common.experiment_data import load_manifest, verified_stage
from common.reporting import run_report
from common.tables import read_table, write_table
from integrations.datasets.ptbxl import CANONICAL_LEADS
from integrations.models import get_integration
from scripts.prepare_manifest import run as prepare_manifest
from scripts.preprocess import run as preprocess
from scripts.extract_embeddings import run as extract


def fixture(base):
    root = Path(__file__).resolve().parents[1]
    bundle = load_experiment(root / "configs/experiment.json")
    bundle["experiment"]["model_ids"] = ["ECGFM-KED", "CLEF", "MERL"]
    bundle["experiment"]["task_ids"] = ["atrial_fibrillation", "st_elevation"]
    bundle["experiment"]["analysis"]["bootstrap_replicates"] = 19
    bundle["probes"]["probes"]["ann"].update(
        hidden_dims=[8, 4], maximum_epochs=2, patience=1, batch_size=8
    )
    dataset = base / "dataset"
    dataset.mkdir()
    rows = []
    for i in range(24):
        fold = 1 if i < 16 else 9 if i < 20 else 10
        codes = {"AFIB": 0.0} if i % 2 else {"SR": 100.0}
        if i % 4 < 2:
            codes["STE_"] = 100.0
        row = {
            "ecg_id": str(i + 1),
            "patient_id": str(i // 2 + 1),
            "strat_fold": fold,
            "scp_codes": repr(codes),
            "heart_axis": "",
            "sex": str(i % 2),
            "age": str(25 + i),
            "filename_lr": f"records100/record{i + 1}",
            "filename_hr": f"records500/record{i + 1}",
        }
        rows.append(row)
        for hz in (100, 500):
            directory = dataset / f"records{hz}"
            directory.mkdir(exist_ok=True)
            time = np.arange(hz * 10) / hz
            signal = np.stack(
                [
                    0.02 * i
                    + 0.01 * lead
                    + (1 + 0.03 * i)
                    * np.sin(2 * np.pi * (1 + 0.03 * i) * time + lead * 0.07)
                    + 0.1 * np.cos(2 * np.pi * (7 + 0.01 * i) * time)
                    for lead in range(12)
                ],
                axis=1,
            )
            wfdb.wrsamp(
                f"record{i + 1}",
                fs=hz,
                units=["mV"] * 12,
                sig_name=CANONICAL_LEADS,
                p_signal=signal,
                fmt=["16"] * 12,
                write_dir=str(directory),
            )
    write_table(dataset / "ptbxl_database.csv", list(rows[0]), rows)
    write_table(
        dataset / "scp_statements.csv",
        ["code", "diagnostic_class"],
        [{"code": code, "diagnostic_class": ""} for code in ("AFIB", "SR", "STE_")],
    )
    configs = base / "configs"
    configs.mkdir()
    for name in ("models", "tasks", "descriptors", "probes"):
        write_json(configs / f"{name}.json", bundle[name])
    write_json(configs / "experiment.json", bundle["experiment"])
    author = base / "fixture_encoder"
    author.mkdir()
    checkpoint = base / "fixture_checkpoint.txt"
    checkpoint.write_text("generated encoder; no pretrained weights")
    paths = {
        "paths": {
            "dataset_root": str(dataset),
            "checkpoint_root": str(base),
            "external_source_root": str(base),
        },
        "models": {
            name: {"source": str(author), "checkpoint": str(checkpoint)}
            for name in bundle["experiment"]["model_ids"]
        },
    }
    write_json(configs / "paths.json", paths)
    settings = read_json(root / "configs/reporting.json")
    settings["figures"].update(formats=["png"], dpi=72)
    write_json(configs / "reporting.json", settings)
    return bundle, configs, dataset


def fixture_builder(context):
    model = torch.nn.Identity().requires_grad_(False).eval()

    def forward(values):
        features = torch.stack(
            [
                values.mean((1, 2)),
                values.square().mean((1, 2)),
                values[:, 0, 0],
                values[:, 0, 71],
                values[:, :, -1].mean(1),
                values.std((1, 2), correction=0),
            ],
            dim=1,
        )
        return model(features)

    return (
        model,
        forward,
        {
            "architecture": "generated frozen encoder fixture",
            "pooling": "deterministic signal summaries",
        },
    )


def run_checks():
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[1]
    (root / "outputs").mkdir(exist_ok=True)
    passed = []
    with TemporaryDirectory(
        prefix="workflow_checks_", dir=root / "outputs"
    ) as directory:
        base = Path(directory).resolve()
        assert (root / "outputs").resolve() in base.parents
        bundle, configs, dataset = fixture(base)
        originals = {
            path: file_hash(path) for path in dataset.rglob("*") if path.is_file()
        }
        prepare_manifest(
            argparse.Namespace(
                config=configs / "experiment.json",
                paths=configs / "paths.json",
                output=base / "manifest",
            )
        )
        manifest = load_manifest(base / "manifest")
        assert len(manifest["ids"]) == 24 and len(manifest["tasks"]) == 2
        passed.append("generated WFDB metadata to task and patient manifest")
        preprocess(
            argparse.Namespace(
                config=configs / "experiment.json",
                paths=configs / "paths.json",
                models=["all"],
                output=base / "preprocessed",
                limit=None,
                ids=None,
                scalers=None,
            )
        )
        prepared_meta = read_json(base / "preprocessed/metadata.json")
        assert prepared_meta["status"] == "COMPLETE"
        scalers = read_json(base / "preprocessed/scalers.json")
        assert (
            scalers["CLEF"]["fit_records"] == scalers["ECGFM-KED"]["fit_records"] == 16
        )
        passed.append("model recipes and training scalers produce extraction inputs")
        embedding_paths = []
        for name in bundle["experiment"]["model_ids"]:
            module = get_integration(name)
            output = base / "embeddings" / name
            args = argparse.Namespace(
                config=configs / "experiment.json",
                paths=configs / "paths.json",
                model=name,
                input=base / "preprocessed" / name,
                output=output,
                checkpoint=None,
                source=None,
                model_config=None,
                batch_size=5,
                limit=None,
                device="cpu",
                threads=1,
                seed=42,
                trust_checkpoint=False,
            )
            with patch.object(module, "build_extractor", side_effect=fixture_builder):
                result = extract(args)
            assert result["status"] == "COMPLETE" and result["records"] == 24
            assert result["upstream_revision"] is None
            embedding_paths.append(output)
        passed.append("preprocessed signals pass through the frozen extraction runner")
        evaluated = run_evaluation(
            base / "manifest", embedding_paths, base / "evaluation"
        )
        assert (
            evaluated["status"] == "COMPLETE"
            and evaluated["fits"] == 48
            and evaluated["skipped_fits"] == 0
        )
        passed.append("extracted embeddings align with manifests and train both probes")
        analyzed = run_analysis(base / "evaluation", base / "analysis")
        assert analyzed["status"] == "COMPLETE"
        passed.append("evaluation tables join into configured model-level analysis")
        report = run_report(
            base / "evaluation",
            base / "analysis",
            configs / "reporting.json",
            base / "report",
            example=True,
        )
        assert (
            report["status"] == "COMPLETE"
            and report["data_kind"] == "generated_example"
        )
        verified_stage(
            base / "report", ["report.md", "tables/performance.csv", "figures.json"]
        )
        assert len(read_table(base / "report/tables/performance.csv")) == 6
        passed.append("analysis and evaluation produce a linked local report")
        assert originals == {path: file_hash(path) for path in originals}
        passed.append(
            "generated source recordings remain unchanged throughout the chain"
        )
    return {
        "status": "PASS",
        "checks": passed,
        "scope": "generated signals and fixture encoders; production stages and model preprocessing recipes",
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
