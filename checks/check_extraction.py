"""Exercise batching and artifact checks using a deterministic small encoder."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import torch
from checks.check_preprocessing import fixture_dataset
from integrations.models import get_integration
from scripts.preprocess import run as preprocess
from scripts.extract_embeddings import run as extract, source_revision
from common.checkpoints import open_structure, read_tensor


def run_checks():
    root = Path(__file__).resolve().parents[1]
    assert source_revision(root / "checks") is None
    (root / "outputs").mkdir(exist_ok=True)
    with TemporaryDirectory(
        prefix="extraction_checks_", dir=root / "outputs"
    ) as directory:
        base = Path(directory).resolve()
        assert (root / "outputs").resolve() in base.parents
        dataset = fixture_dataset(base)
        source = base / "author_code"
        source.mkdir()
        checkpoint = base / "checkpoint.txt"
        checkpoint.write_text("synthetic encoder fixture")
        paths = base / "paths.json"
        paths.write_text(
            json.dumps(
                {
                    "paths": {
                        "dataset_root": str(dataset),
                        "checkpoint_root": str(base),
                        "external_source_root": str(base),
                    },
                    "models": {
                        "ECGFM-KED": {
                            "source": str(source),
                            "checkpoint": str(checkpoint),
                        }
                    },
                }
            )
        )
        config = root / "configs/experiment.json"
        preprocess(
            argparse.Namespace(
                config=config,
                paths=paths,
                models=["ECGFM-KED"],
                output=base / "prepared",
                limit=4,
                ids=None,
                scalers=None,
            )
        )
        actual = get_integration("ECGFM-KED")

        def builder(context):
            model = torch.nn.Linear(12, 3, bias=False)
            with torch.no_grad():
                model.weight.copy_(
                    torch.arange(36, dtype=torch.float32).reshape(3, 12) / 36
                )
            model.requires_grad_(False).eval()
            return (
                model,
                lambda values: model(values.mean(-1)),
                {"architecture": "synthetic linear encoder", "pooling": "time mean"},
            )

        fixture = SimpleNamespace(
            RECIPE=actual.RECIPE, __file__=__file__, build_extractor=builder
        )
        args = argparse.Namespace(
            config=config,
            paths=paths,
            model="ECGFM-KED",
            input=base / "prepared/ECGFM-KED",
            output=base / "batch1",
            checkpoint=None,
            source=None,
            model_config=None,
            batch_size=1,
            limit=None,
            device="cpu",
            trust_checkpoint=False,
        )
        with patch("scripts.extract_embeddings.get_integration", return_value=fixture):
            first = extract(args)
            assert first["seed"] == 42 and first["cpu_threads"] == 1
            assert first["upstream_revision"] is None
            args.output = base / "batch3"
            args.batch_size = 3
            second = extract(args)
            a = np.load(base / "batch1/embeddings.npy", allow_pickle=False)
            b = np.load(base / "batch3/embeddings.npy", allow_pickle=False)
            assert (
                first["status"] == second["status"] == "COMPLETE"
                and a.shape == (4, 3)
                and np.allclose(a, b, rtol=1e-6, atol=1e-6)
            )
            assert np.array_equal(
                np.load(base / "batch1/sample_ids.npy", allow_pickle=False),
                np.array(["1", "2", "3", "4"]),
            )
            signal_file = args.input / "signals.npy"
            original = signal_file.read_bytes()
            values = np.load(signal_file, allow_pickle=False)
            values[0, 0, 0] += 1
            np.save(signal_file, values, allow_pickle=False)
            args.output = base / "tampered"
            try:
                extract(args)
            except ValueError as error:
                assert "fingerprint" in str(error)
            else:
                raise AssertionError("Tampered inputs were accepted")
            signal_file.write_bytes(original)
            args.output = base / "batch1"
            try:
                extract(args)
            except FileExistsError:
                pass
            else:
                raise AssertionError("Output collision was accepted")
        archive_path = base / "synthetic_checkpoint.pt"
        view = torch.arange(100, dtype=torch.float32).reshape(10, 10)[2:8:2, 1:9:2]
        half = torch.arange(8, dtype=torch.bfloat16)
        torch.save(
            {
                "cfg": {},
                "model": {"view": view, "half": half},
                "optimizer": torch.zeros(1024),
            },
            archive_path,
        )
        try:
            open_structure(archive_path, trusted=False)
        except ValueError:
            pass
        else:
            raise AssertionError("Checkpoint metadata loaded without explicit trust")
        archive, prefix, _, descriptors = open_structure(archive_path, trusted=True)
        try:
            assert torch.equal(read_tensor(archive, prefix, descriptors["view"]), view)
            assert torch.equal(read_tensor(archive, prefix, descriptors["half"]), half)
            invalid = {**descriptors["view"], "offset": 10000}
            try:
                read_tensor(archive, prefix, invalid)
            except ValueError:
                pass
            else:
                raise AssertionError("Out-of-bounds tensor view was accepted")
        finally:
            archive.close()
    return {
        "status": "PASS",
        "checks": [
            "fixed encoder and final partial batch",
            "batch-size invariance",
            "record ID alignment",
            "input fingerprint enforcement",
            "output collision rejection",
            "trusted checkpoint gate",
            "streamed strided and bfloat16 tensors",
            "tensor storage bounds",
            "model revision does not inherit a containing repository commit",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(run_checks(), indent=2))
