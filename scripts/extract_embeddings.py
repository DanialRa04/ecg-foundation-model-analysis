"""Extract one model's aligned frozen embeddings in an isolated process."""

import argparse
from datetime import datetime, timezone
from pathlib import Path
import random
import subprocess
import numpy as np
from common.artifacts import (
    array_info,
    file_hash,
    load_ids,
    log_event,
    new_run,
    sequence_hash,
    versions,
    write_json,
)
from common.configuration import load_experiment, read_json
from common.paths import load_paths, model_assets
from integrations.models import get_integration


def checkpoint_hash(path):
    path = Path(path)
    if path.is_file():
        return {"file": path.name, "sha256": file_hash(path)}
    if not path.is_dir():
        raise FileNotFoundError("Checkpoint location is missing")
    files = sorted(file for file in path.rglob("*") if file.is_file())
    if not files:
        raise ValueError("Checkpoint directory is empty")
    return {
        "files": {file.relative_to(path).as_posix(): file_hash(file) for file in files}
    }


def source_revision(path):
    """Return a commit only for the configured model directory's own Git root."""
    path = Path(path).resolve()
    command = ["git", "-c", f"safe.directory={path.as_posix()}", "-C", str(path)]
    root = subprocess.run(
        [*command, "rev-parse", "--show-toplevel"], capture_output=True, text=True
    )
    if root.returncode or Path(root.stdout.strip()).resolve() != path:
        return None
    revision = subprocess.run(
        [*command, "rev-parse", "HEAD"], capture_output=True, text=True
    )
    return revision.stdout.strip() if revision.returncode == 0 else None


def run(args):
    import torch

    threads = getattr(args, "threads", 1)
    if threads < 1:
        raise ValueError("Thread count must be positive")
    torch.set_num_threads(threads)
    bundle = load_experiment(args.config)
    seed = getattr(args, "seed", None)
    if seed is None:
        seed = bundle["experiment"]["representation"].get("seed", 42)
    if not isinstance(seed, int) or isinstance(seed, bool) or not 0 <= seed < 2**32:
        raise ValueError("Extraction seed must be a 32-bit nonnegative integer")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    catalog = {model["name"]: model for model in bundle["models"]["models"]}
    if args.model not in bundle["experiment"]["model_ids"]:
        raise ValueError("Model is not selected in the experiment")
    model_spec = catalog[args.model]
    paths, overrides = load_paths(args.paths)
    assets = model_assets(model_spec, paths, overrides)
    if args.checkpoint:
        assets["checkpoint"] = args.checkpoint.resolve()
    if args.source:
        assets["source"] = args.source.resolve()
    if args.model_config:
        assets["config"] = args.model_config.resolve()
    if (
        args.checkpoint
        and not args.model_config
        and model_spec.get("checkpoint_config")
    ):
        assets["config"] = assets["checkpoint"].parent / model_spec["checkpoint_config"]
    if args.source and not args.model_config and model_spec.get("model_config"):
        assets["config"] = assets["source"] / model_spec["model_config"]
    if not assets["source"].is_dir() or not assets["checkpoint"].exists():
        raise FileNotFoundError(
            "Model code or checkpoint is missing; see docs/models.md"
        )
    prepared = args.input.resolve()
    meta = read_json(prepared / "metadata.json")
    recipe = read_json(prepared / "recipe.json")
    module = get_integration(args.model)
    if (
        meta.get("status") != "COMPLETE"
        or meta.get("model") != args.model
        or recipe != module.RECIPE
    ):
        raise ValueError(
            "Preprocessed input is incomplete or belongs to a different model recipe"
        )
    for filename, key in [
        ("signals.npy", "signals_sha256"),
        ("sample_ids.npy", "ids_file_sha256"),
        ("recipe.json", "recipe_sha256"),
    ]:
        if file_hash(prepared / filename) != meta[key]:
            raise ValueError(f"Input fingerprint mismatch: {filename}")
    if (
        "records_sha256" in meta
        and file_hash(prepared / "records.csv") != meta["records_sha256"]
    ):
        raise ValueError("Paired record metadata fingerprint mismatch")
    signals = array_info(prepared / "signals.npy", 3)
    ids = load_ids(prepared / "sample_ids.npy")
    if (
        len(ids) != len(signals)
        or list(signals.shape) != meta["shape"]
        or list(signals.shape[1:]) != model_spec["shape"]
        or sequence_hash(ids) != meta["record_order_sha256"]
    ):
        raise ValueError("Signal dimensions or recording order do not match metadata")
    if args.batch_size < 1:
        raise ValueError("Batch size must be positive")
    count = len(ids) if args.limit is None else min(args.limit, len(ids))
    if count < 1:
        raise ValueError("Limit must be positive")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output = new_run(
        args.output,
        [prepared, assets["source"], assets["checkpoint"], *assets["dependencies"]],
    )
    metadata = {
        "status": "RUNNING",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "representation": "final",
        "records": count,
        "record_order_sha256": sequence_hash(ids[:count]),
        "input_metadata_sha256": file_hash(prepared / "metadata.json"),
        "device": device,
        "batch_size": args.batch_size,
        "seed": seed,
        "cpu_threads": threads,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "trust_checkpoint": args.trust_checkpoint,
        "versions": versions(
            [
                "numpy",
                "torch",
                "transformers",
                "safetensors",
                "timm",
                "hydra-core",
                "wfdb",
            ]
        ),
    }
    write_json(output / "metadata.json", metadata)
    log_event(output, f"START {args.model}: {count} recordings, device={device}")
    write_json(
        output / "config.resolved.json",
        {"experiment": bundle["experiment"], "model": model_spec, "assets": assets},
    )
    try:
        print(f"{args.model}: fingerprinting checkpoint", flush=True)
        metadata["checkpoint"] = checkpoint_hash(assets["checkpoint"])
        if assets["config"]:
            metadata["model_config_sha256"] = file_hash(assets["config"])
        metadata["upstream_revision"] = source_revision(assets["source"])
        context = {
            **assets,
            "device": device,
            "output": output,
            "trust_checkpoint": args.trust_checkpoint,
        }
        print(f"{args.model}: loading local encoder", flush=True)
        model, forward, details = module.build_extractor(context)
        if model.training or any(
            parameter.requires_grad for parameter in model.parameters()
        ):
            raise ValueError("Encoder is not frozen in evaluation mode")
        metadata.update(details)
        metadata["parameter_dtypes"] = sorted(
            {str(parameter.dtype) for parameter in model.parameters()}
        )
        array = None
        with torch.inference_mode():
            for start in range(0, count, args.batch_size):
                end = min(start + args.batch_size, count)
                values = np.array(signals[start:end], dtype=np.float32, copy=True)
                if not np.isfinite(values).all():
                    raise ValueError("Input batch contains nonfinite values")
                result = forward(torch.from_numpy(values).to(device))
                result = result.detach().float().cpu().numpy()
                if (
                    result.ndim != 2
                    or len(result) != end - start
                    or result.shape[1] < 1
                    or not np.isfinite(result).all()
                ):
                    raise ValueError(
                        "Encoder must return finite [recordings, features] embeddings"
                    )
                if array is None:
                    array = np.lib.format.open_memmap(
                        output / "embeddings.npy",
                        mode="w+",
                        dtype=np.float32,
                        shape=(count, result.shape[1]),
                    )
                    metadata["shape"] = list(array.shape)
                if result.shape[1] != array.shape[1]:
                    raise ValueError("Embedding width changed between batches")
                array[start:end] = result
                print(f"{args.model}: extracted {end}/{count}", flush=True)
                log_event(output, f"extracted {end}/{count}")
        array.flush()
        del array
        np.save(output / "sample_ids.npy", ids[:count], allow_pickle=False)
        # Preserve the paired selection table, including when a smoke run limits rows.
        import csv
        from common.artifacts import save_records

        with (prepared / "records.csv").open(encoding="utf-8", newline="") as stream:
            records = list(csv.DictReader(stream))
        if [row["sample_id"] for row in records] != ids.tolist():
            raise ValueError("Paired record table is misaligned")
        save_records(output / "records.csv", records[:count])
        metadata.update(
            status="COMPLETE",
            embeddings_sha256=file_hash(output / "embeddings.npy"),
            ids_file_sha256=file_hash(output / "sample_ids.npy"),
            records_sha256=file_hash(output / "records.csv"),
            integration_sha256=file_hash(module.__file__),
            builder_sha256=file_hash(
                Path(__file__).resolve().parents[1] / "integrations/models/builders.py"
            ),
            extraction_runner_sha256=file_hash(__file__),
            artifact_io_sha256=file_hash(
                Path(__file__).resolve().parents[1] / "common/artifacts.py"
            ),
        )
    except Exception as error:
        metadata.update(
            status="FAILED", error_type=type(error).__name__, error=str(error)
        )
        raise
    finally:
        write_json(output / "metadata.json", metadata)
        log_event(output, metadata["status"])
    return metadata


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "configs/experiment.json")
    parser.add_argument("--paths", type=Path, default=root / "configs/paths.local.json")
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--input", type=Path, required=True, help="One model preprocessing directory"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--model-config", type=Path)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device")
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU worker threads for this extraction process",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the extraction seed in the experiment configuration",
    )
    parser.add_argument(
        "--trust-checkpoint",
        action="store_true",
        help="Permit pickle-based loading of locally trusted author checkpoints",
    )
    args = parser.parse_args()
    try:
        run(args)
    except Exception as error:
        parser.exit(1, f"Extraction failed ({type(error).__name__}): {error}\n")


if __name__ == "__main__":
    main()
