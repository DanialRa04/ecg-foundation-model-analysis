# Model integration

Model loading and embedding extraction live here. Each integration follows its documented [preprocessing recipe](../../docs/preprocessing/README.md). Checkpoints are obtained using [models.md](../../docs/models.md).

Each model module exposes `RECIPE`, `preprocess`, and `build_extractor`. Shared loading and model-family helpers are in `builders.py`. Run one model per extraction process to keep author imports and environments separate.
