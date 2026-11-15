"""Fixed linear and neural probes on already standardized feature matrices."""

import random
import warnings
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from common.metrics import ranking_metrics


def linear_probe(train, targets, evaluate, settings, seed, class_count):
    if (
        settings["estimator"] != "logistic_regression"
        or settings["solver"] != "liblinear"
        or settings["multiclass_strategy"] != "one_vs_rest"
    ):
        raise ValueError("Unsupported linear probe configuration")
    base = LogisticRegression(
        C=settings["C"],
        penalty=settings["penalty"],
        solver=settings["solver"],
        class_weight=settings["class_weight"],
        max_iter=settings["maximum_iterations"],
        random_state=seed,
    )
    model = base if class_count == 2 else OneVsRestClassifier(base, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(train, targets)
    if model.classes_.tolist() != list(range(class_count)):
        raise ValueError("Probe did not learn every retained class")
    predictions = {
        name: model.predict_proba(values) if len(values) else np.empty((0, class_count))
        for name, values in evaluate.items()
    }
    return predictions, {
        "converged": not any(
            issubclass(item.category, ConvergenceWarning) for item in caught
        ),
        "warnings": [str(item.message) for item in caught],
        "selection": "fixed settings; no hyperparameter search",
    }


def neural_probe(
    train,
    targets,
    validation_targets,
    evaluate,
    settings,
    seed,
    class_count,
    binary,
    device="cpu",
):
    import torch
    from torch import nn

    if (
        settings["normalization"] != "LayerNorm"
        or settings["activation"] != "GELU"
        or settings["optimizer"] != "AdamW"
    ):
        raise ValueError("Unsupported neural probe configuration")
    if settings["selection_metric"] != "validation_average_precision":
        raise ValueError("Unsupported checkpoint selection metric")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    layers, previous = [], train.shape[1]
    for width in settings["hidden_dims"]:
        layers.extend(
            [
                nn.Linear(previous, width),
                nn.LayerNorm(width),
                nn.GELU(),
                nn.Dropout(settings["dropout"]),
            ]
        )
        previous = width
    layers.append(nn.Linear(previous, 1 if binary else class_count))
    model = nn.Sequential(*layers).to(device)
    counts = np.bincount(targets, minlength=class_count)
    if np.any(counts == 0):
        raise ValueError("Training lacks a retained class")
    if binary:
        weight = float(counts[0] / counts[1])
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight, device=device))
    else:
        weight = 1 / counts.astype(float)
        weight /= weight.mean()
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(weight, dtype=torch.float32, device=device)
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings["learning_rate"],
        weight_decay=settings["weight_decay"],
    )
    features = torch.as_tensor(np.asarray(train, dtype=np.float32))
    labels = torch.as_tensor(targets, dtype=torch.float32 if binary else torch.long)
    generator = torch.Generator().manual_seed(seed)

    def predict(values):
        if not len(values):
            return np.empty((0, class_count))
        result = []
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(values), settings["batch_size"]):
                logits = model(
                    torch.as_tensor(
                        np.asarray(
                            values[start : start + settings["batch_size"]],
                            dtype=np.float32,
                        ),
                        device=device,
                    )
                )
                if binary:
                    positive = logits[:, 0].sigmoid()
                    probabilities = torch.stack([1 - positive, positive], dim=1)
                else:
                    probabilities = logits.softmax(1)
                result.append(probabilities.cpu().numpy())
        return np.concatenate(result)

    best, stale, best_state, best_epoch = -np.inf, 0, None, None
    history = []
    for epoch in range(1, settings["maximum_epochs"] + 1):
        model.train()
        loss_sum = 0.0
        order = torch.randperm(len(features), generator=generator)
        for start in range(0, len(features), settings["batch_size"]):
            selected = order[start : start + settings["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(features[selected].to(device))
            loss = criterion(
                logits[:, 0] if binary else logits, labels[selected].to(device)
            )
            if not torch.isfinite(loss):
                raise ValueError("Neural probe loss became nonfinite")
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(selected)
        score = ranking_metrics(
            validation_targets, predict(evaluate["validation"]), class_count, binary
        )["average_precision"]
        if score is None:
            raise ValueError("Validation average precision is undefined")
        history.append(
            {
                "epoch": epoch,
                "training_loss": loss_sum / len(features),
                "validation_average_precision": score,
            }
        )
        if score > best + settings["minimum_improvement"]:
            best, stale, best_epoch = score, 0, epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise ValueError("No validation-selected probe checkpoint")
    model.load_state_dict(best_state)
    return {name: predict(values) for name, values in evaluate.items()}, {
        "best_epoch": best_epoch,
        "epochs": len(history),
        "best_validation_average_precision": best,
        "history": history,
        "loss_weight": weight if binary else weight.tolist(),
        "selection": "validation average precision only",
    }
