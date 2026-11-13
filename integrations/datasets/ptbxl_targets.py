"""PTB-XL task definitions from metadata and its public SCP taxonomy."""

import ast
from collections import Counter
import math
import numpy as np
from common.tables import read_table


def code_keys(text):
    if not isinstance(text, str) or len(text) > 65536:
        raise ValueError("Invalid SCP code field")
    try:
        values = ast.literal_eval(text)
    except (ValueError, SyntaxError, RecursionError) as error:
        raise ValueError("Malformed SCP code dictionary") from error
    if not isinstance(values, dict) or any(
        not isinstance(key, str) or not key.strip() for key in values
    ):
        raise ValueError("SCP codes must be a dictionary with nonempty string keys")
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        for value in values.values()
    ):
        raise ValueError("SCP confidence values must be finite numbers")
    # A key with confidence zero still counts as present in this experiment.
    return set(values)


def build_targets(rows, catalog, selected, taxonomy_path):
    taxonomy = read_table(taxonomy_path)
    if not taxonomy:
        raise ValueError("SCP taxonomy is empty")
    code_column = next(iter(taxonomy[0]))
    codes = [row[code_column] for row in taxonomy]
    if any(not code for code in codes) or len(codes) != len(set(codes)):
        raise ValueError("Taxonomy codes must be nonempty and unique")
    available = set(codes)
    presence = [code_keys(row["scp_codes"]) for row in rows]
    training = np.asarray([row["split"] == "train" for row in rows])
    targets, definitions = {}, {}
    tasks = {task["id"]: task for task in catalog["tasks"]}
    for name in selected:
        task = tasks[name]
        kind = task["target_type"]
        if kind == "multiclass_column":
            raw = [row[task["column"]].strip() for row in rows]
            support = Counter(
                value for value, train in zip(raw, training) if train and value
            )
            classes = sorted(
                value
                for value, count in support.items()
                if count >= task["minimum_full_training_support"]
            )
            mapping = {value: i for i, value in enumerate(classes)}
            targets[name] = np.array(
                [mapping.get(value, -1) for value in raw], dtype=np.int64
            )
            definitions[name] = {
                **task,
                "kind": "multiclass",
                "classes": classes,
                "full_training_support": dict(sorted(support.items())),
            }
        else:
            selected_codes = set(task.get("codes", []))
            if kind in {"binary_union", "binary_taxonomy"}:
                column = task["taxonomy_column"]
                if column not in taxonomy[0]:
                    raise ValueError("Required taxonomy column is absent")
                mapped = {
                    row[code_column]
                    for row in taxonomy
                    if row[column] in task["taxonomy_values"]
                }
                if not mapped:
                    raise ValueError(f"No taxonomy codes found for {name}")
                selected_codes |= mapped
            if not selected_codes or not selected_codes <= available:
                raise ValueError(
                    f"Task {name} contains unknown or empty taxonomy codes"
                )
            targets[name] = np.array(
                [bool(keys & selected_codes) for keys in presence], dtype=np.int64
            )
            definitions[name] = {
                **task,
                "kind": "binary",
                "classes": ["negative", "positive"],
                "resolved_codes": sorted(selected_codes),
            }
    return targets, definitions
