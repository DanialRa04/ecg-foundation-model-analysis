"""Small explicit CSV helpers for local experiment tables."""

import csv
import math
from numbers import Real
from pathlib import Path


def read_table(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(set(reader.fieldnames)) != len(
            reader.fieldnames
        ):
            raise ValueError("Table needs unique column names")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("Table row width differs from its header")
    return rows


def write_table(path, columns, rows):
    with Path(path).open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: ""
                    if value is None
                    or (isinstance(value, Real) and not math.isfinite(value))
                    else value
                    for key, value in row.items()
                }
            )


def unique_index(rows, keys):
    result = {}
    for row in rows:
        key = tuple(row[name] for name in keys)
        if key in result:
            raise ValueError(f"Duplicate table key: {key}")
        result[key] = row
    return result
