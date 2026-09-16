"""Read official UEA TRAIN/TEST splits as lists of [time, channels] arrays."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.preprocessing import LabelEncoder


@dataclass
class UEADataset:
    name: str
    x_train: list[np.ndarray]
    y_train: np.ndarray
    x_test: list[np.ndarray]
    y_test: np.ndarray
    classes: np.ndarray
    label_encoder: LabelEncoder


def _to_float(token: str) -> float:
    token = token.strip()
    if token in {"", "?", "NaN", "nan", "None"}:
        return float("nan")
    return float(token)


def _parse_dimension(text: str, timestamps: bool) -> np.ndarray:
    text = text.strip()
    if not text:
        return np.empty((0,), dtype=np.float64)
    if timestamps:
        raise ValueError(
            "Use non-timestamped UEA .ts files; timestamped rows are not supported."
        )
    return np.asarray([_to_float(v) for v in text.split(",")], dtype=np.float64)


def _interpolate_nan(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return values
    finite = np.isfinite(values)
    if finite.all():
        return values
    if not finite.any():
        return np.zeros_like(values)
    indices = np.arange(values.size)
    return np.interp(indices, indices[finite], values[finite])


def _read_ts(path: Path) -> tuple[list[np.ndarray], list[str]]:
    metadata: dict[str, str] = {}
    data_started = False
    rows: list[np.ndarray] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8-sig") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if not data_started:
                if line.lower() == "@data":
                    data_started = True
                    continue
                if line.startswith("@"):
                    parts = line[1:].split(maxsplit=1)
                    metadata[parts[0].lower()] = parts[1] if len(parts) == 2 else ""
                continue

            timestamps = metadata.get("timestamps", "false").strip().lower() == "true"
            class_decl = metadata.get(
                "classlabel", metadata.get("targetlabel", "false")
            )
            has_label = class_decl.strip().lower().startswith("true")
            if not has_label:
                raise ValueError(f"Classification labels are required in {path}")
            fields = line.split(":")
            if has_label:
                if len(fields) < 2:
                    raise ValueError(f"Malformed labeled row at {path}:{line_no}")
                label = fields[-1].strip()
                dimension_fields = fields[:-1]
            else:
                label = "0"
                dimension_fields = fields

            dimensions = [
                _interpolate_nan(_parse_dimension(field, timestamps))
                for field in dimension_fields
            ]
            if not dimensions:
                raise ValueError(f"No dimensions at {path}:{line_no}")
            lengths = [dim.size for dim in dimensions]
            target_len = max(lengths)
            if target_len == 0:
                raise ValueError(f"Empty row at {path}:{line_no}")
            aligned = []
            for dim in dimensions:
                if dim.size == target_len:
                    aligned.append(dim)
                elif dim.size <= 1:
                    aligned.append(np.full(target_len, dim[0] if dim.size else 0.0))
                else:
                    old_grid = np.linspace(0.0, 1.0, dim.size)
                    new_grid = np.linspace(0.0, 1.0, target_len)
                    aligned.append(np.interp(new_grid, old_grid, dim))
            rows.append(np.stack(aligned, axis=1))  # [T, D]
            labels.append(label)

    if not data_started:
        raise ValueError(f"Missing @data section in {path}")
    if not rows:
        raise ValueError(f"No sequences in {path}")
    return rows, labels


def _resolve_paths(root: str | Path, dataset: str) -> tuple[Path, Path]:
    root = Path(root)
    candidates = [
        root / dataset,
        root,
    ]
    for directory in candidates:
        train = directory / f"{dataset}_TRAIN.ts"
        test = directory / f"{dataset}_TEST.ts"
        if train.exists() and test.exists():
            return train, test
    raise FileNotFoundError(
        f"Could not find {dataset}_TRAIN.ts and {dataset}_TEST.ts under {root}. "
        "Expected either ROOT/DATASET/ or ROOT/."
    )


def load_uea_dataset(root: str | Path, dataset: str) -> UEADataset:
    train_path, test_path = _resolve_paths(root, dataset)
    x_train, train_labels = _read_ts(train_path)
    x_test, test_labels = _read_ts(test_path)
    encoder = LabelEncoder().fit(train_labels)
    unknown = sorted(set(test_labels) - set(encoder.classes_))
    if unknown:
        raise ValueError(f"Test split contains labels absent from train: {unknown}")
    y_train = encoder.transform(train_labels)
    y_test = encoder.transform(test_labels)
    return UEADataset(
        name=dataset,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        classes=encoder.classes_,
        label_encoder=encoder,
    )
