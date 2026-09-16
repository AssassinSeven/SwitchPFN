"""Run SwitchPFN on one official UEA train/test split."""

import argparse
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root", default="./dataset/UEA")
    parser.add_argument("--dataset", required=True)
    parser.add_argument(
        "--checkpoint", default=None, help="Local TabPFN-v3 classifier checkpoint"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--output", default="./results")
    args = parser.parse_args()

    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "1"

    import numpy as np
    from data_provider.data_loader import load_uea_dataset
    from models.SwitchPFN import Model
    from sklearn.metrics import accuracy_score

    data = load_uea_dataset(args.data_root, args.dataset)
    model = Model(
        checkpoint=args.checkpoint, device=args.device, random_state=args.seed
    )
    start = time.perf_counter()
    model.fit(data.x_train, data.y_train)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    probabilities = model.predict_proba(data.x_test)
    prediction = model.classes_[probabilities.argmax(axis=1)]
    metrics = {
        "dataset": args.dataset,
        "seed": args.seed,
        "accuracy": float(accuracy_score(data.y_test, prediction)),
        "fit_seconds": fit_seconds,
        "predict_seconds_including_features": time.perf_counter() - start,
    }
    output = Path(args.output) / args.dataset / str(args.seed)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "predictions.npz",
        y_true=data.classes[data.y_test],
        y_pred=data.classes[prediction],
        probabilities=probabilities,
        classes=data.classes[model.classes_],
    )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    print(f"Saved to {output}")


if __name__ == "__main__":
    main()
