"""Train all three activity heads and evaluate the official test split."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_recognition.evaluate import evaluate_all_models
from activity_recognition.train import train_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train MLP, MobileNetV2, and S3D activity heads.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "activity_cache" / "manifest.json",
    )
    parser.add_argument(
        "--feature-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "activity_cache" / "features",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "activity_models",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "output" / "activity_results",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite-features", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_all_models(
        args.manifest,
        args.feature_dir,
        args.model_dir,
        args.results_dir,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        seed=args.seed,
        device_name=args.device,
        overwrite_features=args.overwrite_features,
    )
    if not args.skip_evaluation:
        evaluate_all_models(
            args.manifest,
            args.feature_dir,
            args.model_dir,
            args.results_dir,
            device_name=args.device,
            overwrite_features=args.overwrite_features,
        )
        print(f"Results: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
