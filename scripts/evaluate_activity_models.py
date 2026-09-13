"""Evaluate saved activity heads without retraining."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from activity_recognition.evaluate import evaluate_all_models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved activity models.")
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
    parser.add_argument("--device", default="auto")
    parser.add_argument("--latency-samples", type=int, default=20)
    parser.add_argument("--overwrite-features", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_all_models(
        args.manifest,
        args.feature_dir,
        args.model_dir,
        args.results_dir,
        device_name=args.device,
        latency_samples=args.latency_samples,
        overwrite_features=args.overwrite_features,
    )
    print(f"Results: {args.results_dir.resolve()}")


if __name__ == "__main__":
    main()
