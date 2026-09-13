from __future__ import annotations

import sys
from pathlib import Path

# Allow execution as:
#   python experiments/run_persistence.py
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from experiments.models.persistence import PersistenceModel
from experiments.tracker import ExperimentConfig, ExperimentTracker
from utils import ScorerStepByStep


def main() -> None:
    package_root = Path(__file__).resolve().parents[1]
    dataset_path = package_root / "datasets" / "train.parquet"
    experiments_dir = package_root / "experiments"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Place the challenge dataset at datasets/train.parquet."
        )

    scorer = ScorerStepByStep(str(dataset_path))
    model = PersistenceModel()

    config = ExperimentConfig(
        experiment_name="baseline_001_persistence",
        model_name=model.name,
        model_version=model.version,
        dataset_path=str(dataset_path),
        description="Predict the next state using the current state.",
        parameters={},
    )

    tracker = ExperimentTracker(experiments_dir)
    run_id = tracker.start_run(config)

    print(f"Run ID: {run_id}")
    print(f"Model: {model.name} v{model.version}")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(scorer.dataset)}")
    print(f"Features: {scorer.dim}")

    try:
        results = scorer.score(model)

        metrics = {
            "mean_r2": float(results["mean_r2"]),
            "feature_r2": {
                feature: float(results[feature])
                for feature in scorer.features
            },
        }

        tracker.finish_run(
            run_id=run_id,
            metrics=metrics,
            status="COMPLETED",
            notes="First non-trivial forecasting baseline: persistence.",
        )

        print("\nResults:")
        print(f"Mean R² across all features: {results['mean_r2']:.6f}")
        print("\nR² for first 5 features:")
        for feature in scorer.features[:5]:
            print(f"  {feature}: {results[feature]:.6f}")

        print(f"\nExperiment record: experiments/runs/{run_id}/")
        print("Registry: experiments/registry.jsonl")

    except Exception as exc:
        tracker.finish_run(
            run_id=run_id,
            metrics={},
            status="FAILED",
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
