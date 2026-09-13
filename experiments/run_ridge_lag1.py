from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import r2_score

from experiments.data.split import make_next_state_samples, split_sequences
from experiments.models.ridge_lag import RidgeLagModel
from experiments.tracker import ExperimentConfig, ExperimentTracker
from utils import ScorerStepByStep


def main() -> None:
    dataset_path = ROOT / "datasets" / "train.parquet"
    experiments_dir = ROOT / "experiments"

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    scorer = ScorerStepByStep(str(dataset_path))
    split = split_sequences(
        scorer.dataset,
        validation_fraction=0.20,
        seed=42,
    )

    X_train, y_train = make_next_state_samples(
        scorer.dataset,
        split.train_seq_ix,
        lookback=1,
        require_prediction_flag=True,
    )
    X_val, y_val = make_next_state_samples(
        scorer.dataset,
        split.validation_seq_ix,
        lookback=1,
        require_prediction_flag=True,
    )

    model = RidgeLagModel(lookback=1, alpha=1.0)

    tracker = ExperimentTracker(experiments_dir)
    config = ExperimentConfig(
        experiment_name="baseline_002_ridge_lag1",
        model_name=model.name,
        model_version=model.version,
        dataset_path=str(dataset_path),
        description="Current 32-D state -> next 32-D state using multivariate Ridge.",
        parameters={
            "lookback": 1,
            "alpha": 1.0,
            "validation_fraction": 0.20,
            "split_seed": 42,
            "require_prediction_flag": True,
        },
    )
    run_id = tracker.start_run(config)

    try:
        model.fit(X_train, y_train)
        predictions = model.model.predict(X_val)

        feature_r2 = {
            feature: float(
                r2_score(y_val[:, ix], predictions[:, ix])
            )
            for ix, feature in enumerate(scorer.features)
        }

        metrics = {
            "mean_r2": float(np.mean(list(feature_r2.values()))),
            "feature_r2": feature_r2,
            "train_samples": int(len(X_train)),
            "validation_samples": int(len(X_val)),
            "train_sequences": int(len(split.train_seq_ix)),
            "validation_sequences": int(len(split.validation_seq_ix)),
        }

        tracker.finish_run(
            run_id,
            metrics,
            notes="Leak-free sequence-held-out validation.",
        )

        print(f"Train sequences: {len(split.train_seq_ix)}")
        print(f"Validation sequences: {len(split.validation_seq_ix)}")
        print(f"Training samples: {X_train.shape}")
        print(f"Validation samples: {X_val.shape}")
        print("\nValidation results:")
        print(f"Mean R² across all features: {metrics['mean_r2']:.6f}")

        print("\nR² for first 5 features:")
        for feature in scorer.features[:5]:
            print(f"  {feature}: {feature_r2[feature]:.6f}")

        print(f"\nExperiment record: experiments/runs/{run_id}/")

    except Exception as exc:
        tracker.finish_run(
            run_id,
            {},
            status="FAILED",
            notes=f"{type(exc).__name__}: {exc}",
        )
        raise


if __name__ == "__main__":
    main()
