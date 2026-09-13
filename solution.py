from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np

from utils import DataPoint


class PredictionModel:
    """Competition inference wrapper for the fitted RidgeLag artifact."""

    def __init__(self) -> None:
        artifact_path = Path(__file__).resolve().parent / "artifacts" / "ridge_lag1.joblib"

        if not artifact_path.exists():
            raise FileNotFoundError(
                f"Missing trained model artifact: {artifact_path}. "
                "Run the training/export step before packaging the submission."
            )

        self.model = joblib.load(artifact_path)

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        return self.model.predict(data_point)
