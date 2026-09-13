from __future__ import annotations

import numpy as np

from utils import DataPoint
from experiments.models.persistence import PersistenceModel


class PredictionModel:
    """Competition submission wrapper.

    Keep this class stable. Individual model implementations live under
    experiments/models so experiments and the final submission share the
    exact same forecasting logic.
    """

    def __init__(self) -> None:
        self.model = PersistenceModel()

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        return self.model.predict(data_point)
