from __future__ import annotations

import numpy as np

from utils import DataPoint


class PersistenceModel:
    """One-step persistence baseline.

    Forecast:
        x[t+1] = x[t]

    This is intentionally simple and deterministic. It is useful as the first
    post-baseline benchmark because many time-series processes exhibit
    short-term persistence.
    """

    name = "persistence"
    version = "1.0.0"

    def __init__(self) -> None:
        self.current_seq_ix: int | None = None

    def predict(self, data_point: DataPoint) -> np.ndarray | None:
        # The scorer calls predict for every row. No prediction is allowed
        # when the challenge marks need_prediction=False.
        if not data_point.need_prediction:
            return None

        # No trainable state is required, but keeping the sequence boundary
        # explicit makes the model interface safe for future stateful models.
        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix

        return data_point.state.copy()
