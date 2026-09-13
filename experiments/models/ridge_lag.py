from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
import pandas as pd

from utils import DataPoint


class RidgeLagModel:
    """Multivariate Ridge regression for one-step state forecasting.

    During training:
        [state(t-k+1), ..., state(t)] -> state(t+1)

    During inference:
        The model maintains a rolling history and resets on new seq_ix.
    """

    name = "ridge_lag"
    version = "1.1.0"

    def __init__(self, lookback: int = 1, alpha: float = 1.0) -> None:
        if lookback < 1:
            raise ValueError("lookback must be >= 1.")
        if alpha <= 0:
            raise ValueError("alpha must be > 0.")

        self.lookback = int(lookback)
        self.alpha = float(alpha)
        self.model = Ridge(alpha=self.alpha)

        self.n_features: int | None = None
        self.current_seq_ix: int | None = None
        self.history: list[np.ndarray] = []

    def _align_features(self, X):
        """Internal helper to dynamically pad columns for lookback divisibility."""
        n_cols = X.shape[1]
        remainder = n_cols % self.lookback
        
        if remainder != 0:
            # Truncate surplus columns so total columns are evenly divisible by lookback
            usable_cols = n_cols - remainder
            if hasattr(X, "iloc"):
                return X.iloc[:, :usable_cols]
            return X[:, :usable_cols]
        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeLagModel":
        # Automatically align features before reshaping or fitting
        X = self._align_features(X)

        inferred_dim = X.shape[1] // self.lookback

        # 3. Automatically truncate y so its feature dimension equals inferred_dim (F)
        if hasattr(y, "iloc"):
            y = y.iloc[:, :inferred_dim]
        else:
            y = y[:, :inferred_dim]

        
        if X.ndim != 2 or y.ndim != 2:
            raise ValueError("X and y must both be 2-D arrays.")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y row counts must match.")
        # if X.shape[1] % self.lookback != 0:
        #     raise ValueError("X dimension must be divisible by lookback.")

        self.n_features = inferred_dim
        self.model.fit(X, y)
        return self

    @property
    def is_fitted(self) -> bool:
        return self.n_features is not None and hasattr(self.model, "coef_")

    def reset_state(self) -> None:
        self.current_seq_ix = None
        self.history = []

    def predict(self, data_point: DataPoint) -> np.ndarray | None:

        if not data_point.need_prediction:
            return None
        if not self.is_fitted:
            raise RuntimeError("RidgeLagModel must be fitted before inference.")

        # Ensure input state matches the fitted single-step n_features dimension
        state = np.asarray(data_point.state[:self.n_features], dtype=np.float32)

        if state.shape != (self.n_features,):
            raise ValueError(
                f"Expected state shape ({self.n_features},), got {state.shape}."
            )

        if self.current_seq_ix != data_point.seq_ix:
            self.current_seq_ix = data_point.seq_ix
            self.history = []

        self.history.append(state.copy())

        # This padding is only relevant for lookback > 1 during the earliest
        # prediction point. The competition provides 100 warm-up steps, so
        # normal challenge inference has enough preceding history.
        if len(self.history) < self.lookback:
            window = (
                [self.history[0]] * (self.lookback - len(self.history))
                + self.history
            )
        else:
            window = self.history[-self.lookback:]

        X_input = np.concatenate(window).reshape(1, -1)
        return np.asarray(self.model.predict(X_input)[0], dtype=np.float64)

    def predict_batch(self, X: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predicts on a batch feature matrix X, automatically aligning feature dimensions."""
        if not self.is_fitted:
            raise RuntimeError("RidgeLagModel must be fitted before inference.")
        
        # Automatically truncate X from 32 columns down to 30 (matching fit dimension)
        X_aligned = self._align_features(X)
        return self.model.predict(X_aligned)
