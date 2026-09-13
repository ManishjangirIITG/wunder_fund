from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SequenceSplit:
    train_seq_ix: np.ndarray
    validation_seq_ix: np.ndarray


def split_sequences(
    dataset: pd.DataFrame,
    validation_fraction: float = 0.20,
    seed: int = 42,
) -> SequenceSplit:
    """Deterministically split independent seq_ix values.

    No sequence can occur in both train and validation.
    """
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    sequence_ids = np.asarray(
        sorted(dataset["seq_ix"].unique()),
        dtype=np.int64,
    )
    if len(sequence_ids) < 2:
        raise ValueError("At least two sequences are required.")

    rng = np.random.default_rng(seed)
    shuffled = sequence_ids.copy()
    rng.shuffle(shuffled)

    n_val = max(1, int(round(len(shuffled) * validation_fraction)))
    n_val = min(n_val, len(shuffled) - 1)

    validation_seq_ix = np.sort(shuffled[:n_val])
    train_seq_ix = np.sort(shuffled[n_val:])

    return SequenceSplit(
        train_seq_ix=train_seq_ix,
        validation_seq_ix=validation_seq_ix,
    )


def make_next_state_samples(
    dataset: pd.DataFrame,
    sequence_ids: np.ndarray,
    lookback: int = 1,
    require_prediction_flag: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build X -> next-state y samples without crossing sequence boundaries.

    lookback=1:
        X = state[t]
        y = state[t+1]

    lookback=k:
        X = [state[t-k+1], ..., state[t]]
        y = state[t+1]
    """
    if lookback < 1:
        raise ValueError("lookback must be >= 1.")

    selected = dataset[dataset["seq_ix"].isin(sequence_ids)]
    feature_columns = list(dataset.columns[3:])
    n_features = len(feature_columns)

    x_parts = []
    y_parts = []

    for seq_ix, group in selected.groupby("seq_ix", sort=False):
        group = group.sort_values("step_in_seq")

        states = group[feature_columns].to_numpy(
            dtype=np.float32, copy=True
        )
        steps = group["step_in_seq"].to_numpy()
        need_prediction = group["need_prediction"].to_numpy(dtype=bool)

        if len(states) < lookback + 1:
            continue

        expected = np.arange(steps[0], steps[0] + len(steps))
        if not np.array_equal(steps, expected):
            raise ValueError(
                f"Sequence {seq_ix} has non-consecutive step_in_seq values."
            )

        for t in range(lookback - 1, len(states) - 1):
            if require_prediction_flag and not need_prediction[t]:
                continue

            x_parts.append(
                states[t - lookback + 1 : t + 1].reshape(-1)
            )
            y_parts.append(states[t + 1])

    if not x_parts:
        raise ValueError("No supervised samples were generated.")

    return (
        np.asarray(x_parts, dtype=np.float32),
        np.asarray(y_parts, dtype=np.float32),
    )
