import numpy as np
import pandas as pd

from src.models.train_models import (
    CLASS_ORDER,
    poisson_outcome_probabilities,
    temporal_split,
)


def test_poisson_probabilities_are_normalized_and_ordered():
    probabilities, top_scores, matrices = poisson_outcome_probabilities(
        np.array([3.0, 0.8]), np.array([0.5, 1.7]), max_goals=9
    )
    assert probabilities.shape == (2, 3)
    assert list(CLASS_ORDER) == ["A", "D", "H"]
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all(probabilities >= 0)
    assert probabilities[0, CLASS_ORDER.index("H")] > probabilities[0, CLASS_ORDER.index("A")]
    assert top_scores[0, 0] > top_scores[0, 1]
    assert np.allclose(matrices.sum(axis=(1, 2)), 1.0)


def test_temporal_split_keeps_calendar_dates_separate():
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    frame = pd.DataFrame(
        {
            "Date": np.repeat(dates, 3),
            "FTR": ["H", "D", "A"] * len(dates),
        }
    )
    train, validation, test, _ = temporal_split(frame)
    assert train["Date"].max() < validation["Date"].min()
    assert validation["Date"].max() < test["Date"].min()
    assert len(train) + len(validation) + len(test) == len(frame)
