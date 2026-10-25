"""Logic dự đoán được chia sẻ, sử dụng bởi bản demo Streamlit"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.models.train_models import (
    align_probabilities,
    poisson_outcome_probabilities,
    predict_goal_rates,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_PATH = PROJECT_ROOT / "models" / "artifacts" / "football_prediction_bundle.joblib"


def load_bundle(path: Path = BUNDLE_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Chưa có model artifact: {path}. Hãy chạy `python run_project.py` trước."
        )
    return joblib.load(path)


def predict_fixture(bundle: dict[str, Any], features: pd.DataFrame) -> dict[str, Any]:
    feature_columns = bundle["feature_columns"]
    X = features[feature_columns]
    class_models = bundle["classification_models"]
    strategy = bundle["base_strategy"]

    if strategy == "soft_voting_equal":
        probabilities = np.mean(
            np.stack([align_probabilities(model, X) for model in class_models.values()]), axis=0
        )
        base_name = "soft_voting_equal"
    elif strategy.startswith("single:"):
        base_name = strategy.split(":", 1)[1]
        probabilities = align_probabilities(class_models[base_name], X)
    else:
        raise ValueError(f"Chiến lược dự đoán không được hỗ trợ: {strategy}")

    balanced_alternative = None
    if "logistic_regression_balanced" in class_models:
        balanced_probs = align_probabilities(class_models["logistic_regression_balanced"], X)[0]
        balanced_alternative = {
            "prediction": str(bundle["classes"][int(np.argmax(balanced_probs))]),
            "probabilities": balanced_probs,
        }

    home_lam, away_lam = predict_goal_rates(
        bundle["poisson_home_model"], bundle["poisson_away_model"], X
    )
    poisson_probs, top_score, score_matrix = poisson_outcome_probabilities(home_lam, away_lam)

    if bundle.get("poisson_selected", False):
        final_probabilities = poisson_probs
        selected_name = "poisson_1x2"
    elif bundle.get("hybrid_selected", False):
        weight = float(bundle["hybrid_weight_base"])
        final_probabilities = weight * probabilities + (1.0 - weight) * poisson_probs
        selected_name = "weighted_hybrid"
    else:
        final_probabilities = probabilities
        selected_name = base_name

    final_probabilities = final_probabilities / final_probabilities.sum(axis=1, keepdims=True)
    matrix = score_matrix[0]
    ranked_flat = np.argsort(matrix.ravel())[::-1][:5]
    width = matrix.shape[1]
    top_scores = [
        {
            "score": f"{int(index // width)}-{int(index % width)}",
            "probability": float(matrix.ravel()[index]),
        }
        for index in ranked_flat
    ]

    return {
        "classes": bundle["classes"],
        "probabilities": final_probabilities[0],
        "base_probabilities": probabilities[0],
        "poisson_probabilities": poisson_probs[0],
        "home_goals_expected": float(home_lam[0]),
        "away_goals_expected": float(away_lam[0]),
        "most_likely_score": f"{int(top_score[0, 0])}-{int(top_score[0, 1])}",
        "top_scores": top_scores,
        "selected_model": selected_name,
        "balanced_alternative": balanced_alternative,
        "hybrid_selected": bool(bundle.get("hybrid_selected", False)),
        "score_matrix": matrix,
    }
