"""Huấn luyện, lựa chọn và đánh giá các bộ phân loại 1X2 cùng các mô hình
dự báo số bàn thắng theo phân phối Poisson.

Validation được dùng để chọn mô hình; giai đoạn Test cuối cùng không được 
tham gia fit, tuning hoặc chọn mô hình
"""
from __future__ import annotations

import json
import math
import platform
import sys
import warnings
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_poisson_deviance,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.features.build_features import FEATURE_COLUMNS, load_matches

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PATH = PROJECT_ROOT / "Data" / "processed" / "features_data.csv"
MODEL_DIR = PROJECT_ROOT / "models" / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
CLASS_ORDER = ["A", "D", "H"]  # fixed order for every probability vector
RANDOM_STATE = 42

""" Đọc dữ liệu đặc trưng từ file CSV """
def load_feature_data(path: Path = FEATURES_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}. Hãy chạy build_features trước.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    missing = [c for c in [*FEATURE_COLUMNS, "FTR", "FTHG", "FTAG", "HomeTeam", "AwayTeam", "Season"] if c not in df]
    if missing:
        raise ValueError(f"Thiếu cột cần thiết trong features_data.csv: {missing}")
    df = df.dropna(subset=["Date", "FTR", "FTHG", "FTAG", *FEATURE_COLUMNS]).copy()
    df = df[df["FTR"].isin(CLASS_ORDER)]
    df = df.sort_values(["Date", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    if len(df) < 500:
        raise ValueError(f"Chỉ có {len(df)} trận hợp lệ; chưa đủ để thiết kế thực nghiệm ổn định.")
    return df

"""Chia dữ liệu theo trình tự thời gian thành 3 tập: Train (70%), Validation (15%), Test (15%)"""
def temporal_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    counts = df.groupby("Date", sort=True).size()
    cumulative = counts.cumsum()
    n = len(df)
    train_target = max(1, int(n * 0.70))
    validation_target = max(train_target + 1, int(n * 0.85))
    train_end = cumulative.index[min(int(np.searchsorted(cumulative.to_numpy(), train_target, side="left")), len(cumulative) - 1)]
    validation_end = cumulative.index[min(int(np.searchsorted(cumulative.to_numpy(), validation_target, side="left")), len(cumulative) - 1)]
    train = df[df["Date"] <= train_end].copy()
    validation = df[(df["Date"] > train_end) & (df["Date"] <= validation_end)].copy()
    test = df[df["Date"] > validation_end].copy()
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Không thể tạo đủ train/validation/test theo thời gian.")
    if not (train["Date"].max() < validation["Date"].min() < validation["Date"].max() < test["Date"].min()):
        raise AssertionError("Temporal split bị chồng lấn ngày.")
    dates = {
        "train_start": str(train["Date"].min().date()),
        "train_end": str(train["Date"].max().date()),
        "validation_start": str(validation["Date"].min().date()),
        "validation_end": str(validation["Date"].max().date()),
        "test_start": str(test["Date"].min().date()),
        "test_end": str(test["Date"].max().date()),
    }
    return train, validation, test, dates

"""sử dụng Logistic Regression, Decision Tree, Random Forest, và HistGradientBoostingClassifier"""
def create_classifiers() -> dict[str, Any]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(C=0.5, max_iter=2500, random_state=RANDOM_STATE)),
            ]
        ),
        "logistic_regression_balanced": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    C=0.5, max_iter=2500, class_weight="balanced", random_state=RANDOM_STATE
                )),
            ]
        ),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=45, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=9,
            min_samples_leaf=20,
            max_features="sqrt",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=15,
            min_samples_leaf=35,
            l2_regularization=2.0,
            random_state=RANDOM_STATE,
        ),
    }

"""Đảm bảo ma trận xác suất đầu ra của mô hình luôn tuân theo đúng thứ tự cột ["A", "D", "H"]   """
def align_probabilities(model: Any, X: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(X)
    result = np.zeros((len(X), len(CLASS_ORDER)), dtype=float)
    for source_index, label in enumerate(model.classes_):
        if label in CLASS_ORDER:
            result[:, CLASS_ORDER.index(label)] = raw[:, source_index]
    row_sums = result.sum(axis=1, keepdims=True)
    result = np.divide(result, row_sums, out=np.full_like(result, 1.0 / len(CLASS_ORDER)), where=row_sums > 0)
    return np.clip(result, 1e-12, 1.0)


def probabilities_to_labels(probabilities: np.ndarray) -> np.ndarray:
    return np.asarray(CLASS_ORDER, dtype=object)[np.argmax(probabilities, axis=1)]


def multiclass_brier(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    one_hot = np.column_stack([(y_true == label).astype(float) for label in CLASS_ORDER])
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))

""" sử dụng Accuracy, Macro F1, Balanced Accuracy, Log loss, và Brier score để đánh giá """
def classification_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = probabilities_to_labels(probabilities)
    return {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=CLASS_ORDER, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "log_loss": float(log_loss(y_true, probabilities, labels=CLASS_ORDER)),
        "brier_score": multiclass_brier(y_true, probabilities),
    }


def fit_goal_models(train: pd.DataFrame) -> tuple[Pipeline, Pipeline]:
    home_model = Pipeline(
        [("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=0.25, max_iter=2000))]
    )
    away_model = Pipeline(
        [("scaler", StandardScaler()), ("model", PoissonRegressor(alpha=0.25, max_iter=2000))]
    )
    X = train[FEATURE_COLUMNS]
    home_model.fit(X, train["FTHG"].astype(float))
    away_model.fit(X, train["FTAG"].astype(float))
    return home_model, away_model


def predict_goal_rates(home_model: Any, away_model: Any, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    home_lam = np.clip(home_model.predict(X), 0.05, 5.5)
    away_lam = np.clip(away_model.predict(X), 0.05, 5.5)
    return home_lam, away_lam


def poisson_outcome_probabilities(home_lam: np.ndarray, away_lam: np.ndarray, max_goals: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ks = np.arange(max_goals + 1, dtype=float)
    log_factorials = np.array([math.lgamma(k + 1.0) for k in ks])
    home_pmf = np.exp(-home_lam[:, None] + ks[None, :] * np.log(home_lam[:, None]) - log_factorials[None, :])
    away_pmf = np.exp(-away_lam[:, None] + ks[None, :] * np.log(away_lam[:, None]) - log_factorials[None, :])
    matrices = home_pmf[:, :, None] * away_pmf[:, None, :]
    totals = matrices.sum(axis=(1, 2), keepdims=True)
    matrices = matrices / np.maximum(totals, 1e-15)
    h_mask = np.tril(np.ones((max_goals + 1, max_goals + 1), dtype=bool), k=-1)
    d_mask = np.eye(max_goals + 1, dtype=bool)
    a_mask = np.triu(np.ones((max_goals + 1, max_goals + 1), dtype=bool), k=1)
    p_home = matrices[:, h_mask].sum(axis=1)
    p_draw = matrices[:, d_mask].sum(axis=1)
    p_away = matrices[:, a_mask].sum(axis=1)
    # CLASS_ORDER = A, D, H
    probabilities = np.column_stack([p_away, p_draw, p_home])
    probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
    top_flat = matrices.reshape(len(matrices), -1).argmax(axis=1)
    width = max_goals + 1
    top_home = top_flat // width
    top_away = top_flat % width
    top_scores = np.column_stack([top_home, top_away])
    return probabilities, top_scores, matrices


def _class_prior_probabilities(y_train: pd.Series, n: int) -> np.ndarray:
    priors = y_train.value_counts(normalize=True).reindex(CLASS_ORDER, fill_value=0.0).to_numpy(dtype=float)
    priors = priors / priors.sum()
    return np.tile(priors, (n, 1))


def _evaluate_and_append(rows: list[dict[str, Any]], split: str, model_name: str, y: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    metrics = classification_metrics(y, probs)
    rows.append({"split": split, "model": model_name, "n_samples": int(len(y)), **metrics})
    return metrics


def _select_base_strategy(val_probabilities: dict[str, np.ndarray], y_val: np.ndarray) -> tuple[str, np.ndarray, float]:
    individual_names = list(val_probabilities)
    individual_losses = {
        name: log_loss(y_val, probs, labels=CLASS_ORDER)
        for name, probs in val_probabilities.items()
    }
    best_name = min(individual_losses, key=individual_losses.get)
    soft_vote = np.mean(np.stack([val_probabilities[name] for name in individual_names]), axis=0)
    soft_loss = log_loss(y_val, soft_vote, labels=CLASS_ORDER)
    if soft_loss < individual_losses[best_name]:
        return "soft_voting_equal", soft_vote, float(soft_loss)
    return f"single:{best_name}", val_probabilities[best_name], float(individual_losses[best_name])


def _optimize_hybrid_weight(base_probs: np.ndarray, poisson_probs: np.ndarray, y_val: np.ndarray) -> tuple[float, float, float]:
    best_weight = 1.0
    best_loss = float(log_loss(y_val, base_probs, labels=CLASS_ORDER))
    for weight in np.linspace(0.0, 1.0, 21):
        combined = weight * base_probs + (1.0 - weight) * poisson_probs
        loss = float(log_loss(y_val, combined, labels=CLASS_ORDER))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return best_weight, best_loss, float(log_loss(y_val, base_probs, labels=CLASS_ORDER))

"""Vẽ và lưu biểu đồ Log Loss của tập test và Confusion Matrix (ma trận nhầm lẫn) dưới dạng ảnh PNG"""
def _save_plots(metrics_df: pd.DataFrame, y_test: np.ndarray, selected_probabilities: np.ndarray, selected_name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    test_metrics = metrics_df[metrics_df["split"] == "test"].sort_values("log_loss")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(test_metrics["model"], test_metrics["log_loss"])
    ax.set_title("Test Log Loss (lower is better)")
    ax.set_ylabel("Log loss")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "test_log_loss_comparison.png", dpi=160)
    plt.close(fig)

    cm = confusion_matrix(y_test, probabilities_to_labels(selected_probabilities), labels=CLASS_ORDER)
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Confusion Matrix - {selected_name}")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")
    ax.set_xticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    ax.set_yticks(range(len(CLASS_ORDER)), CLASS_ORDER)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "selected_confusion_matrix.png", dpi=160)
    plt.close(fig)


def train_and_evaluate() -> dict[str, Any]:
    print("TRAIN / VALIDATION / TEST, SO SÁNH MODEL VÀ POISSON")
    df = load_feature_data()
    train, validation, test, split_dates = temporal_split(df)
    print(f"Train      : {len(train):,} ({split_dates['train_start']} -> {split_dates['train_end']})")
    print(f"Validation : {len(validation):,} ({split_dates['validation_start']} -> {split_dates['validation_end']})")
    print(f"Test       : {len(test):,} ({split_dates['test_start']} -> {split_dates['test_end']})")

    X_train, y_train = train[FEATURE_COLUMNS], train["FTR"].to_numpy()
    X_val, y_val = validation[FEATURE_COLUMNS], validation["FTR"].to_numpy()
    X_test, y_test = test[FEATURE_COLUMNS], test["FTR"].to_numpy()

    metric_rows: list[dict[str, Any]] = []
    prior_val = _class_prior_probabilities(train["FTR"], len(validation))
    _evaluate_and_append(metric_rows, "validation", "majority_prior_baseline", y_val, prior_val)

    val_models: dict[str, Any] = {}
    val_probs: dict[str, np.ndarray] = {}
    print("\nHuấn luyện các mô hình phân loại trên TRAIN...")
    for name, model in create_classifiers().items():
        model.fit(X_train, y_train)
        probs = align_probabilities(model, X_val)
        val_models[name] = model
        val_probs[name] = probs
        metrics = _evaluate_and_append(metric_rows, "validation", name, y_val, probs)
        print(f"{name:24s} log_loss={metrics['log_loss']:.4f} accuracy={metrics['accuracy']:.4f} macro_f1={metrics['macro_f1']:.4f}")

    val_goal_home, val_goal_away = fit_goal_models(train)
    val_home_lam, val_away_lam = predict_goal_rates(val_goal_home, val_goal_away, X_val)
    val_poisson_probs, val_top_scores, _ = poisson_outcome_probabilities(val_home_lam, val_away_lam)
    val_poisson_metrics = _evaluate_and_append(metric_rows, "validation", "poisson_1x2", y_val, val_poisson_probs)

    base_strategy, base_val_probs, base_val_loss = _select_base_strategy(val_probs, y_val)
    soft_val_probs = np.mean(np.stack(list(val_probs.values())), axis=0)
    soft_val_metrics = _evaluate_and_append(metric_rows, "validation", "soft_voting_equal", y_val, soft_val_probs)

    hybrid_weight, hybrid_val_loss, baseline_base_loss = _optimize_hybrid_weight(base_val_probs, val_poisson_probs, y_val)
    hybrid_val_probs = hybrid_weight * base_val_probs + (1.0 - hybrid_weight) * val_poisson_probs
    hybrid_val_metrics = _evaluate_and_append(metric_rows, "validation", "weighted_blend_candidate", y_val, hybrid_val_probs)
    poisson_val_loss = float(log_loss(y_val, val_poisson_probs, labels=CLASS_ORDER))
    hybrid_selected = (
        hybrid_val_loss < base_val_loss - 0.001
        and 1e-9 < hybrid_weight < 1.0 - 1e-9
    )
    poisson_selected = (
        hybrid_weight <= 1e-9
        and poisson_val_loss < base_val_loss - 0.001
    )
    selected_name = (
        "weighted_hybrid" if hybrid_selected
        else "poisson_1x2" if poisson_selected
        else base_strategy
    )
    print(f"\nBase strategy: {base_strategy} (validation log loss={base_val_loss:.4f})")
    print(f"Hybrid weight: base={hybrid_weight:.2f}, Poisson={1.0-hybrid_weight:.2f}; validation log loss={hybrid_val_loss:.4f}")
    if hybrid_selected:
        validation_decision = "CHỌN WEIGHTED HYBRID"
    elif poisson_selected:
        validation_decision = "CHỌN POISSON ĐỘC LẬP (weight Poisson = 1)"
    else:
        validation_decision = "GIỮ BASE, KHÔNG ÉP ENSEMBLE"
    print(f"Validation decision: {validation_decision}")

    # Refit artifacts on train + validation only after selection. Test remains untouched.
    train_validation = pd.concat([train, validation], ignore_index=True)
    X_trainval = train_validation[FEATURE_COLUMNS]
    y_trainval = train_validation["FTR"].to_numpy()
    final_class_models: dict[str, Any] = {}
    print("\nHuấn luyện lại các model lưu cho website trên TRAIN + VALIDATION...")
    for name, model in create_classifiers().items():
        model.fit(X_trainval, y_trainval)
        final_class_models[name] = model

    final_home_model, final_away_model = fit_goal_models(train_validation)
    test_probabilities: dict[str, np.ndarray] = {}
    for name, model in final_class_models.items():
        test_probabilities[name] = align_probabilities(model, X_test)
    soft_test_probs = np.mean(np.stack(list(test_probabilities.values())), axis=0)
    test_home_lam, test_away_lam = predict_goal_rates(final_home_model, final_away_model, X_test)
    poisson_test_probs, test_top_scores, test_score_matrices = poisson_outcome_probabilities(test_home_lam, test_away_lam)

    prior_test = _class_prior_probabilities(train_validation["FTR"], len(test))
    _evaluate_and_append(metric_rows, "test", "majority_prior_baseline", y_test, prior_test)
    for name, probabilities in test_probabilities.items():
        _evaluate_and_append(metric_rows, "test", name, y_test, probabilities)
    _evaluate_and_append(metric_rows, "test", "soft_voting_equal", y_test, soft_test_probs)
    _evaluate_and_append(metric_rows, "test", "poisson_1x2", y_test, poisson_test_probs)

    if base_strategy == "soft_voting_equal":
        base_test_probs = soft_test_probs
    else:
        best_classifier_name = base_strategy.split(":", 1)[1]
        base_test_probs = test_probabilities[best_classifier_name]
    hybrid_test_probs = hybrid_weight * base_test_probs + (1.0 - hybrid_weight) * poisson_test_probs
    _evaluate_and_append(metric_rows, "test", "weighted_blend_candidate", y_test, hybrid_test_probs)
    if hybrid_selected:
        selected_test_probs = hybrid_test_probs
        selected_test_name = "weighted_hybrid"
    elif poisson_selected:
        selected_test_probs = poisson_test_probs
        selected_test_name = "poisson_1x2"
    else:
        selected_test_probs = base_test_probs
        selected_test_name = base_strategy.replace("single:", "")
    selected_test_metrics = classification_metrics(y_test, selected_test_probs)

    # Score model metrics are assessed on the same final test interval.
    home_goal_true = test["FTHG"].to_numpy(dtype=float)
    away_goal_true = test["FTAG"].to_numpy(dtype=float)
    home_goal_mae = float(mean_absolute_error(home_goal_true, test_home_lam))
    away_goal_mae = float(mean_absolute_error(away_goal_true, test_away_lam))
    home_goal_rmse = float(np.sqrt(np.mean((home_goal_true - test_home_lam) ** 2)))
    away_goal_rmse = float(np.sqrt(np.mean((away_goal_true - test_away_lam) ** 2)))
    goal_metrics = {
        "test_matches": int(len(test)),
        "home_goals_mae": home_goal_mae,
        "away_goals_mae": away_goal_mae,
        "combined_goals_mae": float((home_goal_mae + away_goal_mae) / 2),
        "home_goals_rmse": home_goal_rmse,
        "away_goals_rmse": away_goal_rmse,
        "home_goals_poisson_deviance": float(mean_poisson_deviance(home_goal_true, np.maximum(test_home_lam, 1e-8))),
        "away_goals_poisson_deviance": float(mean_poisson_deviance(away_goal_true, np.maximum(test_away_lam, 1e-8))),
    }

    metrics_df = pd.DataFrame(metric_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(REPORT_DIR / "metrics.csv", index=False, float_format="%.6f", encoding="utf-8-sig")
    (REPORT_DIR / "goal_prediction_metrics.json").write_text(json.dumps(goal_metrics, indent=2), encoding="utf-8")

    # Detailed test predictions for the report and reproducibility.
    prediction_frame = test[["Date", "Season", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].copy()
    prediction_frame["Predicted_FTR"] = probabilities_to_labels(selected_test_probs)
    for i, label in enumerate(CLASS_ORDER):
        prediction_frame[f"P_{label}"] = selected_test_probs[:, i]
        prediction_frame[f"Poisson_P_{label}"] = poisson_test_probs[:, i]
    prediction_frame["Poisson_Home_Goals_Expected"] = test_home_lam
    prediction_frame["Poisson_Away_Goals_Expected"] = test_away_lam
    prediction_frame["Poisson_Top_Score"] = [f"{h}-{a}" for h, a in test_top_scores]
    prediction_frame.to_csv(REPORT_DIR / "test_predictions.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    season_rows = []
    for season, group in prediction_frame.groupby("Season", dropna=False):
        idx = group.index.to_numpy()
        # prediction_frame preserves test's fresh 0-based index after copy
        local_positions = np.searchsorted(prediction_frame.index.to_numpy(), idx)
        probs = selected_test_probs[local_positions]
        y_season = group["FTR"].to_numpy()
        season_rows.append({"Season": season, "matches": len(group), **classification_metrics(y_season, probs)})
    pd.DataFrame(season_rows).to_csv(REPORT_DIR / "test_metrics_by_season.csv", index=False, encoding="utf-8-sig", float_format="%.6f")

    cm = confusion_matrix(y_test, probabilities_to_labels(selected_test_probs), labels=CLASS_ORDER)
    pd.DataFrame(cm, index=[f"actual_{x}" for x in CLASS_ORDER], columns=[f"pred_{x}" for x in CLASS_ORDER]).to_csv(
        REPORT_DIR / "selected_confusion_matrix.csv", encoding="utf-8-sig"
    )
    report_dict = classification_report(
        y_test, probabilities_to_labels(selected_test_probs), labels=CLASS_ORDER,
        target_names=["Away win (A)", "Draw (D)", "Home win (H)"], output_dict=True, zero_division=0
    )
    (REPORT_DIR / "classification_report_test.json").write_text(json.dumps(report_dict, indent=2), encoding="utf-8")
    detailed_reports = {}
    for model_name, probabilities in test_probabilities.items():
        detailed_reports[model_name] = classification_report(
            y_test, probabilities_to_labels(probabilities), labels=CLASS_ORDER,
            target_names=["Away win (A)", "Draw (D)", "Home win (H)"], output_dict=True, zero_division=0
        )
    detailed_reports["poisson_1x2"] = classification_report(
        y_test, probabilities_to_labels(poisson_test_probs), labels=CLASS_ORDER,
        target_names=["Away win (A)", "Draw (D)", "Home win (H)"], output_dict=True, zero_division=0
    )
    detailed_reports["selected_model"] = report_dict
    (REPORT_DIR / "classification_reports_all_models_test.json").write_text(
        json.dumps(detailed_reports, indent=2), encoding="utf-8"
    )
    _save_plots(metrics_df, y_test, selected_test_probs, selected_test_name)

    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "joblib": joblib.__version__,
    }
    (REPORT_DIR / "environment.json").write_text(json.dumps(versions, indent=2), encoding="utf-8")

    best_classifier_name = min(
        {name: log_loss(y_val, probs, labels=CLASS_ORDER) for name, probs in val_probs.items()},
        key=lambda name: log_loss(y_val, val_probs[name], labels=CLASS_ORDER),
    )
    bundle = {
        "classes": CLASS_ORDER,
        "feature_columns": FEATURE_COLUMNS,
        "classification_models": final_class_models,
        "best_classifier_name": best_classifier_name,
        "base_strategy": base_strategy,
        "hybrid_weight_base": float(hybrid_weight),
        "hybrid_selected": bool(hybrid_selected),
        "poisson_selected": bool(poisson_selected),
        "selected_model_name": selected_name,
        "poisson_home_model": final_home_model,
        "poisson_away_model": final_away_model,
        "training_start": str(train_validation["Date"].min().date()),
        "training_end": str(train_validation["Date"].max().date()),
        "split_dates": split_dates,
        "validation_base_log_loss": float(base_val_loss),
        "validation_hybrid_log_loss": float(hybrid_val_loss),
        "hybrid_selection_min_improvement": 0.001,
        "random_state": RANDOM_STATE,
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_DIR / "football_prediction_bundle.joblib", compress=3)
    metadata = {
        "project": "Football Match Outcome Prediction Using Machine Learning",
        "artifact": "football_prediction_bundle.joblib",
        "selected_model": selected_name,
        "best_classifier": best_classifier_name,
        "base_strategy": base_strategy,
        "hybrid_selected": bool(hybrid_selected),
        "poisson_selected": bool(poisson_selected),
        "hybrid_weight_base": float(hybrid_weight),
        "hybrid_weight_poisson": float(1.0 - hybrid_weight),
        "classes_order": CLASS_ORDER,
        "features": FEATURE_COLUMNS,
        "split_dates": split_dates,
        "validation_metrics": {
            "base_log_loss": float(base_val_loss),
            "hybrid_log_loss": float(hybrid_val_loss),
        },
        "test_selected_metrics": selected_test_metrics,
        "goal_metrics_test": goal_metrics,
        "environment": versions,
        "disclaimer": "Kết quả chỉ mang tính tham khảo; không bảo đảm dự đoán đúng trận đấu tương lai.",
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    summary = [
        "# Báo cáo thực nghiệm tự động",
        "",
        "## Thiết kế đánh giá",
        "",
        f"- Train: {len(train):,} trận, {split_dates['train_start']} đến {split_dates['train_end']}",
        f"- Validation: {len(validation):,} trận, {split_dates['validation_start']} đến {split_dates['validation_end']}",
        f"- Test: {len(test):,} trận, {split_dates['test_start']} đến {split_dates['test_end']}",
        "- Chia theo ngày lịch để không tách các trận cùng ngày sang hai tập.",
        "- Validation dùng để chọn base strategy và trọng số hybrid; test không tham gia chọn mô hình.",
        "",
        "## Quyết định mô hình",
        "",
        f"- Base strategy: `{base_strategy}` (validation log loss {base_val_loss:.4f})",
        f"- Hybrid weight: base={hybrid_weight:.2f}, Poisson={1.0-hybrid_weight:.2f}",
        f"- Hybrid được chọn: `{hybrid_selected}` (validation log loss {hybrid_val_loss:.4f})",
        f"- Poisson độc lập được chọn: `{poisson_selected}` (validation log loss {poisson_val_loss:.4f})",
        f"- Mô hình được website sử dụng: `{selected_name}`",
        "",
        "## Kết quả test",
        "",
        f"- Accuracy: {selected_test_metrics['accuracy']:.4f}",
        f"- Macro F1: {selected_test_metrics['macro_f1']:.4f}",
        f"- Balanced accuracy: {selected_test_metrics['balanced_accuracy']:.4f}",
        f"- Log loss: {selected_test_metrics['log_loss']:.4f}",
        f"- Multiclass Brier score: {selected_test_metrics['brier_score']:.4f}",
        f"- Recall lớp hòa (D) của mô hình được chọn: {report_dict['Draw (D)']['recall']:.4f}",
        "- Xem `classification_reports_all_models_test.json` để so sánh precision/recall/F1 của từng lớp cho tất cả mô hình.",
        "",
        "## Kết quả dự đoán số bàn",
        "",
        f"- Home goals MAE: {home_goal_mae:.4f}",
        f"- Away goals MAE: {away_goal_mae:.4f}",
        f"- Combined goals MAE: {goal_metrics['combined_goals_mae']:.4f}",
        f"- Home goals RMSE: {home_goal_rmse:.4f}",
        f"- Away goals RMSE: {away_goal_rmse:.4f}",
        "",
        "## Giới hạn",
        "",
        "- Chỉ sử dụng kết quả lịch sử Football-Data.co.uk và các đặc trưng tạo từ kết quả trước trận.",
        "- Chưa đưa dữ liệu chấn thương, đội hình, xG hoặc odds vào pipeline.",
        "- Kết quả test là backtest tuần tự; kết quả của các trận trước trong giai đoạn test được dùng để cập nhật trạng thái cho các ngày sau, đúng với kịch bản dự đoán tuần tự.",
        "- Chưa có kiểm định ý nghĩa thống kê giữa các mô hình; chênh lệch metric nhỏ cần được diễn giải thận trọng.",
        "",
        "Các số liệu trong file này được sinh trực tiếp từ lần chạy pipeline, không phải số liệu minh họa.",
    ]
    (REPORT_DIR / "experiment_summary.md").write_text("\n".join(summary), encoding="utf-8")

    print("\nKẾT QUẢ TEST:")
    print(metrics_df[metrics_df["split"] == "test"].sort_values("log_loss").to_string(index=False))
    print("\nMô hình website:", selected_name)
    print("Bundle:", MODEL_DIR / "football_prediction_bundle.joblib")
    print("Báo cáo:", REPORT_DIR / "experiment_summary.md")

    return {
        "metrics": metrics_df,
        "selected_model": selected_name,
        "hybrid_selected": hybrid_selected,
        "poisson_selected": poisson_selected,
        "test_metrics": selected_test_metrics,
        "goal_metrics": goal_metrics,
        "split_dates": split_dates,
    }


if __name__ == "__main__":
    train_and_evaluate()
