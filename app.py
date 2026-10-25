"""Streamlit demo for the trained football prediction pipeline."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st
import os
import subprocess

from src.features.build_features import build_fixture_features, load_matches
from src.models.inference import load_bundle, predict_fixture
from src.models.train_models import CLASS_ORDER

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

st.set_page_config(
    page_title="Football Match Prediction",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_resources():
    model_path = "models/artifacts/football_prediction_bundle.joblib"
    
    # Nếu chưa có file model trên server, tự động chạy script huấn luyện
    if not os.path.exists(model_path):
        st.info("Đang tự động khởi tạo và huấn luyện mô hình trên server, vui lòng đợi trong giây lát...")
        subprocess.run(["python", "run_project.py"], check=True)
        
    return load_bundle(), load_matches()


st.title("⚽ Football Match Outcome Prediction")
st.caption("Dự đoán kết quả bóng đá bằng Machine Learning · Dữ liệu lịch sử Premier League")

try:
    bundle, matches = get_resources()
except Exception as exc:
    st.error(f"Không tải được dữ liệu hoặc mô hình: {exc}")
    st.info("Hãy chạy `python run_project.py` trước, sau đó mở lại website.")
    st.stop()

with st.sidebar:
    st.header("Thông tin hệ thống")
    st.write("**Nguồn dữ liệu:** Football-Data.co.uk (CSV cục bộ)")
    st.write(f"**Lịch sử:** {matches['Date'].min().date()} – {matches['Date'].max().date()}")
    st.write(f"**Số trận:** {len(matches):,}")
    st.write(f"**Mô hình chọn:** `{bundle['selected_model_name']}`")
    st.write(f"**Huấn luyện đến:** {bundle['training_end']}")
    st.caption("Dự đoán mang tính tham khảo, không bảo đảm kết quả thực tế.")

prediction_tab, evaluation_tab, methodology_tab = st.tabs(
    ["Dự đoán trận đấu", "Đánh giá mô hình", "Phương pháp"]
)

with prediction_tab:
    st.subheader("Tạo dự đoán trước trận")
    teams = sorted(set(matches["HomeTeam"].dropna()) | set(matches["AwayTeam"].dropna()))
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        home_team = st.selectbox("Đội nhà", teams, index=teams.index("Arsenal") if "Arsenal" in teams else 0)
    with col2:
        away_default = teams.index("Liverpool") if "Liverpool" in teams else min(1, len(teams) - 1)
        away_team = st.selectbox("Đội khách", teams, index=away_default)
    with col3:
        prediction_date = st.date_input("Ngày dự đoán", value=date.today())

    if home_team == away_team:
        st.warning("Đội nhà và đội khách phải khác nhau.")
    else:
        if st.button("Dự đoán kết quả", type="primary", use_container_width=True):
            try:
                features = build_fixture_features(matches, home_team, away_team, prediction_date)
                result = predict_fixture(bundle, features)
                probabilities = result["probabilities"]
                labels = {"H": "Đội nhà thắng", "D": "Hòa", "A": "Đội khách thắng"}
                st.success(f"Đã tạo dự đoán bằng mô hình `{result['selected_model']}`.")
                probability_map = dict(zip(result["classes"], probabilities))
                metric_cols = st.columns(3)
                for col, label in zip(metric_cols, ["H", "D", "A"]):
                    with col:
                        st.metric(labels[label], f"{probability_map[label] * 100:.1f}%")
                chart_data = pd.DataFrame(
                    {"Kết quả": [labels[x] for x in ["H", "D", "A"]],
                     "Xác suất": [probability_map[x] for x in ["H", "D", "A"]]}
                ).set_index("Kết quả")
                st.bar_chart(chart_data)

                goal_col1, goal_col2, score_col = st.columns(3)
                goal_col1.metric("Bàn kỳ vọng đội nhà (Poisson)", f"{result['home_goals_expected']:.2f}")
                goal_col2.metric("Bàn kỳ vọng đội khách (Poisson)", f"{result['away_goals_expected']:.2f}")
                score_col.metric("Tỷ số có xác suất cao nhất", result["most_likely_score"])
                if result["top_scores"]:
                    st.caption(
                        f"Xác suất của đúng tỷ số {result['most_likely_score']}: "
                        f"{result['top_scores'][0]['probability'] * 100:.2f}%. "
                        "Tỷ số chính xác có xác suất cao nhất không nhất thiết trùng với kết quả 1X2 có xác suất cao nhất, vì mỗi lớp 1X2 gộp nhiều tỷ số."
                    )

                st.markdown("**Các tỷ số có xác suất cao nhất theo Poisson**")
                top_scores = pd.DataFrame(result["top_scores"])
                top_scores["probability"] = (top_scores["probability"] * 100).round(2).astype(str) + "%"
                top_scores = top_scores.rename(columns={"score": "Tỷ số", "probability": "Xác suất"})
                st.dataframe(top_scores, hide_index=True, use_container_width=True)

                with st.expander("Xem đặc trưng trước trận được đưa vào mô hình"):
                    st.dataframe(features.T.rename(columns={0: "Giá trị"}), use_container_width=True)
                    st.caption("Các trận diễn ra cùng ngày với ngày dự đoán không được đưa vào lịch sử, nhằm tránh dùng kết quả chưa biết.")
                if result.get("balanced_alternative") is not None:
                    balanced_label = {"H": "Đội nhà thắng", "D": "Hòa", "A": "Đội khách thắng"}.get(
                        result["balanced_alternative"]["prediction"], result["balanced_alternative"]["prediction"]
                    )
                    st.info(
                        f"Tham khảo thêm: Logistic Regression có class_weight='balanced' dự đoán **{balanced_label}**. "
                        "Mô hình này ưu tiên cân bằng khả năng nhận diện ba lớp; xác suất của nó chưa được calibration riêng."
                    )
                st.caption("Poisson dự đoán phân phối tỷ số; xác suất 1X2 cuối cùng dùng mô hình được chọn trong validation. Hai đầu ra có thể khác nhau.")
            except Exception as exc:
                st.error(f"Không thể dự đoán trận này: {exc}")

with evaluation_tab:
    st.subheader("Kết quả đánh giá thực nghiệm")
    metrics_path = REPORTS / "metrics.csv"
    if metrics_path.exists():
        metrics = pd.read_csv(metrics_path)
        st.caption("Bảng dưới đây lấy từ lần chạy pipeline gần nhất. Chỉ số test được giữ tách khỏi bước chọn mô hình.")
        split_choice = st.selectbox("Tập dữ liệu", ["test", "validation"])
        view = metrics[metrics["split"] == split_choice].sort_values("log_loss")
        st.dataframe(view, hide_index=True, use_container_width=True)
        if split_choice == "test":
            report_path = REPORTS / "classification_reports_all_models_test.json"
            if report_path.exists():
                import json
                detailed = json.loads(report_path.read_text(encoding="utf-8"))
                selected_report = detailed.get("selected_model", {})
                draw_recall = selected_report.get("Draw (D)", {}).get("recall", 0.0)
                balanced_report = detailed.get("logistic_regression_balanced", {})
                balanced_draw_recall = balanced_report.get("Draw (D)", {}).get("recall", 0.0)
                st.warning(
                    f"Lưu ý về lớp hòa: Draw recall của mô hình được chọn là {draw_recall:.1%}; "
                    f"Logistic Regression cân bằng lớp đạt {balanced_draw_recall:.1%}. "
                    "Đây là hạn chế thực nghiệm cần được nêu rõ, không nên tuyên bố hệ thống nhận diện tốt cả ba lớp."
                )
                rows = []
                for label in ["Away win (A)", "Draw (D)", "Home win (H)"]:
                    values = selected_report.get(label, {})
                    rows.append({
                        "Lớp": label,
                        "Precision": values.get("precision"),
                        "Recall": values.get("recall"),
                        "F1": values.get("f1-score"),
                        "Số mẫu": values.get("support"),
                    })
                st.markdown("**Chi tiết từng lớp — mô hình được chọn**")
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        if split_choice == "test":
            figure = REPORTS / "figures" / "test_log_loss_comparison.png"
            confusion = REPORTS / "figures" / "selected_confusion_matrix.png"
            if figure.exists():
                st.image(str(figure), caption="So sánh Log Loss trên test")
            if confusion.exists():
                st.image(str(confusion), caption="Confusion matrix của mô hình được chọn")
        season_path = REPORTS / "test_metrics_by_season.csv"
        if season_path.exists():
            st.markdown("**Hiệu năng theo mùa giải trong test**")
            st.dataframe(pd.read_csv(season_path), hide_index=True, use_container_width=True)
    else:
        st.info("Chưa có kết quả. Chạy `python run_project.py` để huấn luyện và đánh giá.")

with methodology_tab:
    st.subheader("Quy trình và giới hạn")
    st.markdown(
        """
        **Đặc trưng trước trận:** phong độ 5 trận, bàn thắng/bàn thua gần đây, số trận đã chơi,
        số ngày nghỉ và Elo. Với các mùa không có giờ thi đấu, mọi trận cùng ngày được tính
        đặc trưng trước khi cập nhật kết quả của ngày đó.

        **Mô hình 1X2:** Logistic Regression, Decision Tree, Random Forest và HistGradientBoosting.
        Baseline là xác suất lớp theo tần suất trong tập train.

        **Dự đoán bàn thắng:** hai mô hình Poisson độc lập dự đoán bàn đội nhà và đội khách,
        sau đó suy ra phân phối tỷ số và xác suất 1X2. Tỷ số có xác suất cao nhất không nhất thiết
        tương ứng với lớp 1X2 có xác suất cao nhất vì một lớp 1X2 tổng hợp nhiều tỷ số.

        **Lựa chọn mô hình:** validation được dùng để chọn mô hình và trọng số hybrid.
        Test chỉ được dùng cho đánh giá cuối cùng.

        **Chưa sử dụng:** xG, đội hình, chấn thương, dữ liệu cầu thủ và tỷ lệ cược. Các yếu tố này
        chỉ nên bổ sung khi có nguồn dữ liệu lịch sử phù hợp và timestamp đáng tin cậy.
        """
    )
    st.write("**Thứ tự lớp xác suất:** A = đội khách thắng, D = hòa, H = đội nhà thắng.")
    st.write("**Đặc trưng sử dụng:**", ", ".join(bundle["feature_columns"]))
    metadata_path = ROOT / "models" / "artifacts" / "metadata.json"
    if metadata_path.exists():
        with st.expander("Thông tin artifact và khoảng chia dữ liệu"):
            import json
            st.json(json.loads(metadata_path.read_text(encoding="utf-8")))
