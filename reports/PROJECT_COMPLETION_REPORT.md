# Báo cáo hoàn thiện project

## 1. Kết quả triển khai

Đã nâng cấp project hiện có, giữ lại nguồn dữ liệu và hướng bài toán ban đầu, thay các script xử lý rời rạc bằng pipeline có thể chạy lại từ đầu. Các số liệu dưới đây được sinh từ lần chạy thực tế của pipeline trên các CSV có trong archive người dùng gửi.

## 2. Dữ liệu

- Nguồn: các file CSV Football-Data.co.uk đã cung cấp trong `Data/raw/Kết quả các năm/`.
- Số trận chuẩn hóa: **9.950**.
- Số đội: **46**.
- Khoảng ngày: **2000-08-19 đến 2026-10-19**.
- 26 mùa hoàn chỉnh từ 2000/01 đến 2025/26: mỗi mùa có 380 trận.
- Mùa 2026/27: 70 trận đã có kết quả trong file hiện tại; đây là mùa chưa hoàn chỉnh.
- 90 dòng có extra fields ở phần cuối (45 ở EPL0304 và 45 ở EPL0405) vẫn được đọc đúng các cột cốt lõi theo header.
- Không phát hiện trận trùng theo Date + HomeTeam + AwayTeam; không phát hiện FTR sai khác với tỷ số sau khi chuẩn hóa.
- Một dòng rỗng trong file EPL1415 được loại do thiếu ngày và kết quả.

## 3. Feature engineering

Đã tạo 15 đặc trưng trước trận: phong độ điểm 5 trận, bàn thắng/bàn thua 5 trận của hai đội, số trận đã chơi, số ngày nghỉ, Elo hai đội, chênh lệch Elo có lợi thế sân nhà và chênh lệch tấn công tương đối.

Quy tắc chống leakage: tất cả trận trong cùng ngày được tính feature trước khi cập nhật kết quả của ngày đó. Đây là quy tắc bảo thủ, kể cả với file có giờ kickoff, vì giờ kickoff không chứng minh chắc chắn trận trước đã kết thúc trước trận sau. Với trận cần dự đoán trong website, chỉ các trận có Date nhỏ hơn ngày dự đoán mới được đưa vào lịch sử.

Không sử dụng FTHG, FTAG, FTR hoặc thống kê sau trận của chính trận đang dự đoán làm đầu vào trực tiếp.

## 4. Thiết kế thực nghiệm

| Tập | Số trận | Từ ngày | Đến ngày |
|---|---:|---|---|
| Train | 6,967 | 2000-08-19 | 2018-11-24 |
| Validation | 1,490 | 2018-11-25 | 2022-10-16 |
| Test | 1,493 | 2022-10-18 | 2026-10-19 |

Tỷ lệ mục tiêu là 70/15/15 theo thứ tự thời gian. Các mốc được điều chỉnh về ngày lịch để một ngày không bị tách giữa hai tập. Mô hình chọn và trọng số được xác định bằng validation; test không tham gia lựa chọn. Artifact triển khai được fit lại trên train + validation sau khi lựa chọn; test vẫn giữ độc lập.

## 5. Kết quả test thực tế

| Mô hình | Accuracy | Macro F1 | Balanced Accuracy | Log Loss | Brier |
|---|---:|---:|---:|---:|---:|
| poisson_1x2 | 0.5271 | 0.3897 | 0.4495 | 0.9834 | 0.5864 |
| logistic_regression | 0.5278 | 0.3908 | 0.4507 | 0.9859 | 0.5882 |
| random_forest | 0.5291 | 0.3884 | 0.4480 | 0.9867 | 0.5879 |
| soft_voting_equal | 0.5291 | 0.3934 | 0.4516 | 0.9872 | 0.5888 |
| decision_tree | 0.5198 | 0.4180 | 0.4471 | 1.0090 | 0.6010 |
| logistic_regression_balanced | 0.4977 | 0.4371 | 0.4547 | 1.0123 | 0.6045 |
| hist_gradient_boosting | 0.5037 | 0.3868 | 0.4315 | 1.0322 | 0.6141 |
| majority_prior_baseline | 0.4394 | 0.2035 | 0.3333 | 1.0707 | 0.6480 |

### Cách diễn giải

- Poisson 1X2 có Log Loss test thấp nhất trong các mô hình chính (**0,9834**) và Accuracy **52,71%**.
- Random Forest có Accuracy test nhỉnh hơn một chút (**52,91%**), nhưng Log Loss cao hơn Poisson (**0,9867**). Không đổi mô hình theo kết quả test vì như vậy sẽ biến test thành validation.
- Logistic Regression có Accuracy **52,78%**, Log Loss **0,9859**.
- Logistic Regression `class_weight="balanced"` có Macro F1 tốt nhất trong các mô hình đã thử (**0,4371**) nhưng Accuracy thấp hơn (**49,77%**) và Log Loss cao hơn (**1,0123**). Mô hình này được giữ làm lựa chọn tham khảo khi quan tâm hơn đến các lớp ít gặp.
- Majority prior baseline có Accuracy **43,94%** và Log Loss **1,0707**. Các mô hình Machine Learning/Poisson cải thiện rõ so với baseline này theo các metric trên.

### Hạn chế lớp hòa

Đây là hạn chế cần trình bày rõ khi bảo vệ: mô hình Poisson được chọn theo Log Loss validation có **Draw recall = 0** khi chuyển xác suất sang một nhãn bằng argmax. Logistic Regression cân bằng lớp nâng Draw recall trên test lên **13,37%**, nhưng vẫn thấp. Điều đó cho thấy hệ thống chưa nhận diện tốt các trận hòa; xác suất hòa vẫn được hiển thị nhưng không nên tuyên bố mô hình dự đoán tốt cả ba lớp. Báo cáo chi tiết từng lớp nằm trong `classification_reports_all_models_test.json`.

### Dự đoán số bàn

- Home goals MAE: **0,9725**.
- Away goals MAE: **0,8921**.
- Combined goals MAE: **0,9323**.
- Home goals RMSE: **1,2233**; Away goals RMSE: **1,1492**.
- Goal model là hai Poisson Regression độc lập. Phân phối tỷ số được suy ra bằng tích hai phân phối Poisson, có chuẩn hóa trên lưới 0–9 bàn.

### Quyết định model triển khai

Validation chọn Logistic Regression là base classifier theo Log Loss (0,9811), nhưng khi thử blend với Poisson thì trọng số tối ưu trên lưới validation là base = 0,00 và Poisson = 1,00; Log Loss validation là 0,9739. Vì trọng số đó tương đương Poisson độc lập, pipeline ghi nhận mô hình triển khai là `poisson_1x2`, **không gọi nhầm đây là ensemble**. Weighted hybrid chưa được chứng minh có lợi hơn Poisson riêng trong lần chạy này.

## 6. File đầu ra quan trọng

- `Data/processed/premier_league_all_seasons.csv`: dữ liệu chuẩn hóa.
- `Data/processed/features_data.csv`: 9.950 dòng và 15 đặc trưng.
- `reports/data_quality_report.json`: thống kê chất lượng theo file nguồn.
- `reports/metrics.csv`: metric validation/test của các mô hình.
- `reports/test_predictions.csv`: dự đoán từng trận trong test.
- `reports/test_metrics_by_season.csv`: metric theo mùa trong test.
- `reports/goal_prediction_metrics.json`: MAE/RMSE/Poisson deviance.
- `reports/classification_reports_all_models_test.json`: precision/recall/F1 theo lớp cho từng mô hình.
- `reports/figures/`: biểu đồ Log Loss và confusion matrix.
- `models/artifacts/football_prediction_bundle.joblib`: bundle mô hình được website sử dụng.
- `models/artifacts/metadata.json`: feature list, model selection, mốc thời gian và metric.

## 7. Kiểm thử và mức độ xác minh

- `python -m compileall -q src models app.py run_project.py`: PASS.
- `python -m pytest -q`: **7 tests passed**.
- Chạy trọn pipeline `python run_project.py`: PASS trong môi trường thực thi hiện tại.
- Đã smoke-test hàm inference: load bundle, tạo 15 feature cho một fixture giả định và nhận xác suất 1X2 tổng bằng 1, cùng phân phối tỷ số Poisson.
- `app.py` đã được compile kiểm tra cú pháp; Streamlit chưa được cài trong môi trường thực thi hiện tại nên chưa thể xác minh việc mở giao diện trình duyệt. Cài theo `requirements.txt`, sau đó chạy `streamlit run app.py` trên máy Windows.

## 8. Hạn chế còn lại / việc nâng cấp sau

1. Chưa calibration xác suất bằng một tập riêng; không tuyên bố xác suất đã calibration tốt.
2. Validation hiện là một cửa sổ thời gian giữ riêng; chưa có nested walk-forward để ước lượng độ biến động của model selection.
3. Chưa bổ sung xG, đội hình, chấn thương, dữ liệu cầu thủ hoặc odds do chưa có nguồn lịch sử đủ điều kiện trong project này.
4. Poisson hiện là baseline độc lập, chưa phải Dixon–Coles hoặc Bivariate Poisson. Stacking và Deep Learning chưa triển khai vì chưa có bằng chứng validation cho thấy độ phức tạp bổ sung là đáng giá trong phạm vi project.
5. Chưa chạy kiểm định ý nghĩa thống kê cho chênh lệch giữa các mô hình; không nên diễn giải khác biệt vài phần nghìn là vượt trội chắc chắn.
6. Website cần được chạy trên máy Windows sau khi cài `requirements.txt`; nếu có lỗi do phiên bản thư viện, chạy lại `python run_project.py` trong chính `.venv` đó để tái tạo artifact.
7. Kiểm tra điều khoản nguồn dữ liệu trước khi commit hoặc chia sẻ công khai CSV. `.gitignore` loại `Data/raw/` mặc định.

## 9. Cách chạy trên Windows

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python run_project.py
python -m pytest -q
streamlit run app.py
```

**Kết luận:** phần dữ liệu, feature engineering, huấn luyện, đánh giá, model artifact, inference và code website đã được tạo/cập nhật. Pipeline và unit tests đã chạy qua trong môi trường hiện tại. Việc duy nhất chưa xác minh trực tiếp là khởi chạy giao diện Streamlit trên trình duyệt Windows của người dùng.
