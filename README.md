# Football Match Outcome Prediction Using Machine Learning

Project dự đoán kết quả trận đấu Premier League theo ba lớp:

- `H`: đội nhà thắng.
- `D`: hòa.
- `A`: đội khách thắng.

Mục tiêu là tạo một pipeline có thể chạy lại, có chia dữ liệu theo thời gian, so sánh nhiều thuật toán và website Streamlit dùng artifact đã huấn luyện.

## 1. Tình trạng phiên bản này

Phiên bản được hoàn thiện dựa trên cấu trúc project và dữ liệu đã được cung cấp. Pipeline đọc dữ liệu từ `Data/raw/Kết quả các năm/`, tạo dữ liệu chuẩn, sinh đặc trưng trước trận, huấn luyện và đánh giá các mô hình.

Dữ liệu có trong project khi đóng gói gồm các mùa 2000/01–2025/26 và một phần mùa 2026/27. Sau khi chuẩn hóa, lần chạy trên dữ liệu đính kèm thu được **9.950 trận hợp lệ**: 26 mùa hoàn chỉnh (9.880 trận) và 70 trận của mùa 2026/27. Báo cáo `reports/data_quality_report.json` ghi chi tiết số dòng từng file.

Các vấn đề được xử lý so với code cũ:

1. Đọc cột theo tên thay vì vị trí cố định, vì file từ 2019/20 có thêm cột `Time`.
2. Giữ những dòng có thêm trường ở cuối, tránh làm mất 90 trận từ hai file EPL0304/EPL0405.
3. Bỏ dòng trống/thiếu kết quả và loại trùng theo ngày + đội nhà + đội khách.
4. Tạo đặc trưng từ lịch sử trước trận; nếu không có giờ thi đấu, toàn bộ trận cùng ngày được feature-score trước khi cập nhật kết quả của ngày đó.
5. Thay phép chia train/test bằng train/validation/test theo các mốc ngày, không tách trận cùng ngày sang hai tập.
6. Chọn mô hình và trọng số ensemble trên validation; test chỉ dùng để đánh giá cuối cùng.
7. Sửa Poisson thành hai mô hình dự đoán bàn đội nhà/đội khách, sau đó tạo phân phối tỷ số và xác suất 1X2.

## 2. Cấu trúc chính

```text
football-match-prediction-complete/
│
├── Data/
│   ├── raw/
│   │   └── Kết quả các năm/
│   │       ├── EPL0001.csv
│   │       ├── EPL0102.csv
│   │       ├── ...
│   │       ├── 2024-2025.csv
│   │       ├── 2025-2026.csv
│   │       └── 2026-2027.csv
│   │
│   └── processed/
│       ├── premier_league_all_seasons.csv
│       └── features_data.csv
│
├── src/
│   ├── data/
│   │   ├── prepare_data.py
│   │   ├── autodowload.py
│   │   └── checkfilecsv.py
│   │
│   ├── features/
│   │   └── build_features.py
│   │
│   ├── models/
│   │   ├── train_models.py
│   │   └── inference.py
│   │
│   ├── pipeline/
│   │   └── run_all.py
│   │
│   └── check_env.py
│
├── models/
│   └── artifacts/
│       ├── football_prediction_bundle.joblib
│       └── metadata.json
│
├── reports/
│   ├── metrics.csv
│   ├── test_predictions.csv
│   ├── data_quality_report.json
│   ├── classification_report_test.json
│   ├── goal_prediction_metrics.json
│   ├── experiment_summary.md
│   └── figures/
│
├── tests/
│   ├── test_data_and_features.py
│   └── test_model_logic.py
│
├── app.py
├── run_project.py
├── requirements.txt
├── pytest.ini
├── README.md
└── .gitignore
```

## 3. Cài đặt trên Windows

Mở PowerShell tại thư mục project:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường, có thể chạy trực tiếp:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra môi trường:

```powershell
python -m src.check_env
```

## 4. Chạy toàn bộ pipeline

Đảm bảo các CSV nguồn nằm trong:

```text
Data/raw/Kết quả các năm/
```

Sau đó chạy tại thư mục gốc project:

```powershell
python run_project.py
```

Hoặc chạy riêng từng bước:

```powershell
python -m src.data.prepare_data
python -m src.features.build_features
python -m src.models.train_models
```

Đầu ra chính:

- `Data/processed/premier_league_all_seasons.csv`
- `Data/processed/features_data.csv`
- `reports/data_quality_report.json`
- `reports/metrics.csv`
- `reports/test_predictions.csv`
- `reports/test_metrics_by_season.csv`
- `reports/goal_prediction_metrics.json`
- `reports/experiment_summary.md`
- `reports/PROJECT_COMPLETION_REPORT.md`
- `reports/figures/`
- `models/artifacts/football_prediction_bundle.joblib`
- `models/artifacts/metadata.json`

Mỗi lần chạy sẽ ghi lại metric mới từ dữ liệu cục bộ hiện có.

## 5. Chia dữ liệu và đánh giá

Pipeline dùng tỷ lệ mục tiêu 70% train, 15% validation, 15% test theo thứ tự thời gian. Các mốc được điều chỉnh về ngày lịch để không chia các trận trong cùng một ngày sang hai tập khác nhau.

- Train: fit tham số mô hình.
- Validation: chọn mô hình, so sánh soft voting và tìm trọng số kết hợp Poisson.
- Test: đánh giá cuối cùng; không dùng để chọn mô hình hoặc trọng số.

Các mô hình phân loại:

- Majority-prior baseline.
- Logistic Regression (có StandardScaler).
- Logistic Regression `class_weight="balanced"` để đối chiếu khả năng nhận diện lớp hòa.
- Decision Tree.
- Random Forest.
- HistGradientBoosting.
- Soft voting trung bình xác suất.
- Weighted hybrid giữa chiến lược classifier được chọn và xác suất 1X2 suy ra từ Poisson.

Weighted hybrid chỉ được chọn để triển khai khi log loss validation tốt hơn base strategy ít nhất 0,001 và trọng số thật sự nằm giữa 0 và 1. Nếu trọng số Poisson bằng 1, pipeline ghi rõ Poisson độc lập được chọn, không gọi nhầm là ensemble. Nếu không có cải thiện, pipeline giữ mô hình cơ sở thay vì ép ensemble.

Metric phân loại: Accuracy, Macro F1, Balanced Accuracy, Log Loss và multiclass Brier score. Mô hình bàn thắng có MAE, RMSE và Poisson deviance cho bàn đội nhà/đội khách.

## 6. Feature engineering và chống leakage

Các feature chỉ dùng dữ liệu trận đã kết thúc trước thời điểm dự đoán:

- Điểm trung bình 5 trận gần nhất của mỗi đội (có prior smoothing).
- Bàn thắng và bàn thua trung bình 5 trận gần nhất.
- Số trận đã chơi trước trận.
- Số ngày nghỉ.
- Elo đội nhà, Elo đội khách và chênh lệch Elo có điều chỉnh lợi thế sân nhà.
- Chênh lệch tấn công tương đối.

Không đưa `FTHG`, `FTAG`, `FTR` hoặc thống kê sau trận của chính fixture vào feature columns. Tất cả mùa giải được xử lý theo batch ngày: feature của toàn bộ trận cùng ngày được tính trước khi cập nhật kết quả của ngày đó. Quy tắc này bảo thủ hơn việc suy luận thời điểm kết thúc trận từ giờ kickoff.

Các prior khởi tạo hiện tại là giả định kỹ thuật cố định (1,5 điểm/trận, 1,35 bàn/trận, Elo 1500 và home advantage Elo 60). Đây là giá trị khởi tạo, không phải tham số đã được tối ưu. Có thể nghiên cứu tuning sau này trên validation.

## 7. Chạy website

Sau khi chạy pipeline thành công:

```powershell
streamlit run app.py
```

Website có ba khu vực:

1. Dự đoán trận đấu: chọn đội nhà, đội khách và ngày dự đoán; hiển thị xác suất 1X2, bàn kỳ vọng Poisson và các tỷ số có xác suất cao.
2. Đánh giá mô hình: xem metric validation/test, hiệu năng theo mùa và biểu đồ được sinh từ lần chạy pipeline.
3. Phương pháp: xem feature, nguồn dữ liệu, cách chia tập và giới hạn.

Với ngày dự đoán, dữ liệu lịch sử cùng ngày không được sử dụng; chỉ những trận có ngày nhỏ hơn ngày dự đoán được đưa vào trạng thái đội bóng.

## 8. Kiểm thử

Chạy:

```powershell
pytest -q
```

Các test kiểm tra đọc CSV có extra fields, không leakage trong cùng ngày, feature columns, temporal split và xác suất Poisson được chuẩn hóa.

## 9. Tải dữ liệu bổ sung

Project đã có CSV nguồn trong `Data/raw/Kết quả các năm/`. Không cần tải lại nếu chỉ muốn chạy project hiện tại.

Khi cần tải mùa còn thiếu, chạy:

```powershell
python -m src.data.autodowload
```

Script bỏ qua mùa đã tồn tại và chỉ thử tải các mùa bị thiếu. Lưu ý kiểm tra điều khoản sử dụng/redistribution của nguồn dữ liệu trước khi công khai dữ liệu gốc trong repository.

## 10. Hạn chế cần ghi đúng trong báo cáo

- Chỉ dùng dữ liệu kết quả lịch sử Premier League. Chưa có xG, chấn thương, đội hình, cầu thủ hoặc odds trong pipeline.
- Poisson giả định số bàn của hai đội độc lập có điều kiện theo feature; đây là mô hình cơ sở, chưa phải Dixon–Coles.
- Chưa triển khai calibration chuyên biệt; không nên diễn giải xác suất là đã được hiệu chỉnh tốt nếu chưa có kiểm định calibration riêng.
- Kết quả test là backtest tuần tự. Kết quả của các trận đã diễn ra trước đó trong test period được dùng để cập nhật form/Elo cho các ngày tiếp theo; không dùng kết quả của chính trận đang dự đoán.
- Một chênh lệch metric nhỏ không đủ để kết luận mô hình vượt trội có ý nghĩa thống kê.
- Mùa 2026/27 chỉ là dữ liệu một phần đến ngày cuối cùng có trong file nguồn; không xem đây là mùa giải hoàn chỉnh.

## 11. Ghi chú GitHub

`.gitignore` mặc định loại `Data/raw/`, `Data/processed/*.csv` và các model artifacts khỏi Git để tránh đẩy dữ liệu/artifacts lớn hoặc dữ liệu có điều kiện sử dụng. Nếu repository private và bạn có quyền chia sẻ các CSV đó, hãy tự quyết định có đưa chúng vào Git hay cung cấp hướng dẫn tải riêng. Không commit `.venv/`.
