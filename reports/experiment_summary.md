# Báo cáo thực nghiệm tự động

## Thiết kế đánh giá

- Train: 6,967 trận, 2000-08-19 đến 2018-11-24
- Validation: 1,490 trận, 2018-11-25 đến 2022-10-16
- Test: 1,493 trận, 2022-10-18 đến 2026-10-19
- Chia theo ngày lịch để không tách các trận cùng ngày sang hai tập.
- Validation dùng để chọn base strategy và trọng số hybrid; test không tham gia chọn mô hình.

## Quyết định mô hình

- Base strategy: `single:logistic_regression` (validation log loss 0.9811)
- Hybrid weight: base=0.00, Poisson=1.00
- Hybrid được chọn: `False` (validation log loss 0.9739)
- Poisson độc lập được chọn: `True` (validation log loss 0.9739)
- Mô hình được website sử dụng: `poisson_1x2`

## Kết quả test

- Accuracy: 0.5271
- Macro F1: 0.3897
- Balanced accuracy: 0.4495
- Log loss: 0.9834
- Multiclass Brier score: 0.5864
- Recall lớp hòa (D) của mô hình được chọn: 0.0000
- Xem `classification_reports_all_models_test.json` để so sánh precision/recall/F1 của từng lớp cho tất cả mô hình.

## Kết quả dự đoán số bàn

- Home goals MAE: 0.9725
- Away goals MAE: 0.8921
- Combined goals MAE: 0.9323
- Home goals RMSE: 1.2233
- Away goals RMSE: 1.1492

## Giới hạn

- Chỉ sử dụng kết quả lịch sử Football-Data.co.uk và các đặc trưng tạo từ kết quả trước trận.
- Chưa đưa dữ liệu chấn thương, đội hình, xG hoặc odds vào pipeline.
- Kết quả test là backtest tuần tự; kết quả của các trận trước trong giai đoạn test được dùng để cập nhật trạng thái cho các ngày sau, đúng với kịch bản dự đoán tuần tự.
- Chưa có kiểm định ý nghĩa thống kê giữa các mô hình; chênh lệch metric nhỏ cần được diễn giải thận trọng.

Các số liệu trong file này được sinh trực tiếp từ lần chạy pipeline, không phải số liệu minh họa.