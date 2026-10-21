import os
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

def evaluate_baseline():
    print("=" * 60)
    print("ĐÁNH GIÁ MÔ HÌNH CƠ SỞ (BASELINE)")
    print("=" * 60)

    data_path = os.path.join("Data", "raw", "premier_league_2017_2027.csv")
    
    if not os.path.exists(data_path):
        print(f"[LỖI] Không tìm thấy file: {data_path}")
        return

    # 1. Đọc dữ liệu    
    df = pd.read_csv(data_path, encoding='latin1')
    
    # 2. Xử lý ngày tháng và sắp xếp chuẩn theo thời gian (Chống Data Leakage)
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.sort_values(by='Date').reset_index(drop=True)
    
    # 3. Chia tập Train và Test theo thời gian (Temporal Split)
    # Chúng ta dùng 80% dữ liệu cũ làm Train, 20% dữ liệu mới nhất làm Test
    split_index = int(len(df) * 0.8)
    
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]
    
    print(f"Tổng số trận đấu: {len(df)}")
    print(f"Tập huấn luyện (Train): {len(train_data)} trận (từ {train_data['Date'].dt.date.iloc[0]} đến {train_data['Date'].dt.date.iloc[-1]})")
    print(f"Tập kiểm tra (Test): {len(test_data)} trận (từ {test_data['Date'].dt.date.iloc[0]} đến {test_data['Date'].dt.date.iloc[-1]})")
    print("-" * 60)

    # 4. Xây dựng mô hình Baseline: Luôn dự đoán nhãn 'H' (Chủ nhà thắng)
    # y_true là kết quả thực tế trên tập Test
    y_true = test_data['FTR'].values 
    # y_pred_baseline là mảng toàn chữ 'H' có độ dài bằng tập Test
    y_pred_baseline = ['H'] * len(test_data)
    
    # 5. Đánh giá
    acc = accuracy_score(y_true, y_pred_baseline)
    print(f"👉 Điểm Accuracy của Baseline: {acc * 100:.2f}%")
    print("\nChi tiết (Classification Report):")
    
    # zero_division=0 để tránh cảnh báo vì mô hình không bao giờ đoán D hoặc A
    print(classification_report(y_true, y_pred_baseline, zero_division=0))

if __name__ == "__main__":
    evaluate_baseline()