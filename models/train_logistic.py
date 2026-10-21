import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib

def train_logistic_regression():
    print("=" * 60)
    print("BƯỚC 5: HUẤN LUYỆN LOGISTIC REGRESSION")
    print("=" * 60)

    # 1. Đọc dữ liệu đã có đặc trưng
    data_path = os.path.join("Data", "processed", "features_data.csv")
    df = pd.read_csv(data_path)
    
    # Ép kiểu ngày tháng và sắp xếp
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    # 2. Xác định Đặc trưng (X) và Nhãn cần đoán (y)
    features = ['Home_Form_Pts', 'Away_Form_Pts']
    target = 'FTR'

    # 3. Chia dữ liệu theo thời gian (80% Train - 20% Test)
    split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    X_train = train_data[features]
    y_train = train_data[target]

    X_test = test_data[features]
    y_test = test_data[target]

    print(f"Số lượng trận để Học (Train): {len(X_train)}")
    print(f"Số lượng trận để Thi (Test): {len(X_test)}")
    print("-" * 60)

    # 4. Khởi tạo và Huấn luyện mô hình
    # max_iter=1000: Cho mô hình đủ thời gian hội tụ
    print("Đang huấn luyện mô hình Logistic Regression...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 5. Dự đoán trên tập Test
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) # Tính xác suất %

    # 6. Đánh giá
    acc = accuracy_score(y_test, y_pred)
    loss = log_loss(y_test, y_pred_proba)

    print(f"\n👉 Điểm Accuracy của Logistic Regression: {acc * 100:.2f}%")
    print(f"👉 Điểm Log Loss: {loss:.4f} (Số càng nhỏ chứng tỏ xác suất đoán càng tự tin)")
    print("\nChi tiết (Classification Report):")
    print(classification_report(y_test, y_pred, zero_division=0))

    # 7. Lưu mô hình ra file
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "logistic_model.pkl")
    
    joblib.dump(model, model_path)
    print("-" * 60)
    print(f"[OK] Đã lưu file mô hình tại: {model_path}")

if __name__ == "__main__":
    train_logistic_regression()