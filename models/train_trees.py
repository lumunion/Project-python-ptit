import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib

def train_tree_models():
    print("=" * 60)
    print("BƯỚC 6: HUẤN LUYỆN DECISION TREE & RANDOM FOREST")
    print("=" * 60)

    # 1. Đọc dữ liệu và chia tập Train/Test giống hệt Bước 5
    # Để đảm bảo tính công bằng khi so sánh các mô hình
    data_path = os.path.join("Data", "processed", "features_data.csv")
    df = pd.read_csv(data_path)
    
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by='Date').reset_index(drop=True)

    features = ['Home_Form_Pts', 'Away_Form_Pts']
    target = 'FTR'

    split_index = int(len(df) * 0.8)
    train_data = df.iloc[:split_index]
    test_data = df.iloc[split_index:]

    X_train, y_train = train_data[features], train_data[target]
    X_test, y_test = test_data[features], test_data[target]

    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # ==========================================
    # 2. Huấn luyện DECISION TREE
    # ==========================================
    print("Đang huấn luyện Decision Tree...")
    # max_depth=5 để tránh cây mọc quá sâu dẫn đến học vẹt (overfitting)
    dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt_model.fit(X_train, y_train)

    dt_pred = dt_model.predict(X_test)
    dt_proba = dt_model.predict_proba(X_test)
    dt_acc = accuracy_score(y_test, dt_pred)
    dt_loss = log_loss(y_test, dt_proba)

    print(f"👉 Accuracy của Decision Tree : {dt_acc * 100:.2f}%")
    print(f"👉 Log Loss của Decision Tree : {dt_loss:.4f}")
    
    dt_path = os.path.join(model_dir, "decision_tree_model.pkl")
    joblib.dump(dt_model, dt_path)

    print("-" * 60)

    # ==========================================
    # 3. Huấn luyện RANDOM FOREST
    # ==========================================
    print("Đang huấn luyện Random Forest...")
    # n_estimators=100: Tạo ra 100 cây quyết định
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rf_model.fit(X_train, y_train)

    rf_pred = rf_model.predict(X_test)
    rf_proba = rf_model.predict_proba(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_loss = log_loss(y_test, rf_proba)

    print(f"👉 Accuracy của Random Forest : {rf_acc * 100:.2f}%")
    print(f"👉 Log Loss của Random Forest : {rf_loss:.4f}")

    rf_path = os.path.join(model_dir, "random_forest_model.pkl")
    joblib.dump(rf_model, rf_path)

    print("=" * 60)
    print(f"[OK] Đã lưu 2 mô hình tại thư mục: {model_dir}")

if __name__ == "__main__":
    train_tree_models()