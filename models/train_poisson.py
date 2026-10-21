import os
import math
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, log_loss
import joblib

def poisson_probability(lmbda, k):
    # Tính xác suất ghi k bàn theo công thức Poisson
    return math.exp(-lmbda) * (lmbda ** k) / math.factorial(k)

def train_poisson():
    print("=" * 60)
    print("BƯỚC 7: HUẤN LUYỆN MÔ HÌNH POISSON (DỰ ĐOÁN BÀN THẮNG)")
    print("=" * 60)

    # 1. Đọc dữ liệu thô ban đầu (chưa qua xử lý đặc trưng điểm số)
    data_path = os.path.join("Data", "raw", "premier_league_2017_2027.csv")
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Ép kiểu ngày tháng và sắp xếp để chống rò rỉ dữ liệu
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Từ điển lưu lịch sử BÀN THẮNG của mỗi đội
    team_goals_scored = {}

    y_true = []
    y_pred = []
    y_pred_proba = []

    # Chia tập Test giống các mô hình trước (20% cuối)
    split_index = int(len(df) * 0.8)

    print("Đang quét dữ liệu và tính toán phân phối Poisson...")

    # Quét qua từng trận đấu
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        if home not in team_goals_scored:
            team_goals_scored[home] = []
        if away not in team_goals_scored:
            team_goals_scored[away] = []

        # --- A. TÍNH KỲ VỌNG BÀN THẮNG (LAMBDA) TỪ LỊCH SỬ ---
        # Lấy tối đa 10 trận gần nhất để tính trung bình bàn thắng
        h_history = team_goals_scored[home][-10:]
        a_history = team_goals_scored[away][-10:]

        home_lambda = sum(h_history) / len(h_history) if len(h_history) > 0 else 1.0
        away_lambda = sum(a_history) / len(a_history) if len(a_history) > 0 else 1.0

        # --- B. DỰ ĐOÁN (Chỉ dự đoán cho các trận thuộc tập Test) ---
        if index >= split_index:
            prob_H, prob_D, prob_A = 0.0, 0.0, 0.0

            # Lập ma trận tỷ số từ 0-0 đến 5-5 (Giới hạn tối đa 5 bàn để dễ tính)
            for i in range(6): # i là số bàn của đội nhà
                for j in range(6): # j là số bàn của đội khách
                    # Xác suất xảy ra tỷ số i-j
                    prob = poisson_probability(home_lambda, i) * poisson_probability(away_lambda, j)
                    
                    if i > j:
                        prob_H += prob
                    elif i == j:
                        prob_D += prob
                    else:
                        prob_A += prob

            # Chuẩn hóa để tổng 3 xác suất đúng bằng 100%
            total_prob = prob_H + prob_D + prob_A
            prob_H /= total_prob
            prob_D /= total_prob
            prob_A /= total_prob

            # Căn cứ vào xác suất cao nhất để chốt nhãn dự đoán
            if prob_H > prob_D and prob_H > prob_A:
                pred = 'H'
            elif prob_A > prob_H and prob_A > prob_D:
                pred = 'A'
            else:
                pred = 'D'

            y_true.append(row['FTR'])
            y_pred.append(pred)
            # Lưu mảng xác suất theo thứ tự A, D, H (chuẩn của scikit-learn)
            y_pred_proba.append([prob_A, prob_D, prob_H])

        # --- C. CẬP NHẬT KẾT QUẢ VÀO LỊCH SỬ ---
        # Phải cập nhật sau khi dự đoán xong để không bị Data Leakage
        team_goals_scored[home].append(row['FTHG'])
        team_goals_scored[away].append(row['FTAG'])

    # 2. Đánh giá
    acc = accuracy_score(y_true, y_pred)
    loss = log_loss(y_true, y_pred_proba)

    print(f"\n👉 Accuracy của Poisson Model: {acc * 100:.2f}%")
    print(f"👉 Log Loss của Poisson Model: {loss:.4f}")
    print("\nChi tiết (Classification Report):")
    print(classification_report(y_true, y_pred, zero_division=0))

    # 3. Lưu thông số Lambda cuối cùng của các đội để website sử dụng
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    
    final_lambdas = {
        team: (sum(goals[-10:]) / len(goals[-10:]) if len(goals[-10:]) > 0 else 1.0)
        for team, goals in team_goals_scored.items()
    }
    
    joblib.dump(final_lambdas, os.path.join(model_dir, "poisson_lambdas.pkl"))
    print("-" * 60)
    print(f"[OK] Đã lưu thông số Poisson tại: {os.path.join(model_dir, 'poisson_lambdas.pkl')}")

if __name__ == "__main__":
    train_poisson()