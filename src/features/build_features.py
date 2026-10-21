import os
import pandas as pd

def build_features():
    print("=" * 60)
    print("BƯỚC 4: THIẾT KẾ ĐẶC TRƯNG - TÍNH ĐIỂM PHONG ĐỘ")
    print("=" * 60)

    # 1. Đọc dữ liệu đã được gộp
    data_path = os.path.join("Data", "raw", "premier_league_2017_2027.csv")
    df = pd.read_csv(data_path, encoding='latin1')
    
    # Ép kiểu ngày tháng và sắp xếp CHUẨN XÁC theo thời gian
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, format='mixed')
    df = df.sort_values(by='Date').reset_index(drop=True)

    # Khởi tạo 2 danh sách rỗng để lưu đặc trưng mới
    home_form = []
    away_form = []

    # Dictionary lưu trữ lịch sử điểm số của mỗi đội
    # Quy tắc: Thắng = 3 điểm, Hòa = 1 điểm, Thua = 0 điểm
    team_points_history = {}

    # Dùng vòng lặp quét qua từng trận đấu theo trình tự thời gian
    for index, row in df.iterrows():
        home = row['HomeTeam']
        away = row['AwayTeam']

        # Nếu đội bóng lần đầu xuất hiện, tạo danh sách lịch sử trống cho đội đó
        if home not in team_points_history:
            team_points_history[home] = []
        if away not in team_points_history:
            team_points_history[away] = []

        # --- A. TÍNH TOÁN ĐẶC TRƯNG TỪ LỊCH SỬ TRƯỚC TRẬN ĐẤU ---
        # Lấy tối đa 5 trận gần nhất trong lịch sử
        h_history = team_points_history[home][-5:]
        a_history = team_points_history[away][-5:]

        # Tính điểm trung bình mỗi trận trong 5 trận gần nhất.
        # Nếu đội chưa đá trận nào (đầu mùa), tạm gán phong độ = 0
        h_form = sum(h_history) / len(h_history) if len(h_history) > 0 else 0
        a_form = sum(a_history) / len(a_history) if len(a_history) > 0 else 0

        # Ghi nhận đặc trưng cho trận đấu hiện tại
        home_form.append(h_form)
        away_form.append(a_form)

        # --- B. CẬP NHẬT KẾT QUẢ TRẬN NÀY VÀO LỊCH SỬ (ĐỂ DÙNG CHO TRẬN SAU) ---
        if row['FTR'] == 'H': # Chủ nhà thắng
            team_points_history[home].append(3)
            team_points_history[away].append(0)
        elif row['FTR'] == 'A': # Khách thắng
            team_points_history[home].append(0)
            team_points_history[away].append(3)
        else: # Hòa
            team_points_history[home].append(1)
            team_points_history[away].append(1)

    # Gắn 2 cột đặc trưng mới vào DataFrame
    df['Home_Form_Pts'] = home_form
    df['Away_Form_Pts'] = away_form

    # 2. Xóa các trận đấu mà các đội chưa có lịch sử (chưa đá đủ 1 trận)
    # Vì phong độ = 0 lúc này không phản ánh đúng thực lực mà là do thiếu dữ liệu
    print(f"Số trận trước khi lọc: {len(df)}")
    df = df[(df['Home_Form_Pts'] > 0) | (df['Away_Form_Pts'] > 0)].reset_index(drop=True)
    print(f"Số trận sau khi lọc các trận đầu tiên thiếu dữ liệu: {len(df)}\n")

    print("Mẫu dữ liệu sau khi thêm đặc trưng phong độ (10 dòng đầu):")
    cols_to_show = ['Date', 'HomeTeam', 'AwayTeam', 'FTR', 'Home_Form_Pts', 'Away_Form_Pts']
    print(df[cols_to_show].head(10))

    # 3. Lưu file dữ liệu đã có Features
    processed_dir = os.path.join("Data", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, "features_data.csv")
    df.to_csv(out_path, index=False)
    
    print(f"\n[OK] Đã tạo đặc trưng và lưu thành công tại: {out_path}")

if __name__ == "__main__":
    build_features()