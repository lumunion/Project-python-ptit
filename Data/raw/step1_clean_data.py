import pandas as pd
import warnings
warnings.filterwarnings('ignore')

print("⏳ Đang nạp toàn bộ lịch sử Ngoại hạng Anh từ năm 2000...")

# 1. Đọc dữ liệu thô (thêm low_memory=False để hệ thống không phàn nàn khi file quá lớn)
raw_data_path = 'all_seasons.csv'
df = pd.read_csv(raw_data_path, encoding='latin1', low_memory=False)

# 2. Đồng bộ tên cột ngày tháng (Đổi tên cột Date thành DateTime nếu cần)
if 'Date' in df.columns and 'DateTime' not in df.columns:
    df.rename(columns={'Date': 'DateTime'}, inplace=True)

# 3. Chuẩn hóa định dạng thời gian (Ép kiểu dữ liệu nghiêm ngặt)
df['DateTime'] = pd.to_datetime(df['DateTime'], format='mixed', dayfirst=True, errors='coerce')

# 4. CHỌN LỌC CỘT CỐT LÕI (Tương đương lệnh SELECT trong truy vấn)
# Lọc bỏ các thông tin dư thừa của các mùa cũ (như số thẻ phạt, trọng tài...)
core_columns = ['DateTime', 'HomeTeam', 'AwayTeam', 'FTR', 'FTHG', 'FTAG']
df = df[core_columns] 

# 5. XÓA DỮ LIỆU LỖI (Tương đương WHERE ... IS NOT NULL)
df = df.dropna()

# 6. SẮP XẾP THỜI GIAN (Tương đương ORDER BY DateTime ASC)
# Đây là bước sống còn để AI tính toán chính xác phong độ 5 trận gần nhất
df = df.sort_values(by='DateTime').reset_index(drop=True)

# 7. Lưu kết quả ra một bảng dữ liệu sạch tinh tươm
clean_data_path = 'cleaned_all_seasons.csv'
df.to_csv(clean_data_path, index=False)

print(f"✅ HOÀN TẤT! Đã lưu dữ liệu chuẩn hóa vào '{clean_data_path}'.")
print(f"📅 Dữ liệu đã sẵn sàng, kéo dài từ năm {df['DateTime'].dt.year.min()} đến năm {df['DateTime'].dt.year.max()}.")
print(f"📊 Tổng số trận đấu hợp lệ để huấn luyện: {len(df)} trận.")