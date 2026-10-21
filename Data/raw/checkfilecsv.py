from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------
# 1. Đọc dữ liệu
# ---------------------------------------------------------------
# Tự tìm file CSV: ưu tiên Data/raw, sau đó thư mục Project
base = Path(__file__).resolve().parent
base = Path(__file__).resolve().parent          # Project/Data/raw
files = sorted(base.glob("premier_league_*.csv"))
if not files:
    raise FileNotFoundError(f"Không có file premier_league_*.csv trong {base}")

csv_path = files[-1]   # lấy file mới nhất theo tên
print("Đọc file:", csv_path)
df = pd.read_csv(csv_path, low_memory=False)

print("Đọc file:", csv_path)
df = pd.read_csv(csv_path, low_memory=False)

COLS = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]

# Kiểm tra các cột cần thiết có tồn tại không
missing_cols = [c for c in COLS if c not in df.columns]
if missing_cols:
    raise KeyError(f"Thiếu cột: {missing_cols}")

data = df[COLS].copy()

print("\n=== 1. THÔNG TIN CÁC CỘT CHÍNH ===")
print(data.dtypes)
print(data.head())

# Chuẩn hóa Date: file gộp nhiều mùa nên có thể lẫn dd/mm/yy và dd/mm/yyyy
data["Date_parsed"] = pd.to_datetime(data["Date"], dayfirst=True, errors="coerce")
bad_dates = data["Date_parsed"].isna().sum()
print(f"\nSố ngày không parse được: {bad_dates}")
print("Khoảng thời gian:", data["Date_parsed"].min(), "->", data["Date_parsed"].max())

# ---------------------------------------------------------------
# 2. Thống kê
# ---------------------------------------------------------------
print("\n=== 2a. SỐ DÒNG / SỐ CỘT ===")
print("Tổng số dòng :", len(df))
print("Tổng số cột  :", df.shape[1])

print("\n=== 2b. DỮ LIỆU THIẾU (6 cột chính) ===")
na = data[COLS].isna().sum().to_frame("so_thieu")
na["ty_le_%"] = (na["so_thieu"] / len(data) * 100).round(2)
print(na)

print("\n=== 2c. DỮ LIỆU TRÙNG LẶP ===")
dup_full = df.duplicated().sum()
dup_key = data.duplicated(subset=["Date", "HomeTeam", "AwayTeam"]).sum()
print("Dòng trùng toàn bộ cột          :", dup_full)
print("Trận trùng (Date+Home+Away)     :", dup_key)
if dup_key:
    print(data[data.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep=False)]
          .sort_values(["Date", "HomeTeam"]).head(20))

print("\n=== 2d. PHÂN PHỐI KẾT QUẢ (FTR: H = chủ nhà thắng, D = hòa, A = khách thắng) ===")
ftr = data["FTR"].value_counts(dropna=False)
ftr_pct = (data["FTR"].value_counts(normalize=True, dropna=False) * 100).round(2)
print(pd.DataFrame({"so_tran": ftr, "ty_le_%": ftr_pct}))

# Kiểm tra FTR có khớp với tỉ số FTHG/FTAG không
expected = pd.Series("D", index=data.index)
expected[data["FTHG"] > data["FTAG"]] = "H"
expected[data["FTHG"] < data["FTAG"]] = "A"
mismatch = ((expected != data["FTR"]) & data[["FTHG", "FTAG", "FTR"]].notna().all(axis=1)).sum()
print("\nSố dòng FTR không khớp tỉ số:", mismatch)

# Số trận theo mùa (mỗi mùa Premier League chuẩn là 380 trận)
season_start = data["Date_parsed"].dt.year.where(data["Date_parsed"].dt.month >= 7,
                                                 data["Date_parsed"].dt.year - 1)
print("\n=== Số trận theo mùa (mỗi mùa đủ = 380) ===")
print(season_start.value_counts().sort_index())