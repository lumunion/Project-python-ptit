import pandas as pd
import glob

print("⏳ Đang quét các file dữ liệu...")
danh_sach_file = glob.glob("data_seasons/*.csv")

cac_bang_du_lieu = []

for file in danh_sach_file:
    try:
        # Thêm on_bad_lines='skip' để tự động bỏ qua các dòng bị lỗi dấu phẩy
        df_tam = pd.read_csv(file, encoding='latin1', on_bad_lines='skip')
        cac_bang_du_lieu.append(df_tam)
        print(f"✔ Đã đọc thành công: {file}")
    except Exception as e:
        print(f"❌ File bị lỗi cấu trúc: {file} (Lỗi: {e})")

if cac_bang_du_lieu:
    du_lieu_tong = pd.concat(cac_bang_du_lieu, ignore_index=True)
    du_lieu_tong.to_csv("all_seasons.csv", index=False)
    print(f"\n✅ HOÀN TẤT! Đã gộp thành công {len(cac_bang_du_lieu)} file hợp lệ.")
    print(f"📁 File tổng hợp được lưu tại: all_seasons.csv")
else:
    print("\n❌ Không đọc được file nào cả.")