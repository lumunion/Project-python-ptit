# phải VPN mới tải được


import os
import urllib.request

def download_premier_league_data():
    print("=" * 55)
    print("TIẾN HÀNH TẢI DỮ LIỆU PREMIER LEAGUE (E0.csv)")
    print("=" * 55)
    raw_dir = os.path.join("data", "raw")
    os.makedirs(raw_dir, exist_ok=True)
    seasons = [
        "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526", "2627"
    ]
    base_url = "https://www.football-data.co.uk/mmz4281/{}/E0.csv"

    for season in seasons:
        url = base_url.format(season)
        file_path = os.path.join(raw_dir, f"E0_{season}.csv")
        
        print(f"Đang tải mùa {season[:2]}/{season[2:]} từ: {url}")
        
        try:
            urllib.request.urlretrieve(url, file_path)
            print(f"  -> Đã lưu thành công: {file_path}")
        except Exception as e:
            print(f"  -> [LỖI] Không thể tải mùa {season}: {e}")

    print("\nHOÀN THÀNH TẢI DỮ LIỆU!")

if __name__ == "__main__":
    download_premier_league_data()