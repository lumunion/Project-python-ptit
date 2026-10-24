"""Download missing Premier League season CSVs from Football-Data.co.uk.

Existing season files are not overwritten. Download only seasons that are
missing from Data/raw/Kết quả các năm.
"""
from __future__ import annotations

import re
import urllib.request
from pathlib import Path

from src.data.prepare_data import RAW_DIR


def existing_seasons(folder: Path) -> set[str]:
    seasons: set[str] = set()
    for path in folder.glob("*.csv"):
        stem = path.stem
        match = re.fullmatch(r"(\d{4})-(\d{4})", stem)
        if match:
            seasons.add(match.group(1)[-2:] + match.group(2)[-2:])
            continue
        match = re.fullmatch(r"EPL(\d{4})", stem, flags=re.IGNORECASE)
        if match:
            seasons.add(match.group(1))
    return seasons


def download_missing_seasons(start_year: int = 2000, end_year: int = 2026) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = existing_seasons(RAW_DIR)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; student-research-project/1.0)"}
    for year in range(start_year, end_year + 1):
        season = f"{year % 100:02d}{(year + 1) % 100:02d}"
        if season in downloaded:
            print(f"[SKIP] Đã có mùa {year}-{year + 1}")
            continue
        url = f"https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
        destination = RAW_DIR / f"EPL{season}.csv"
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read()
            if len(payload) < 100:
                raise ValueError("Phản hồi quá nhỏ, có thể không phải CSV hợp lệ")
            destination.write_bytes(payload)
            print(f"[OK] Đã tải mùa {year}-{year + 1}: {destination.name}")
        except Exception as exc:  # network/source availability varies by season
            print(f"[WARN] Không tải được mùa {year}-{year + 1}: {exc}")


if __name__ == "__main__":
    download_missing_seasons()
