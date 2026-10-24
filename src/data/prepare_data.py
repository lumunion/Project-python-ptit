"""Đọc các file csv từ "Data/raw/kết quả các năm" và 
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "Data" / "raw" / "Kết quả các năm"
PROCESSED_DIR = PROJECT_ROOT / "Data" / "processed"
REPORT_DIR = PROJECT_ROOT / "reports"
REQUIRED_COLUMNS = ("Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR")


def _clean_header(value: Any) -> str:
    return str(value or "").replace("\ufeff", "").strip()


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).replace("\xa0", " ").strip()
    return result or None


def _season_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    match = re.fullmatch(r"(\d{4})-(\d{4})", stem)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    match = re.fullmatch(r"EPL(\d{2})(\d{2})", stem, flags=re.IGNORECASE)
    if match:
        start_two, end_two = map(int, match.groups())
        start_year = 2000 + start_two if start_two < 80 else 1900 + start_two
        end_year = start_year + 1
        # The two-digit suffix is checked but the next-year convention is authoritative.
        if end_year % 100 != end_two:
            return f"{start_year}-{start_year + 1}"
        return f"{start_year}-{end_year}"
    return "unknown"


def _read_one_file(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    stats: dict[str, Any] = {
        "file": path.name,
        "rows_seen": 0,
        "rows_kept": 0,
        "rows_skipped_missing_fields": 0,
        "rows_skipped_invalid_date": 0,
        "rows_skipped_invalid_score": 0,
        "rows_with_extra_fields": 0,
        "ftr_mismatches": 0,
    }
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="latin1", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV không có header: {path}")
        original_headers = reader.fieldnames
        normalized_headers = []
        used_headers = set()
        for index, item in enumerate(original_headers):
            name = _clean_header(item) or f"__unnamed_{index}"
            base_name = name
            suffix = 1
            while name in used_headers:
                name = f"{base_name}_{suffix}"
                suffix += 1
            used_headers.add(name)
            normalized_headers.append(name)
        reader.fieldnames = normalized_headers
        missing = sorted(set(REQUIRED_COLUMNS) - set(normalized_headers))
        if missing:
            raise ValueError(f"{path.name} thiếu cột bắt buộc: {missing}")

        for row in reader:
            stats["rows_seen"] += 1
            if None in row and row[None]:
                stats["rows_with_extra_fields"] += 1
            raw_date = _clean_text(row.get("Date"))
            home = _clean_text(row.get("HomeTeam"))
            away = _clean_text(row.get("AwayTeam"))
            raw_hg = _clean_text(row.get("FTHG"))
            raw_ag = _clean_text(row.get("FTAG"))
            raw_ftr = (_clean_text(row.get("FTR")) or "").upper()
            if not all((raw_date, home, away, raw_hg is not None, raw_ag is not None, raw_ftr)):
                stats["rows_skipped_missing_fields"] += 1
                continue

            date = pd.to_datetime(raw_date, dayfirst=True, format="mixed", errors="coerce")
            if pd.isna(date):
                stats["rows_skipped_invalid_date"] += 1
                continue
            try:
                home_goals = int(float(raw_hg))
                away_goals = int(float(raw_ag))
                if home_goals < 0 or away_goals < 0:
                    raise ValueError("negative goals")
            except (TypeError, ValueError, OverflowError):
                stats["rows_skipped_invalid_score"] += 1
                continue

            calculated_ftr = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
            if raw_ftr not in {"H", "D", "A"} or raw_ftr != calculated_ftr:
                stats["ftr_mismatches"] += 1
                # Keep the score-derived label as the canonical target only when the source label is invalid.
                if raw_ftr not in {"H", "D", "A"}:
                    raw_ftr = calculated_ftr
                else:
                    # Do not silently keep contradictory target/score records.
                    continue

            time_value = _clean_text(row.get("Time"))
            time_parsed = None
            if time_value:
                parsed_time = pd.to_datetime(time_value, format="mixed", errors="coerce")
                if not pd.isna(parsed_time):
                    time_parsed = parsed_time.strftime("%H:%M:%S")

            records.append(
                {
                    "Date": date.normalize(),
                    "Time": time_parsed,
                    "Kickoff": pd.Timestamp.combine(date.date(), pd.Timestamp(time_parsed).time())
                    if time_parsed
                    else date.normalize(),
                    "Season": _season_from_filename(path.name),
                    "HomeTeam": home,
                    "AwayTeam": away,
                    "FTHG": home_goals,
                    "FTAG": away_goals,
                    "FTR": calculated_ftr,
                    "Source_File": path.name,
                }
            )
            stats["rows_kept"] += 1
    return records, stats


def prepare_data(raw_dir: Path = RAW_DIR, output_path: Path | None = None) -> pd.DataFrame:
    raw_files = sorted(Path(raw_dir).glob("*.csv"))
    if not raw_files:
        raise FileNotFoundError(
            f"Không tìm thấy CSV trong {raw_dir}. Hãy đặt dữ liệu Football-Data.co.uk vào thư mục này."
        )

    all_records: list[dict[str, Any]] = []
    file_reports: list[dict[str, Any]] = []
    for path in raw_files:
        records, stats = _read_one_file(path)
        all_records.extend(records)
        file_reports.append(stats)

    df = pd.DataFrame(all_records)
    if df.empty:
        raise ValueError("Không có trận hợp lệ sau khi đọc dữ liệu nguồn.")

    duplicate_mask = df.duplicated(subset=["Date", "HomeTeam", "AwayTeam"], keep="first")
    duplicate_count = int(duplicate_mask.sum())
    df = df.loc[~duplicate_mask].copy()
    df = df.sort_values(["Date", "Kickoff", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)

    if (df["FTHG"] < 0).any() or (df["FTAG"] < 0).any():
        raise ValueError("Phát hiện số bàn thắng âm.")
    if not df["FTR"].isin(["H", "D", "A"]).all():
        raise ValueError("Nhãn FTR không hợp lệ sau chuẩn hóa.")

    target = Path(output_path) if output_path else PROCESSED_DIR / "premier_league_all_seasons.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    export = df.copy()
    export["Date"] = export["Date"].dt.strftime("%Y-%m-%d")
    export["Kickoff"] = export["Kickoff"].dt.strftime("%Y-%m-%d %H:%M:%S")
    export.to_csv(target, index=False, encoding="utf-8-sig")

    season_counts = df.groupby("Season").size().sort_index().to_dict()
    summary = {
        "source": "Football-Data.co.uk - local season CSV files",
        "raw_directory": (
            Path(raw_dir).resolve().relative_to(PROJECT_ROOT).as_posix()
            if Path(raw_dir).resolve().is_relative_to(PROJECT_ROOT)
            else str(Path(raw_dir))
        ),
        "output_file": (
            target.resolve().relative_to(PROJECT_ROOT).as_posix()
            if target.resolve().is_relative_to(PROJECT_ROOT)
            else str(target)
        ),
        "total_valid_matches": int(len(df)),
        "unique_teams": int(pd.unique(pd.concat([df["HomeTeam"], df["AwayTeam"]])).size),
        "first_date": str(df["Date"].min().date()),
        "last_date": str(df["Date"].max().date()),
        "duplicate_matches_removed": duplicate_count,
        "total_ftr_mismatches": int(sum(x["ftr_mismatches"] for x in file_reports)),
        "season_match_counts": {str(k): int(v) for k, v in season_counts.items()},
        "files": file_reports,
        "notes": [
            "Các trường được ánh xạ theo tên cột, không theo vị trí cột.",
            "Các trường thừa ở cuối dòng được bỏ qua; các cột bắt buộc vẫn được đọc theo header.",
            "Dòng thiếu thông tin kết quả hoặc ngày không hợp lệ được loại khỏi dữ liệu huấn luyện.",
            "Các trận trùng theo Date + HomeTeam + AwayTeam chỉ giữ một bản ghi.",
        ],
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "data_quality_report.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"Đã tạo dữ liệu chuẩn: {target}")
    print(f"Số trận hợp lệ: {len(df):,}")
    print(f"Số đội: {summary['unique_teams']}")
    print(f"Khoảng thời gian: {summary['first_date']} -> {summary['last_date']}")
    print(f"Trận trùng đã loại: {duplicate_count}")
    print(f"Dòng có trường thừa nhưng vẫn đọc được: {sum(x['rows_with_extra_fields'] for x in file_reports)}")
    print(f"Báo cáo chất lượng: {REPORT_DIR / 'data_quality_report.json'}")
    return df


if __name__ == "__main__":
    prepare_data()
