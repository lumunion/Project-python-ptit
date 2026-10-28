""" 1. Đọc dữ liệu lịch sử các trận bóng đá đã được chuẩn hóa
    2. Theo dõi lịch sử thi đấu của từng đội bóng
    3. Tính toán 15 features cho mỗi trận đấu
    4. Đảm bảo các features chỉ được sử dụng thông tin có trước trận đấu, tránh Data lackage
    5. Cập nhập phong độ, bàn thắng và Elo của các đội sau khi kết quả trận đấu đã được xác định
    6. Xuất dữ liệu hoàn chỉnh để đưa vào các thuật toán Machine Learning
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = PROJECT_ROOT / "Data" / "processed" / "premier_league_all_seasons.csv"
OUTPUT_PATH = PROJECT_ROOT / "Data" / "processed" / "features_data.csv"

""" 5 nhóm: Phong độ, Khả năng ghi bàn/phòng ngự, số trận đã chơi/ngày nghỉ, Elo, khả năng tấn công/phòng ngự"""
FEATURE_COLUMNS = [
    "Home_Form_Pts_5",
    "Away_Form_Pts_5",
    "Home_Goals_For_5",
    "Home_Goals_Against_5",
    "Away_Goals_For_5",
    "Away_Goals_Against_5",
    "Home_Matches_Played",
    "Away_Matches_Played",
    "Home_Rest_Days",
    "Away_Rest_Days",
    "Home_Elo",
    "Away_Elo",
    "Elo_Diff",
    "Attack_Advantage_Home",
    "Attack_Advantage_Away",
]

PRIOR_POINTS = 1.5 # thắng được 3 điểm, hòa 1 điểm, thua 0 điểm
PRIOR_GOALS = 1.35 # theo thống kê hằng năm trung bình mỗi trận đấu có khoảng 2.7 bàn
PRIOR_STRENGTH = 2.0 # Con số 2.0 đóng vai trò như 2 trận đấu "ảo"
INITIAL_ELO = 1500.0 # quy ước toán học tiêu chuẩn của hệ thống Elo toàn cầu
HOME_ADVANTAGE_ELO = 60.0 # Trong bóng đá, đá sân nhà mang lại lợi thế cực kỳ lớn (khán giả, sân bãi, không phải di chuyển).
ELO_K = 20.0 # K-factor là "tốc độ học" (learning rate) của mô hình.

# khởi tạo trạng thái ban đầu của đội bóng
def _empty_state() -> dict[str, Any]:
    return {"history": [], "elo": INITIAL_ELO, "last_date": None}

# tính mean 5 trận gần nhất
def _history_mean(history: list[dict[str, float]], key: str, prior: float) -> float:
    values = [float(item[key]) for item in history[-5:]]
    return (sum(values) + PRIOR_STRENGTH * prior) / (len(values) + PRIOR_STRENGTH)

# tính số ngày nghỉ của độ bóng
def _rest_days(state: dict[str, Any], current_date: pd.Timestamp) -> float:
    previous = state["last_date"]
    if previous is None:
        return 7.0
    days = (current_date.normalize() - previous.normalize()).days
    return float(min(max(days, 0), 30))


# tính 15 feature cho 1 trận đấu
def _fixture_features(home: str, away: str, current_date: pd.Timestamp, states: dict[str, dict[str, Any]]) -> dict[str, float]:
    h = states.setdefault(home, _empty_state())
    a = states.setdefault(away, _empty_state())
    h_hist = h["history"][-5:]
    a_hist = a["history"][-5:]
    h_form = _history_mean(h_hist, "points", PRIOR_POINTS)
    a_form = _history_mean(a_hist, "points", PRIOR_POINTS)
    h_gf = _history_mean(h_hist, "gf", PRIOR_GOALS)
    h_ga = _history_mean(h_hist, "ga", PRIOR_GOALS)
    a_gf = _history_mean(a_hist, "gf", PRIOR_GOALS)
    a_ga = _history_mean(a_hist, "ga", PRIOR_GOALS)
    h_elo = float(h["elo"])
    a_elo = float(a["elo"])
    return {
        "Home_Form_Pts_5": h_form,
        "Away_Form_Pts_5": a_form,
        "Home_Goals_For_5": h_gf,
        "Home_Goals_Against_5": h_ga,
        "Away_Goals_For_5": a_gf,
        "Away_Goals_Against_5": a_ga,
        "Home_Matches_Played": float(len(h["history"])),
        "Away_Matches_Played": float(len(a["history"])),
        "Home_Rest_Days": _rest_days(h, current_date),
        "Away_Rest_Days": _rest_days(a, current_date),
        "Home_Elo": h_elo,
        "Away_Elo": a_elo,
        "Elo_Diff": h_elo + HOME_ADVANTAGE_ELO - a_elo,
        "Attack_Advantage_Home": h_gf - a_ga,
        "Attack_Advantage_Away": a_gf - h_ga,
    }


def _iter_match_batches(df: pd.DataFrame) -> Iterator[pd.DataFrame]:
    """tạo ra yield một lô batch cho mỗi ngày trên lịch thi đấu để tránh data lackage trong cùng một ngày
       không quan tâm kickoff times trong ngày 
       tất cả các trận đấu trong cùng một ngày dều được trích xuất featurized trước khi bất kỳ kết quả
       của ngày đó dùng để cập nhật 
    """
    for _, day in df.groupby("Date", sort=True):
        yield day.sort_values(["Kickoff", "HomeTeam", "AwayTeam"], kind="mergesort")

# cập nhật trạng thái các đội sau khi có kết quả trận đấu
def _advance_states(batch: pd.DataFrame, states: dict[str, dict[str, Any]]) -> None:
    deltas: dict[str, float] = defaultdict(float)
    for row in batch.itertuples(index=False):
        home = str(row.HomeTeam)
        away = str(row.AwayTeam)
        current_date = pd.Timestamp(row.Date).normalize()
        h = states.setdefault(home, _empty_state())
        a = states.setdefault(away, _empty_state())
        h_elo = float(h["elo"])
        a_elo = float(a["elo"])
        expected_home = 1.0 / (1.0 + 10.0 ** (-(h_elo + HOME_ADVANTAGE_ELO - a_elo) / 400.0))
        result_home = 1.0 if row.FTR == "H" else 0.5 if row.FTR == "D" else 0.0
        change = ELO_K * (result_home - expected_home)
        deltas[home] += change
        deltas[away] -= change

        home_points = 3.0 if row.FTR == "H" else 1.0 if row.FTR == "D" else 0.0
        away_points = 3.0 if row.FTR == "A" else 1.0 if row.FTR == "D" else 0.0
        h["history"].append({"points": home_points, "gf": float(row.FTHG), "ga": float(row.FTAG)})
        a["history"].append({"points": away_points, "gf": float(row.FTAG), "ga": float(row.FTHG)})
        h["last_date"] = current_date
        a["last_date"] = current_date
    for team, delta in deltas.items():
        states[team]["elo"] += delta

# đọc dữ liệu từ csv
def load_matches(path: Path = INPUT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Chưa có dữ liệu chuẩn: {path}. Hãy chạy bước prepare_data trước.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Kickoff"] = pd.to_datetime(df["Kickoff"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df.sort_values(["Date", "Kickoff", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    return df

# tạo features cho tất cả các trận đấu để huấn luyện 
def build_features_dataframe(matches: pd.DataFrame) -> pd.DataFrame:
    df = matches.copy().reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    df["Kickoff"] = pd.to_datetime(df["Kickoff"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df = df.sort_values(["Date", "Kickoff", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)

    feature_rows: dict[int, dict[str, float]] = {}
    states: dict[str, dict[str, Any]] = {}
    for batch in _iter_match_batches(df):
        for idx, row in batch.iterrows():
            feature_rows[idx] = _fixture_features(
                str(row["HomeTeam"]), str(row["AwayTeam"]), pd.Timestamp(row["Date"]), states
            )
        _advance_states(batch, states)

    features = pd.DataFrame.from_dict(feature_rows, orient="index").sort_index()
    result = df.copy()
    for column in FEATURE_COLUMNS:
        result[column] = features[column].astype(float)
    if result[FEATURE_COLUMNS].isna().any().any():
        raise ValueError("Feature engineering tạo ra giá trị thiếu.")
    return result

# tạo features 1 trận cụ thể để dự đoán 
def build_fixture_features(
    matches: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: str | date | pd.Timestamp,
) -> pd.DataFrame:
    """Xây dựng một hàng dự báo chỉ sử dụng các trận đấu diễn ra hoàn toàn trước"""
    if home_team == away_team:
        raise ValueError("Đội nhà và đội khách phải khác nhau.")
    target_date = pd.Timestamp(as_of_date).normalize()
    df = matches.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.normalize()
    history = df[df["Date"] < target_date].copy()
    history = history.sort_values(["Date", "Kickoff", "HomeTeam", "AwayTeam"], kind="mergesort")
    states: dict[str, dict[str, Any]] = {}
    for batch in _iter_match_batches(history):
        _advance_states(batch, states)
    row = _fixture_features(home_team, away_team, target_date, states)
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

# main 
def build_features() -> pd.DataFrame:
    matches = load_matches()
    result = build_features_dataframe(matches)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Số trận đầu vào: {len(matches):,}")
    print(f"Số trận sau tạo feature: {len(result):,} (không loại các trận đầu mùa)")
    print(f"Số đặc trưng: {len(FEATURE_COLUMNS)}")
    print(f"Đã lưu: {OUTPUT_PATH}")
    return result


if __name__ == "__main__":
    build_features()
