"""Print a concise quality summary for the canonical match dataset."""
from __future__ import annotations

from src.data.prepare_data import PROCESSED_DIR
import pandas as pd


def main() -> None:
    path = PROCESSED_DIR / "premier_league_all_seasons.csv"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy {path}; hãy chạy python run_project.py trước.")
    df = pd.read_csv(path, encoding="utf-8-sig")
    print("Shape:", df.shape)
    print("Date range:", df["Date"].min(), "->", df["Date"].max())
    print("Missing values:")
    print(df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].isna().sum())
    print("Duplicate fixtures:", df.duplicated(["Date", "HomeTeam", "AwayTeam"]).sum())
    print("FTR distribution:")
    print(df["FTR"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")
    expected = df.apply(lambda r: "H" if r.FTHG > r.FTAG else "A" if r.FTHG < r.FTAG else "D", axis=1)
    print("FTR/score mismatches:", int((expected != df["FTR"]).sum()))
    print("Matches by season:")
    print(df.groupby("Season").size().to_string())


if __name__ == "__main__":
    main()
