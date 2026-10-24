import csv

import pandas as pd

from src.data.prepare_data import _read_one_file
from src.features.build_features import FEATURE_COLUMNS, build_features_dataframe, build_fixture_features


def _matches():
    rows = [
        # Two matches on the same date: neither can use the other's result.
        {"Date": "2020-01-01", "Time": None, "Kickoff": "2020-01-01 00:00:00", "Season": "2019-2020", "HomeTeam": "A", "AwayTeam": "B", "FTHG": 3, "FTAG": 0, "FTR": "H", "Source_File": "x.csv"},
        {"Date": "2020-01-01", "Time": None, "Kickoff": "2020-01-01 00:00:00", "Season": "2019-2020", "HomeTeam": "C", "AwayTeam": "A", "FTHG": 0, "FTAG": 2, "FTR": "A", "Source_File": "x.csv"},
        {"Date": "2020-01-08", "Time": None, "Kickoff": "2020-01-08 00:00:00", "Season": "2019-2020", "HomeTeam": "A", "AwayTeam": "C", "FTHG": 1, "FTAG": 1, "FTR": "D", "Source_File": "x.csv"},
    ]
    return pd.DataFrame(rows)


def test_feature_columns_are_pre_match_only():
    assert "FTR" not in FEATURE_COLUMNS
    assert "FTHG" not in FEATURE_COLUMNS
    assert "FTAG" not in FEATURE_COLUMNS


def test_same_day_results_do_not_change_same_day_features():
    data = _matches()
    base = build_features_dataframe(data)
    changed = data.copy()
    changed.loc[0, "FTHG"] = 0
    changed.loc[0, "FTAG"] = 4
    changed.loc[0, "FTR"] = "A"
    altered = build_features_dataframe(changed)
    # The second fixture is away-team A. Its features must not see the first fixture's result.
    pd.testing.assert_series_equal(
        base.loc[1, FEATURE_COLUMNS], altered.loc[1, FEATURE_COLUMNS], check_names=False
    )


def test_upcoming_fixture_uses_only_earlier_dates():
    data = _matches()
    feature = build_fixture_features(data, "A", "B", "2020-01-08")
    assert len(feature) == 1
    assert feature.loc[0, "Home_Matches_Played"] == 2
    # Both matches on 2020-01-01 are historical; the 2020-01-08 fixture itself is excluded.
    assert feature.loc[0, "Home_Form_Pts_5"] >= 0


def test_csv_reader_maps_by_header_and_keeps_rows_with_extra_tail_fields(tmp_path):
    path = tmp_path / "2019-2020.csv"
    with path.open("w", encoding="latin1", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Div", "Date", "Time", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        writer.writerow(["E0", "09/08/2019", "20:00", "Liverpool", "Norwich", 4, 1, "H", "extra", "tail"])
    records, stats = _read_one_file(path)
    assert len(records) == 1
    assert records[0]["HomeTeam"] == "Liverpool"
    assert records[0]["AwayTeam"] == "Norwich"
    assert records[0]["FTHG"] == 4
    assert records[0]["Time"] == "20:00:00"
    assert stats["rows_with_extra_fields"] == 1


def test_same_date_is_a_single_batch_even_when_kickoff_times_exist():
    data = _matches()
    data.loc[0, "Time"] = "12:30:00"
    data.loc[1, "Time"] = "19:00:00"
    base = build_features_dataframe(data)
    changed = data.copy()
    changed.loc[0, "FTHG"] = 0
    changed.loc[0, "FTAG"] = 5
    changed.loc[0, "FTR"] = "A"
    altered = build_features_dataframe(changed)
    pd.testing.assert_series_equal(
        base.loc[1, FEATURE_COLUMNS], altered.loc[1, FEATURE_COLUMNS], check_names=False
    )
