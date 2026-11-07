
"""Unit tests for market dataset validation and cleaning."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from quantfin.data import cleaning


def make_ohlcv_data(
    *,
    tickers: tuple[str, ...] = ("AAPL",),
    dates: pd.DatetimeIndex | None = None,
) -> pd.DataFrame:
    """Build a small valid multi-ticker OHLCV dataset."""

    if dates is None:
        dates = pd.date_range("2025-01-02", periods=3, freq="B")

    fields = ["Open", "High", "Low", "Close", "Volume"]
    columns = pd.MultiIndex.from_product(
        [tickers, fields],
        names=["Ticker", "Field"],
    )

    values = []

    for _ in dates:
        for _ticker in tickers:
            values.append([100.0, 105.0, 95.0, 102.0, 1000.0])

    return pd.DataFrame(
        np.asarray(values).reshape(len(dates), len(columns)),
        index=dates,
        columns=columns,
    )


def validate(
    data: pd.DataFrame,
    *,
    dataset_name: str = "test_dataset",
    allow_complete_missing_rows: bool = False,
) -> tuple[pd.DataFrame, dict]:
    """Call the production validator with standard test arguments."""

    return cleaning.validate_dataset(
        data=data,
        dataset_name=dataset_name,
        allow_complete_missing_rows=allow_complete_missing_rows,
    )


def test_validate_dataset_accepts_valid_data() -> None:
    data = make_ohlcv_data()

    clean_data, report = validate(data)

    pd.testing.assert_frame_equal(clean_data, data)
    assert report["dataset"] == "test_dataset"
    assert report["rows"] == 3
    assert report["columns"] == 5
    assert report["duplicate_dates"] == 0
    assert report["total_missing_cells"] == 0
    assert report["status"] == "PASS"
    assert report["tickers"]["AAPL"]["status"] == "PASS"
    assert report["tickers"]["AAPL"]["valid_rows"] == 3


def test_validate_dataset_accepts_multiple_tickers() -> None:
    data = make_ohlcv_data(tickers=("AAPL", "MSFT", "NVDA"))

    clean_data, report = validate(data)

    assert list(clean_data.columns.get_level_values(0).unique()) == [
        "AAPL",
        "MSFT",
        "NVDA",
    ]
    assert list(report["tickers"]) == ["AAPL", "MSFT", "NVDA"]


def test_validate_dataset_sorts_dates() -> None:
    data = make_ohlcv_data()
    data = data.iloc[[2, 0, 1]]

    clean_data, report = validate(data)

    assert clean_data.index.is_monotonic_increasing
    assert clean_data.index[0] == pd.Timestamp("2025-01-02")
    assert report["first_date"] == str(clean_data.index.min())
    assert report["last_date"] == str(clean_data.index.max())


def test_validate_dataset_rejects_empty_data() -> None:
    data = make_ohlcv_data().iloc[0:0]

    with pytest.raises(ValueError, match="dataset is empty"):
        validate(data)


def test_validate_dataset_rejects_non_multiindex_columns() -> None:
    data = make_ohlcv_data()
    data.columns = ["Open", "High", "Low", "Close", "Volume"]

    with pytest.raises(ValueError, match="expected MultiIndex columns"):
        validate(data)


def test_validate_dataset_rejects_non_datetime_index() -> None:
    data = make_ohlcv_data()
    data.index = ["day1", "day2", "day3"]

    with pytest.raises(ValueError, match="expected a DatetimeIndex"):
        validate(data)


def test_validate_dataset_rejects_duplicate_dates() -> None:
    data = make_ohlcv_data()
    data.index = pd.DatetimeIndex(
        ["2025-01-02", "2025-01-03", "2025-01-02"]
    )

    with pytest.raises(ValueError, match="duplicate dates detected"):
        validate(data)


@pytest.mark.parametrize(
    "missing_field",
    ["Open", "High", "Low", "Close", "Volume"],
)
def test_validate_dataset_rejects_missing_required_field(
    missing_field: str,
) -> None:
    data = make_ohlcv_data()
    data = data.drop(columns=[("AAPL", missing_field)])

    with pytest.raises(ValueError, match="missing required fields"):
        validate(data)


def test_validate_dataset_rejects_partially_missing_ohlcv_row() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", "Close")] = np.nan

    with pytest.raises(
        ValueError,
        match="partially missing OHLCV rows detected",
    ):
        validate(data, allow_complete_missing_rows=True)


def test_validate_dataset_rejects_complete_missing_row_by_default() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], "AAPL"] = np.nan

    with pytest.raises(
        ValueError,
        match="missing OHLCV observations detected",
    ):
        validate(data)


def test_validate_dataset_allows_complete_missing_rows_when_enabled() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], "AAPL"] = np.nan

    clean_data, report = validate(
        data,
        allow_complete_missing_rows=True,
    )

    assert clean_data.isna().sum().sum() == 5

    ticker_report = report["tickers"]["AAPL"]

    assert ticker_report["missing_cells"] == 5
    assert ticker_report["missing_rows"] == 1
    assert ticker_report["valid_rows"] == 2
    assert ticker_report["status"] == "PASS"
    assert report["total_missing_cells"] == 5


def test_validate_dataset_reports_missing_values_by_field() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], "AAPL"] = np.nan

    _, report = validate(
        data,
        allow_complete_missing_rows=True,
    )

    missing_by_field = report["tickers"]["AAPL"]["missing_by_field"]

    assert missing_by_field == {
        "Open": 1,
        "High": 1,
        "Low": 1,
        "Close": 1,
        "Volume": 1,
    }


def test_validate_dataset_rejects_non_numeric_ohlcv_values() -> None:
    data = make_ohlcv_data().astype(object)
    data.loc[data.index[0], ("AAPL", "Close")] = "invalid"

    with pytest.raises(ValueError, match="non-numeric OHLCV values"):
        validate(data)


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_validate_dataset_rejects_infinite_values(
    invalid_value: float,
) -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", "Volume")] = invalid_value

    with pytest.raises(ValueError, match="infinite values detected"):
        validate(data)


@pytest.mark.parametrize(
    ("field", "invalid_price"),
    [
        ("Open", 0.0),
        ("High", -1.0),
        ("Low", 0.0),
        ("Close", -5.0),
    ],
)
def test_validate_dataset_rejects_non_positive_prices(
    field: str,
    invalid_price: float,
) -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", field)] = invalid_price

    with pytest.raises(ValueError, match="non-positive OHLC prices"):
        validate(data)


def test_validate_dataset_rejects_inconsistent_high_price() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", "High")] = 101.0

    with pytest.raises(ValueError, match="inconsistent OHLC values"):
        validate(data)


def test_validate_dataset_rejects_inconsistent_low_price() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", "Low")] = 103.0

    with pytest.raises(ValueError, match="inconsistent OHLC values"):
        validate(data)


def test_validate_dataset_accepts_negligible_ohlc_precision_difference() -> None:
    data = make_ohlcv_data()

    data.loc[data.index[0], ("AAPL", "High")] = 102.0 - 5e-11
    data.loc[data.index[0], ("AAPL", "Low")] = 100.0 + 5e-11

    _, report = validate(data)

    assert report["status"] == "PASS"


def test_validate_dataset_rejects_negative_volume() -> None:
    data = make_ohlcv_data()
    data.loc[data.index[0], ("AAPL", "Volume")] = -1.0

    with pytest.raises(ValueError, match="negative volume detected"):
        validate(data)


def test_validate_dataset_does_not_modify_original_data() -> None:
    data = make_ohlcv_data()
    original = data.copy(deep=True)

    validate(data)

    pd.testing.assert_frame_equal(data, original)


def test_save_clean_dataset_writes_parquet_and_csv(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = make_ohlcv_data()

    monkeypatch.setattr(cleaning, "PROCESSED_DIR", tmp_path)

    cleaning.save_clean_dataset(data, "test_equities")

    parquet_path = tmp_path / "test_equities_clean.parquet"
    csv_path = tmp_path / "test_equities_clean.csv"

    assert parquet_path.is_file()
    assert csv_path.is_file()

    restored_parquet = pd.read_parquet(parquet_path)

    # Parquet preserves the index values but may not preserve its frequency.
    pd.testing.assert_frame_equal(
        restored_parquet,
        data,
        check_freq=False,
    )

    restored_csv = pd.read_csv(
        csv_path,
        header=[0, 1],
        index_col=0,
        parse_dates=True,
    )

    # CSV may represent the date index differently, so compare its values
    # and column labels after normalizing the index.
    restored_csv.index = pd.DatetimeIndex(restored_csv.index)
    restored_csv.columns = data.columns

    pd.testing.assert_frame_equal(
        restored_csv,
        data,
        check_freq=False,
        check_names=False,
    )


def test_save_clean_dataset_creates_output_directory(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_dir = tmp_path / "nested" / "processed"

    monkeypatch.setattr(cleaning, "PROCESSED_DIR", output_dir)

    cleaning.save_clean_dataset(
        make_ohlcv_data(),
        "test_equities",
    )

    assert output_dir.is_dir()
    assert (output_dir / "test_equities_clean.parquet").exists()
    assert (output_dir / "test_equities_clean.csv").exists()


def test_main_raises_when_raw_dataset_is_missing(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    monkeypatch.setattr(cleaning, "RAW_DIR", raw_dir)

    with pytest.raises(FileNotFoundError, match="Dataset not found"):
        cleaning.main()


def test_main_validates_and_generates_quality_reports(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "reports"
    processed_dir = tmp_path / "processed"

    raw_dir.mkdir()

    for settings in cleaning.DATASETS.values():
        (raw_dir / settings["file"]).touch()

    sample_data = make_ohlcv_data()

    monkeypatch.setattr(cleaning, "RAW_DIR", raw_dir)
    monkeypatch.setattr(cleaning, "REPORT_DIR", report_dir)
    monkeypatch.setattr(cleaning, "PROCESSED_DIR", processed_dir)

    monkeypatch.setattr(
        cleaning.pd,
        "read_parquet",
        lambda path: sample_data.copy(),
    )

    saved_datasets = []

    def fake_save_clean_dataset(
        data: pd.DataFrame,
        name: str,
    ) -> None:
        saved_datasets.append(name)

    monkeypatch.setattr(
        cleaning,
        "save_clean_dataset",
        fake_save_clean_dataset,
    )

    cleaning.main()

    assert saved_datasets == list(cleaning.DATASETS)

    report_path = report_dir / "data_quality_report.json"
    summary_path = report_dir / "data_quality_summary.csv"

    assert report_path.exists()
    assert summary_path.exists()

    reports = json.loads(report_path.read_text(encoding="utf-8"))
    summary = pd.read_csv(summary_path)

    assert len(reports) == 3
    assert set(summary["dataset"]) == set(cleaning.DATASETS)
    assert (summary["status"] == "PASS").all()
    assert (summary["rows"] == 3).all()

    output = capsys.readouterr().out

    assert "DATA VALIDATION SUMMARY" in output
    assert "All datasets passed validation." in output
    assert "Raw files were not modified." in output


def test_main_stops_when_a_dataset_fails_validation(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    for settings in cleaning.DATASETS.values():
        (raw_dir / settings["file"]).touch()

    invalid_data = make_ohlcv_data()
    invalid_data.loc[invalid_data.index[0], ("AAPL", "Close")] = np.nan

    monkeypatch.setattr(cleaning, "RAW_DIR", raw_dir)

    monkeypatch.setattr(
        cleaning.pd,
        "read_parquet",
        lambda path: invalid_data.copy(),
    )

    with pytest.raises(ValueError, match="partially missing OHLCV rows"):
        cleaning.main()
