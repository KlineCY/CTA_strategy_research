import hashlib

import numpy as np
import pandas as pd
import pytest

from cta_project.config import FEATURES
from cta_project.data import EFFR_SHA256
from cta_project.signals import SignalEngine, period


def test_complete_inputs_and_adjustment(config, market):
    assert len(market.symbols) == 37
    assert len(market.calendar) == 1630
    assert market.calendar.is_unique
    assert market.close.isna().any(axis=1).sum() == 12
    assert market.manifest["category"].notna().all()
    assert hashlib.sha256((config.data_dir / "EFFR_2020_2026.json").read_bytes()).hexdigest() == EFFR_SHA256


def test_target_uses_next_close_to_sixth_close(config, market):
    date = market.calendar[300]
    symbol = "SPY"
    expected = (market.close.loc[market.calendar[306], symbol]
                / market.close.loc[market.calendar[301], symbol] - 1)
    assert market.target.loc[date, symbol] == pytest.approx(expected)
    assert market.label_end.loc[date] == market.calendar[306]


def test_gap_does_not_shortcut_five_market_sessions(market):
    gap = market.close.index[market.close.isna().any(axis=1)]
    affected = market.calendar.get_loc(gap[0]) - 5
    signal_date = market.calendar[affected]
    assert market.label_end.loc[signal_date] == market.calendar[affected + 6]
    missing_symbols = market.close.loc[gap[0]].index[market.close.loc[gap[0]].isna()]
    assert market.target.loc[signal_date, missing_symbols].isna().all()
    assert market.target.loc[signal_date, "SPY"] == pytest.approx(
        market.close.loc[market.calendar[affected + 6], "SPY"]
        / market.close.loc[market.calendar[affected + 1], "SPY"] - 1
    )


def test_train_labels_mature_before_application(config, market):
    signals = SignalEngine(market, config)
    month = pd.Timestamp("2024-06-01", tz="UTC")
    spec = ("EWMA", 0.2, 10, 1, 7)
    panel = signals.feature_panel(spec)
    available = period(panel, month - pd.DateOffset(months=4), month,
                       maturity=month)
    apply = period(panel, month, month + pd.DateOffset(months=1))
    assert available["label_end"].max() < apply.index.get_level_values("date").min()
    assert not available[list(FEATURES)].isna().any().any()


def test_historical_percentile_excludes_current_value():
    from cta_project.signals import historical_percentile
    index = pd.date_range("2020-01-01", periods=4, tz="UTC")
    series = pd.Series([1.0, 2.0, 3.0, 4.0], index=index)
    percentile = historical_percentile(series, window=3)
    assert np.isnan(percentile.iloc[2])
    assert percentile.iloc[3] == 1.0
