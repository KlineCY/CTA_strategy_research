"""原始行情、市场状态和预测标签。"""

from dataclasses import dataclass
import hashlib
import json

import numpy as np
import pandas as pd

from .config import Config


SPLIT_SYMBOLS = frozenset({"XLB", "XLE", "XLK", "XLU", "XLY"})
SPLIT_DATE = pd.Timestamp("2025-12-05", tz="UTC")
EFFR_SHA256 = "8df2e9a9f6a08fa5122f22e5632aaa0c3d2d0065ec28cb605c19d14def21ae22"
BONDS = frozenset({"AGG", "HYG", "IEF", "LQD", "SHY", "TIP", "TLT"})


@dataclass
class Market:
    prices: dict[str, pd.DataFrame]
    close: pd.DataFrame
    daily_return: pd.DataFrame
    target: pd.DataFrame
    label_end: pd.Series
    manifest: pd.DataFrame
    quality: pd.DataFrame
    missing_counts: pd.Series
    vol_indices: dict[str, pd.Series]
    effr: pd.Series

    @property
    def calendar(self) -> pd.DatetimeIndex:
        return self.close.index

    @property
    def symbols(self) -> pd.Index:
        return self.close.columns


def _load_vol_index(path, calendar: pd.DatetimeIndex) -> pd.Series:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.str.lower()
    date_column = "ts_event" if "ts_event" in frame else "date"
    dates = pd.to_datetime(frame[date_column], utc=True).dt.normalize()
    series = pd.Series(frame["close"].to_numpy(), index=dates).sort_index()
    if series.index.duplicated().any():
        raise ValueError(f"{path.name}存在重复日期。")
    return series.reindex(series.index.union(calendar)).ffill().reindex(calendar)


def _load_effr(path) -> pd.Series:
    if hashlib.sha256(path.read_bytes()).hexdigest() != EFFR_SHA256:
        raise ValueError("EFFR输入文件与项目登记的SHA256不一致。")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    frame = pd.DataFrame(payload["refRates"])[["effectiveDate", "percentRate"]]
    frame["date"] = pd.to_datetime(frame["effectiveDate"], utc=True)
    series = frame.set_index("date")["percentRate"].astype(float).sort_index()
    if not series.index.is_unique:
        raise ValueError("EFFR存在重复日期。")
    days = pd.date_range(series.index.min(), series.index.max(), freq="D", tz="UTC")
    return series.reindex(days).ffill().bfill()


def load_market(config: Config) -> Market:
    files = sorted(config.data_dir.glob("*_ohlcv_1d.csv"))
    if len(files) != 37:
        raise ValueError(f"预期37只ETF，实际找到{len(files)}只。")
    raw, quality = {}, []
    for path in files:
        symbol = path.name.removesuffix("_ohlcv_1d.csv")
        frame = pd.read_csv(path)
        frame["ts_event"] = pd.to_datetime(frame["ts_event"], utc=True)
        if frame["ts_event"].duplicated().any():
            raise ValueError(f"{symbol}存在重复日期。")
        frame = frame.set_index("ts_event").sort_index()
        if not frame[["open", "high", "low", "close"]].gt(0).all().all():
            raise ValueError(f"{symbol}存在非正OHLC价格。")
        if symbol in SPLIT_SYMBOLS:
            frame.loc[frame.index < SPLIT_DATE, ["open", "high", "low", "close"]] *= 0.5
        raw[symbol] = frame
        quality.append({"资产": symbol, "行数": len(frame),
                        "首日": frame.index.min().date(), "末日": frame.index.max().date()})
    close_all = pd.concat({s: frame["close"] for s, frame in raw.items()}, axis=1)
    calendar = close_all.index
    prices = {field: pd.concat({s: frame[field] for s, frame in raw.items()}, axis=1).reindex(calendar)
              for field in ("open", "high", "low", "close")}
    close = prices["close"]
    manifest = pd.read_csv(config.data_dir / "_manifest.csv").set_index("symbol").reindex(close.columns)
    if manifest["category"].isna().any():
        raise ValueError("资产清单缺少类别。")
    daily = close.pct_change(fill_method=None)
    target = close.shift(-(config.delay + config.holding)) / close.shift(-config.delay) - 1
    label_end = pd.Series(calendar, index=calendar).shift(-(config.delay + config.holding))
    required = close.notna()
    complete_holding = required.rolling(config.delay + config.holding + 1,
                                        min_periods=config.delay + config.holding + 1).sum()
    complete_holding = complete_holding.shift(-(config.delay + config.holding)).eq(
        config.delay + config.holding + 1
    )
    target = target.where(complete_holding, np.nan)
    volatility = {name: _load_vol_index(config.data_dir / f"{name}_index_1d_2020_now.csv", calendar)
                  for name in ("VIX", "MOVE")}
    effr = _load_effr(config.data_dir / "EFFR_2020_2026.json")
    return Market(prices, close, daily, target, label_end, manifest,
                  pd.DataFrame(quality), close_all.isna().sum(), volatility, effr)
