"""六特征研究面板与逐月MACD选择。"""

from functools import lru_cache

import numpy as np
import pandas as pd

from .config import Config, FEATURES
from .data import BONDS, Market


SHORT_WINDOWS = (1, 5, 10, 21)
LONG_WINDOWS = (5, 10, 21, 63, 252)
EWMA_ALPHAS = (1 / 2, 1 / 3, 1 / 5, 1 / 8)
MACD_DELAYS = (1, 5)
SMOOTHERS = (1, 5, 7)


def screening_weights(signal: pd.DataFrame) -> pd.DataFrame:
    """构造MACD筛选所用的净敞口约１、总敞口约２的临时组合。"""
    centered = signal.sub(signal.mean(axis=1), axis=0)
    norm = centered.abs().sum(axis=1)
    relative = centered.div(norm.replace(0, np.nan), axis=0).fillna(0)
    count = signal.notna().sum(axis=1).replace(0, np.nan)
    base = signal.notna().div(count, axis=0).fillna(0)
    low, high = np.zeros(len(signal)), np.full(len(signal), 10.0)
    for _ in range(45):
        mid = (low + high) / 2
        gross = (base + relative.mul(mid, axis=0)).abs().sum(axis=1).to_numpy()
        low, high = np.where(gross < 2, mid, low), np.where(gross >= 2, mid, high)
    result = base + relative.mul((low + high) / 2, axis=0)
    result.loc[norm.le(1e-14)] = base.loc[norm.le(1e-14)]
    return result


def historical_percentile(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window + 1).apply(
        lambda x: np.searchsorted(np.sort(x[:-1]), x[-1], side="right") / window,
        raw=True,
    )


class SignalEngine:
    def __init__(self, market: Market, config: Config):
        self.market, self.config = market, config
        daily = market.daily_return
        self.mom = daily.rolling(config.mom_window).mean()
        realized_vol = daily.rolling(config.mom_window).std()
        self.momra = self.mom / realized_vol.replace(0, np.nan)
        self.csmom = self.mom.rank(axis=1, pct=True) - 0.5
        self.vol_percentile = {
            name: historical_percentile(series, config.quantile_window)
            for name, series in market.vol_indices.items()
        }
        self.macd_specs = [
            ("Rolling", short, long, delay)
            for short in SHORT_WINDOWS for long in LONG_WINDOWS
            for delay in MACD_DELAYS if short < long
        ]
        self.macd_specs += [
            ("EWMA", alpha, long, delay)
            for alpha in EWMA_ALPHAS for long in LONG_WINDOWS
            for delay in MACD_DELAYS if 2 / alpha - 1 < long
        ]

    @lru_cache(None)
    def macd_signal(self, method: str, short: float, long: int,
                    smoother: int = 1) -> pd.DataFrame:
        daily = self.market.daily_return
        fast = (daily.rolling(int(short)).mean() if method == "Rolling"
                else daily.ewm(alpha=short, adjust=False, min_periods=long).mean())
        return (fast - daily.rolling(long).mean()).rolling(smoother).mean()

    @lru_cache(None)
    def signal_pnl(self, method: str, short: float, long: int,
                   delay: int, smoother: int = 1) -> pd.Series:
        close = self.market.close
        weights = screening_weights(self.macd_signal(method, short, long, smoother))
        additions = weights.shift(delay).fillna(0) / close / self.config.holding
        holdings = additions.rolling(self.config.holding, min_periods=1).sum()
        pnl = holdings.shift(1).fillna(0) * close.diff()
        complete = close.notna().rolling(
            self.config.holding + delay + 1,
            min_periods=self.config.holding + delay + 1,
        ).sum().eq(self.config.holding + delay + 1)
        return pnl.where(complete).sum(axis=1, min_count=1)

    @lru_cache(None)
    def select_macd(self, start: pd.Timestamp, end: pd.Timestamp,
                    execution_delay: int | None = None):
        raw_rows, smooth_rows = [], []
        for spec in self.macd_specs:
            if execution_delay is not None and spec[3] != execution_delay:
                continue
            raw = self.signal_pnl(*spec).loc[lambda x: (x.index >= start) & (x.index < end)]
            raw_rows.append({"spec": spec, "mean": raw.mean(), "std": raw.std()})
            effective = spec[1] if spec[0] == "Rolling" else 2 / spec[1] - 1
            for smoother in SMOOTHERS:
                if smoother <= effective:
                    values = self.signal_pnl(*spec, smoother).loc[
                        lambda x: (x.index >= start) & (x.index < end)
                    ]
                    smooth_rows.append({"spec": (*spec, smoother),
                                        "mean": values.mean(), "std": values.std()})
        raw_grid = pd.DataFrame(raw_rows).sort_values("mean", ascending=False, kind="stable")
        smooth_grid = pd.DataFrame(smooth_rows).sort_values("mean", ascending=False, kind="stable")
        return smooth_grid.iloc[0]["spec"], raw_grid, smooth_grid

    @lru_cache(None)
    def feature_panel(self, spec: tuple) -> pd.DataFrame:
        method, short, long, _delay, smoother = spec
        state = pd.DataFrame({symbol: self.vol_percentile["MOVE" if symbol in BONDS else "VIX"]
                              for symbol in self.market.symbols})
        signals = {"MOM": self.mom, "MOMRA": self.momra,
                   "CSMOM": self.csmom,
                   "MACD": self.macd_signal(method, short, long),
                   "MACD_SMOOTH": self.macd_signal(method, short, long, smoother),
                   "VOL_STATE": state}
        normalized = {}
        for name, frame in signals.items():
            history = frame.shift(1).rolling(self.config.winsor_window)
            clipped = frame.clip(history.quantile(self.config.winsor_limits[0]),
                                 history.quantile(self.config.winsor_limits[1]))
            if name == "VOL_STATE" and self.config.vol_state_variant == "level":
                normalized[name] = clipped.stack()
            else:
                denominator = clipped.abs().sum(axis=1).replace(0, np.nan)
                normalized[name] = clipped.div(denominator, axis=0).stack()
        panel = pd.concat(normalized, axis=1)
        panel["target"] = self.market.target.stack()
        panel.index.names = ["date", "symbol"]
        panel["label_end"] = self.market.label_end.reindex(
            panel.index.get_level_values("date")
        ).to_numpy()
        return panel.sort_index()


def period(panel: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp,
           maturity: pd.Timestamp | None = None) -> pd.DataFrame:
    date = panel.index.get_level_values("date")
    keep = (date >= start) & (date < end)
    if maturity is not None:
        keep &= panel["label_end"].lt(maturity)
    required = list(FEATURES) + (["target"] if maturity is not None else [])
    return panel.loc[keep].dropna(subset=required)
