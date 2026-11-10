import numpy as np
import pandas as pd
import pytest

from cta_project.portfolio import (PortfolioEngine,
                                   active_period_statistics, effr_cash_sharpe,
                                   performance, section31_rate)
from cta_project.risk import RiskEngine, choose_control_grid
from cta_project.signals import SignalEngine


def test_fee_free_account_identity(config, market):
    engine = PortfolioEngine(market, config)
    dates = market.calendar[300:310]
    shares = pd.DataFrame(0.0, index=dates, columns=market.symbols)
    shares.loc[dates[1]:, "SPY"] = 2.0
    daily, trades = engine.account_from_shares(shares)
    assert np.allclose(daily["equity"], daily["cash"] + daily["asset_value"])
    assert trades.loc[dates[1], "SPY"] == 2.0 * market.close.loc[dates[1], "SPY"]


def test_equal_weight_buy_and_hold_marks_only_complete_quote_days(config, market):
    engine = PortfolioEngine(market, config)
    missing_date = market.close.index[market.close.isna().any(axis=1)][0]
    position = market.calendar.get_loc(missing_date)
    dates = market.calendar[position - 4:position + 8]
    benchmark = engine.equal_weight_buy_and_hold(
        dates[0], dates[-1] + pd.Timedelta(days=1)
    )
    assert pd.isna(benchmark["equity"].loc[missing_date])
    assert benchmark["statistics"]["缺价日数"] > 0
    entry_price = market.close.loc[benchmark["entry"]]
    assert np.allclose(benchmark["shares"] * entry_price,
                       config.capital / len(market.symbols))
    buy_fee = sum(engine.transaction_cost(
        benchmark["shares"].to_numpy(), entry_price.to_numpy(), benchmark["entry"]
    ))
    assert benchmark["equity"].loc[benchmark["entry"]] == pytest.approx(
        config.capital - buy_fee
    )
    assert benchmark["statistics"]["交易费用美元"] > buy_fee
    exit_price = market.close.loc[benchmark["exit"]]
    sell_fee = sum(engine.transaction_cost(
        -benchmark["shares"].to_numpy(), exit_price.to_numpy(), benchmark["exit"]
    ))
    expected_end = (config.capital
                    + (benchmark["shares"] * (exit_price - entry_price)).sum()
                    - buy_fee - sell_fee)
    assert benchmark["equity"].loc[benchmark["exit"]] == pytest.approx(expected_end)


def test_ibkr_fixed_commission_uses_one_percent_order_cap(config, market):
    engine = PortfolioEngine(market, config)
    quantity = np.zeros(len(market.symbols))
    price = np.full(len(market.symbols), np.nan)
    location = market.symbols.get_loc("SPY")
    quantity[location] = 1.0
    price[location] = 100.0
    commission, section31 = engine.transaction_cost(
        quantity, price, pd.Timestamp("2025-06-02", tz="UTC")
    )
    assert commission == pytest.approx(1.00)
    assert section31 == 0


def test_effr_sharpe_uses_excess_return_volatility(market):
    dates = pd.date_range("2024-06-03", periods=4, freq="D", tz="UTC")
    equity = pd.Series([100.0, 101.0, 100.0, 102.0], index=dates)
    effr = pd.Series([0.0, 18.0, 0.0, 36.0], index=dates)
    returns = equity.pct_change().iloc[1:]
    excess = returns - effr.iloc[:-1].to_numpy() / 36000
    annual_periods = len(returns) / ((dates[-1] - dates[0]).days / 365.25)
    expected = excess.mean() / excess.std(ddof=1) * np.sqrt(annual_periods)
    assert effr_cash_sharpe(equity, effr)["EFFR现金基准Sharpe"] == pytest.approx(expected)


def test_active_period_retains_first_trade_cost_and_excludes_idle_warmup(config, market):
    engine = PortfolioEngine(market, config)
    dates = market.calendar[300:330]
    target = np.zeros(len(market.symbols))
    target[market.symbols.get_loc("SPY")] = 0.5
    account = engine.simulate(pd.DatetimeIndex([dates[10]]), dates[0],
                              dates[-1] + pd.Timedelta(days=1),
                              lambda *_: (target, {}))
    active = active_period_statistics(account, market.effr)
    assert active["起始成交日"] == dates[11]
    assert active["空仓准备交易日数"] == 11
    assert account["daily"].loc[dates[11], "fee"] > 0
    expected = performance(account["daily"].loc[dates[11]:])
    assert active["绩效"]["几何年化收益"] == pytest.approx(expected["几何年化收益"])
    cash = effr_cash_sharpe(account["daily"]["equity"].loc[dates[10]:], market.effr)
    assert active["EFFR现金基准Sharpe"] == pytest.approx(cash["EFFR现金基准Sharpe"])


def test_fees_enter_cash_before_next_sleeve_allocation(config, market):
    engine = PortfolioEngine(market, config)
    dates = market.calendar[300:316]
    decisions = dates[[0, 5]]
    target = np.zeros(len(market.symbols))
    target[market.symbols.get_loc("SPY")] = 0.5
    account = engine.simulate(decisions, dates[0], dates[-1] + pd.Timedelta(days=1),
                              lambda *_: (target, {}))
    daily = account["daily"]
    assert daily["fee"].sum() > 0
    assert np.allclose(daily["equity"], daily["cash"] + daily["asset_value"])
    assert np.isclose(daily["equity"].iloc[-1] - config.capital,
                      account["records"]["pnl_5d"].sum() - daily["fee"].sum())
    assert account["records"].iloc[1]["entry_capital"] < (
        config.capital / config.holding
        + account["records"].iloc[0]["pnl_5d"]
    )
    net_daily, _ = engine.after_fee(account, sum(engine.fee_components(account)))
    assert net_daily["equity"].equals(daily["equity"])


def test_unaffected_etf_can_be_held_during_another_etf_gap(config, market):
    engine = PortfolioEngine(market, config)
    gap_date = market.close.index[market.close.isna().any(axis=1)][0]
    location = market.calendar.get_loc(gap_date)
    dates = market.calendar[location - 3:location + 5]
    target = np.zeros(len(market.symbols))
    target[market.symbols.get_loc("SPY")] = 0.5
    account = engine.simulate(pd.DatetimeIndex([dates[0]]), dates[0],
                              dates[-1] + pd.Timedelta(days=1),
                              lambda *_: (target, {}))
    assert account["shares"].loc[gap_date, "SPY"] > 0
    assert account["shares"].loc[gap_date, "IEF"] == 0
    assert np.isfinite(account["daily"].loc[gap_date, "equity"])


def test_optimizer_uses_available_assets_across_gap(config, market):
    engine = PortfolioEngine(market, config)
    gap_date = market.close.index[market.close.isna().any(axis=1)][0]
    position = market.calendar.get_loc(gap_date) - 3
    signal_date = market.calendar[position]
    end = market.calendar[position + 7] + pd.Timedelta(days=1)
    index = pd.MultiIndex.from_product([[signal_date], market.symbols],
                                       names=["date", "symbol"])
    prediction = pd.Series(np.linspace(-0.01, 0.01, len(market.symbols)), index=index)
    covariance = np.eye(len(market.symbols)) * config.scale**2 * 0.0001
    month = signal_date.normalize().replace(day=1)
    account = engine.optimized_account(prediction, {("Ridge", month): covariance},
                                       signal_date, end, 0.005, 0.0025, "Ridge")
    excluded = market.target.loc[signal_date].index[
        market.target.loc[signal_date].isna()
    ]
    assert account["weights"].loc[signal_date, excluded].eq(0).all()
    assert account["records"].loc[signal_date, "eligible_assets"] < len(market.symbols)
    assert account["records"].loc[signal_date, "eligible_assets"] >= 20
    assert np.isfinite(account["daily"]["equity"]).all()


def test_optimizer_compares_multiple_feasible_starts(config, market):
    count = len(market.symbols)
    mu = np.linspace(-100.0, 100.0, count)
    sigma = np.eye(count) * config.scale**2 * 0.0001
    solution = PortfolioEngine(market, config).optimize(
        mu, sigma, np.zeros(count), 0.005, 0.0025
    )
    weights = solution["weights"]
    assert solution["attempted_starts"] >= 2
    assert solution["converged_starts"] >= 1
    assert solution["constraint_error"] <= config.tolerance
    assert np.isfinite(weights).all()
    assert config.asset_bounds[0] - config.tolerance <= weights.min()
    assert weights.max() <= config.asset_bounds[1] + config.tolerance
    assert config.net_band[0] - config.tolerance <= weights.sum()
    assert weights.sum() <= config.net_band[1] + config.tolerance
    assert config.gross_band[0] - config.tolerance <= np.abs(weights).sum()
    assert np.abs(weights).sum() <= config.gross_band[1] + config.tolerance


@pytest.mark.parametrize("date,expected", [
    ("2020-02-17", 20.70), ("2020-02-18", 22.10),
    ("2021-02-24", 22.10), ("2021-02-25", 5.10),
    ("2022-05-13", 5.10), ("2022-05-14", 22.90),
    ("2023-02-26", 22.90), ("2023-02-27", 8.00),
    ("2024-05-21", 8.00), ("2024-05-22", 27.80),
    ("2025-05-13", 27.80), ("2025-05-14", 0.00),
    ("2026-04-03", 0.00), ("2026-04-04", 20.60),
])
def test_section31_rate_boundaries(date, expected):
    assert section31_rate(pd.Timestamp(date, tz="UTC")) == pytest.approx(expected / 1e6)


def test_risk_scale_only_reduces_exposure(config, market):
    portfolio = PortfolioEngine(market, config)
    risk = RiskEngine(market, SignalEngine(market, config), portfolio, config)
    dates = market.calendar[300:302]
    decisions = pd.DataFrame({"历史估计年化波动率": [0.08, 0.08],
                              "高波动状态": [False, True]}, index=dates)
    scale = risk.exposure_scale(decisions, 0.04, 0.25)
    assert np.allclose(scale.to_numpy(), [0.5, 0.125])
    assert np.allclose(risk.exposure_scale(decisions, None, 1.0), 1.0)


def test_risk_grid_retains_baseline_if_state_candidates_miss_limit(config):
    grid = pd.DataFrame({
        "方案": ["基础组合", "状态控制组合", "状态控制组合"],
        "扣费后几何年化收益": [0.10, 0.20, 0.15],
        "扣费后最大回撤": [-0.10, -0.20, -0.16],
    })
    checked, selected, diagnostic = choose_control_grid(grid, config)
    assert selected["方案"] == "基础组合"
    assert diagnostic["扣费后最大回撤"] == -0.16
    assert not checked.loc[checked["方案"].eq("状态控制组合"),
                           "满足Validation回撤门槛"].any()
