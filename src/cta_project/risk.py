"""在优化器目标权重上施加历史波动率与市场状态执行比例。"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import Config
from .data import Market
from .portfolio import PortfolioEngine, active_period_statistics, effr_cash_sharpe
from .signals import SignalEngine


def choose_control_grid(grid: pd.DataFrame, config: Config):
    grid = grid.copy()
    grid["满足Validation回撤门槛"] = grid["扣费后最大回撤"].ge(
        -config.max_drawdown_magnitude
    )
    qualified = grid.loc[grid["满足Validation回撤门槛"]].sort_values(
        ["扣费后几何年化收益", "扣费后最大回撤"],
        ascending=[False, False], kind="stable"
    )
    if qualified.empty:
        raise ValueError("组合风险控制候选均未满足回撤约束。")
    selected = qualified.iloc[0]
    state_candidates = qualified.loc[qualified["方案"].eq("状态控制组合")]
    if state_candidates.empty:
        state_candidates = grid.loc[grid["方案"].eq("状态控制组合")].sort_values(
            ["扣费后最大回撤", "扣费后几何年化收益"],
            ascending=[False, False], kind="stable"
        )
    if state_candidates.empty:
        raise ValueError("缺少状态控制候选账户。")
    return grid, selected, state_candidates.iloc[0]


class RiskEngine:
    def __init__(self, market: Market, signals: SignalEngine,
                 portfolio: PortfolioEngine, config: Config):
        self.market, self.signals, self.portfolio, self.config = (
            market, signals, portfolio, config
        )

    def prepare_decisions(self, base_account) -> pd.DataFrame:
        config, market = self.config, self.market
        base_weights = base_account["weights"].reindex(columns=market.symbols)
        asset_daily_returns = market.close.pct_change(fill_method=None)
        rows = []
        for date, row in base_weights.iterrows():
            history = asset_daily_returns.loc[:date].tail(config.risk_lookback)
            weights = row.to_numpy(dtype=float)
            active = np.abs(weights) > 1e-10
            valid_history = (len(history) == config.risk_lookback
                             and bool(active.any())
                             and not history.loc[:, active].isna().any().any())
            if valid_history:
                covariance = LedoitWolf().fit(history.loc[:, active].to_numpy()).covariance_
                expected = float(np.sqrt(max(weights[active] @ covariance @ weights[active], 0))
                                 * np.sqrt(252))
            else:
                expected = np.nan
            states = [float(self.signals.vol_percentile[name].loc[date])
                      for name in ("VIX", "MOVE")]
            if not np.isfinite(states).all():
                raise ValueError(f"{date}的风险输入无效。")
            rows.append({"date": date, "历史估计年化波动率": expected,
                         "历史波动率缺失": not valid_history,
                         "VIX历史分位数": states[0], "MOVE历史分位数": states[1],
                         "高波动状态": max(states) >= config.high_state_percentile})
        return pd.DataFrame(rows).set_index("date")

    def exposure_scale(self, decisions: pd.DataFrame,
                       target_volatility: float | None,
                       high_multiplier: float) -> pd.Series:
        if target_volatility is None:
            return pd.Series(1.0, index=decisions.index, name="执行比例")
        volatility = decisions["历史估计年化波动率"].clip(lower=1e-12)
        base = (target_volatility / volatility).clip(upper=1.0).fillna(1.0)
        state = np.where(decisions["高波动状态"], high_multiplier, 1.0)
        scale = base * state
        if not np.isfinite(scale).all() or not np.all((scale > 0) & (scale <= 1)):
            raise ValueError("执行比例不在有效范围内。")
        return pd.Series(scale, index=decisions.index, name="执行比例")

    def controlled_account(self, base_account, decisions,
                           target_volatility: float | None,
                           high_multiplier: float):
        config, market = self.config, self.market
        decision_weights = base_account["weights"].reindex(columns=market.symbols)
        scale = self.exposure_scale(decisions, target_volatility, high_multiplier)
        if (len(decision_weights.index) != len(scale.index)
                or not decision_weights.index.difference(scale.index).empty
                or not scale.index.difference(decision_weights.index).empty):
            raise ValueError("优化器权重与风险决策日期不一致。")
        scale = scale.reindex(decision_weights.index)
        entries = []
        def decide(date, _old_q, _known_nav, _signal_price):
            position = market.calendar.get_loc(date)
            entry = market.calendar[position + config.delay]
            weights = decision_weights.loc[date]
            target_weights = weights.to_numpy(dtype=float) * float(scale.loc[date])
            entries.append({"决策日": date, "执行日": entry,
                            "高波动状态": bool(decisions.loc[date, "高波动状态"]),
                            "历史波动率缺失": bool(decisions.loc[date, "历史波动率缺失"]),
                            "历史估计年化波动率": decisions.loc[date, "历史估计年化波动率"],
                            "执行比例": float(scale.loc[date]),
                            "目标净敞口": target_weights.sum(),
                            "目标总敞口": np.abs(target_weights).sum(),
                            "目标最大权重": target_weights.max(),
                            "目标最小权重": target_weights.min()})
            return target_weights, {}

        dates = base_account["daily"].index
        result = self.portfolio.simulate(decision_weights.index, dates[0],
                                         dates[-1] + pd.Timedelta(days=1), decide)
        result["risk_decisions"] = pd.DataFrame(entries).set_index("决策日")
        net_daily, statistics = self.portfolio.after_fee(
            result, result["commission"] + result["section31"]
        )
        result.update({"net_daily": net_daily, "statistics": statistics})
        risk = result["risk_decisions"]
        if (risk["目标最大权重"].max() > config.asset_bounds[1] + config.tolerance
                or risk["目标最小权重"].min() < config.asset_bounds[0] - config.tolerance
                or risk["目标净敞口"].max() > config.net_band[1] + config.tolerance
                or risk["目标总敞口"].max() > config.gross_band[1] + config.tolerance):
            raise ValueError("风险执行账户越过原仓位上限。")
        return result

    def comparison_row(self, label: str, target_volatility,
                       high_multiplier: float, result) -> dict:
        stats = result["statistics"]
        active = active_period_statistics(result, self.market.effr)
        active_stats = active["绩效"]
        risk = result["risk_decisions"]
        values = result["shares"] * self.market.close.reindex(result["shares"].index)
        realized_net = values.sum(axis=1) / result["net_daily"]["equity"]
        realized_gross = values.abs().sum(axis=1) / result["net_daily"]["equity"]
        return {"方案": label, "目标年化波动率": target_volatility,
                "高波动乘数": high_multiplier,
                "扣费后几何年化收益": float(stats["几何年化收益"]),
                "扣费后最大回撤": float(stats["最大回撤"]),
                "实现年化波动率": float(stats["年化波动"]),
                 "有效交易期起始日": active["起始成交日"],
                 "空仓准备交易日数": active["空仓准备交易日数"],
                 "有效交易期几何年化收益": float(active_stats["几何年化收益"]),
                 "有效交易期最大回撤": float(active_stats["最大回撤"]),
                 "有效交易期年化波动率": float(active_stats["年化波动"]),
                 "有效交易期EFFR现金基准Sharpe": active["EFFR现金基准Sharpe"],
                "EFFR现金基准Sharpe": effr_cash_sharpe(
                    result["net_daily"]["equity"], self.market.effr
                )["EFFR现金基准Sharpe"],
                "平均每日换手": float(stats["平均每日换手"]),
                "平均执行比例": float(risk["执行比例"].mean()),
                "平均目标净敞口": float(risk["目标净敞口"].mean()),
                "平均目标总敞口": float(risk["目标总敞口"].mean()),
                "实际净敞口最小值": float(realized_net.min()),
                "实际净敞口最大值": float(realized_net.max()),
                "实际总敞口最小值": float(realized_gross.min()),
                "实际总敞口最大值": float(realized_gross.max()),
                "高波动决策日数": int(risk["高波动状态"].sum()),
                "历史波动率缺失决策日数": int(risk["历史波动率缺失"].sum()),
                "总费用美元": float((result["commission"] + result["section31"]).sum())}

    def select_control(self, validation_base):
        decisions = self.prepare_decisions(validation_base)
        baseline = self.controlled_account(validation_base, decisions, None, 1.0)
        base_commission, base_section31 = self.portfolio.fee_components(validation_base)
        base_daily, _ = self.portfolio.after_fee(validation_base,
                                                 base_commission + base_section31)
        if not (np.allclose(baseline["daily"]["equity"],
                                validation_base["daily"]["equity"],
                                atol=1e-6, rtol=1e-10)
                and np.allclose(baseline["net_daily"]["equity"],
                                base_daily["equity"], atol=1e-6, rtol=1e-10)):
            raise ValueError("执行比例为１的账户未与原组合逐日一致。")
        rows = [self.comparison_row("基础组合", None, 1.0, baseline)]
        for target in self.config.target_volatilities:
            for multiplier in self.config.high_state_multipliers:
                candidate = self.controlled_account(validation_base, decisions,
                                                    target, multiplier)
                rows.append(self.comparison_row("状态控制组合", target,
                                                multiplier, candidate))
        return choose_control_grid(pd.DataFrame(rows), self.config)
