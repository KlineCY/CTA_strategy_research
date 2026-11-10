"""带符号权重优化、五批滚动账户、成交费与绩效口径。"""

import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize

from .config import Config
from .data import Market, SPLIT_DATE, SPLIT_SYMBOLS


SMOOTH_EPS = 1e-9
SECTION31_RATES = tuple((pd.Timestamp(date, tz="UTC"), rate) for date, rate in (
    ("2020-01-01", 20.70),
    ("2020-02-18", 22.10),
    ("2021-02-25", 5.10),
    ("2022-05-14", 22.90),
    ("2023-02-27", 8.00),
    ("2024-05-22", 27.80),
    ("2025-05-14", 0.00),
    ("2026-04-04", 20.60),
))


def section31_rate(date: pd.Timestamp) -> float:
    for start, rate in reversed(SECTION31_RATES):
        if date >= start:
            return rate / 1e6
    raise ValueError("Section 31费率表未覆盖成交日期。")


def performance(daily: pd.DataFrame) -> pd.Series:
    returns = daily["return"].dropna()
    wealth = (1 + returns).cumprod()
    high = wealth.cummax().clip(lower=1)
    years = max((returns.index[-1] - returns.index[0]).days + 1, 1) / 365.25
    return pd.Series({"累计收益": wealth.iloc[-1] - 1,
                      "几何年化收益": wealth.iloc[-1] ** (1 / years) - 1,
                      "年化波动": returns.std() * np.sqrt(252),
                      "最大回撤": (wealth / high - 1).min(),
                      "平均每日换手": daily["turnover"].mean(),
                      "交易日数": len(returns)})


def effr_cash_sharpe(equity: pd.Series, effr: pd.Series) -> dict:
    equity = equity.sort_index()
    dates = equity.index
    if len(dates) <= 2 or not dates.is_unique:
        raise ValueError("Sharpe账户日期不足或重复。")
    returns = equity.div(equity.shift()).sub(1).iloc[1:]
    cash_returns = []
    for previous, current in zip(dates[:-1], dates[1:], strict=True):
        days = pd.date_range(previous.normalize(), current.normalize(),
                             freq="D", inclusive="left", tz="UTC")
        rate = effr.reindex(days)
        if rate.isna().any():
            raise ValueError("EFFR未覆盖账户日期。")
        cash_returns.append(float((1 + rate.div(100 * 360)).prod() - 1))
    cash_returns = pd.Series(cash_returns, index=returns.index)
    excess = returns - cash_returns
    years = (dates[-1] - dates[0]).days / 365.25
    periods_per_year = len(returns) / years
    return {"EFFR现金基准Sharpe": float(excess.mean() / excess.std(ddof=1)
                                    * np.sqrt(periods_per_year)),
            "年化波动率": float(returns.std(ddof=1) * np.sqrt(periods_per_year)),
            "EFFR现金累计收益": float((1 + cash_returns).prod() - 1)}


def active_period_statistics(account: dict, effr: pd.Series) -> dict:
    """从首笔实际成交日起计算绩效，并保留该日的费用与收益。"""
    daily = account["daily"]
    first_entry = account["records"]["entry"].min()
    first_position = daily.index.get_loc(first_entry)
    if first_position < 1:
        raise ValueError("首笔成交日前缺少用于计算收益的账户净值。")
    active_daily = daily.iloc[first_position:]
    cash = effr_cash_sharpe(daily["equity"].iloc[first_position - 1:], effr)
    return {"起始成交日": first_entry,
            "空仓准备交易日数": first_position,
            "绩效": performance(active_daily),
            "EFFR现金基准Sharpe": cash["EFFR现金基准Sharpe"]}


class PortfolioEngine:
    def __init__(self, market: Market, config: Config):
        self.market, self.config = market, config

    def equal_weight_buy_and_hold(self, start, end):
        """完整报价日估值的３７只ETF等权买入并持有账户。"""
        dates = self.market.calendar[(self.market.calendar >= start)
                                     & (self.market.calendar < end)]
        prices = self.market.close.reindex(dates)
        complete = prices.notna().all(axis=1)
        quoted = prices.loc[complete]
        if len(quoted) < 3:
            raise ValueError("等权买入并持有账户的完整报价日不足。")
        entry, exit_date = quoted.index[0], quoted.index[-1]
        entry_price = quoted.iloc[0].to_numpy(dtype=float)
        exit_price = quoted.iloc[-1].to_numpy(dtype=float)
        shares = self.config.capital / len(self.market.symbols) / entry_price
        buy_commission, buy_section31 = self.transaction_cost(shares, entry_price, entry)
        sell_commission, sell_section31 = self.transaction_cost(
            -shares, exit_price, exit_date
        )
        purchase_fee = buy_commission + buy_section31
        liquidation_fee = sell_commission + sell_section31
        observed_equity = (self.config.capital
                           + (quoted - quoted.iloc[0]).mul(shares).sum(axis=1)
                           - purchase_fee)
        observed_equity.iloc[-1] -= liquidation_fee
        equity = observed_equity.reindex(dates)
        years = max((exit_date - entry).days + 1, 1) / 365.25
        wealth = observed_equity / self.config.capital
        high = wealth.cummax().clip(lower=1)
        statistics = pd.Series({
            "累计收益": wealth.iloc[-1] - 1,
            "几何年化收益": wealth.iloc[-1] ** (1 / years) - 1,
            "最大回撤": (wealth / high - 1).min(),
            "交易费用美元": purchase_fee + liquidation_fee,
            "完整报价日数": int(complete.sum()),
            "缺价日数": int((~complete).sum()),
        })
        return {"equity": equity, "statistics": statistics,
                "entry": entry, "exit": exit_date,
                "shares": pd.Series(shares, index=self.market.symbols)}

    def signal_dates(self, start, end):
        calendar = self.market.calendar
        complete = self.market.target.notna().any(axis=1)
        return calendar[(calendar >= start) & (self.market.label_end < end)
                        & complete]

    def account_from_shares(self, shares: pd.DataFrame, cost_bps: float = 0,
                            fees: pd.Series | None = None):
        close, config = self.market.close, self.config
        price = close.reindex(shares.index)
        traded = shares.diff().fillna(shares.iloc[0])
        if ((traded.abs() > 1e-10) & price.isna()).any().any():
            raise ValueError("成交资产的执行日收盘价缺失。")
        if ((shares.abs() > 1e-10) & price.isna()).any().any():
            raise ValueError("持仓资产的估值日收盘价缺失。")
        traded_value = traded.mul(price)
        fee = (traded_value.abs().sum(axis=1) * cost_bps / config.scale
               if fees is None else fees.reindex(shares.index))
        if fee.isna().any() or fee.lt(0).any():
            raise ValueError("交易费用缺失或为负。")
        cash = config.capital - traded_value.sum(axis=1).cumsum() - fee.cumsum()
        value = shares.mul(price)
        equity = cash + value.sum(axis=1)
        previous_equity = equity.shift(1).fillna(config.capital)
        daily = pd.DataFrame({"cash": cash, "asset_value": value.sum(axis=1),
                              "equity": equity, "return": equity / previous_equity - 1,
                              "turnover": traded_value.abs().sum(axis=1) / previous_equity,
                              "fee": fee, "net": value.sum(axis=1) / equity,
                              "gross": value.abs().sum(axis=1) / equity})
        if not equity.gt(0).all() or not np.allclose(daily["equity"],
                                                     daily["cash"] + daily["asset_value"]):
            raise ValueError("账户净值未通过现金与持仓核对。")
        return daily, traded_value

    def transaction_cost(self, quantity: np.ndarray, price: np.ndarray,
                         date: pd.Timestamp) -> tuple[float, float]:
        quantity = np.asarray(quantity, dtype=float)
        price = np.asarray(price, dtype=float)
        if not np.isfinite(price[np.abs(quantity) > 1e-10]).all():
            raise ValueError("成交价格缺失。")
        notional = np.abs(quantity * np.nan_to_num(price, nan=0.0))
        actual_quantity = np.abs(quantity).copy()
        if date < SPLIT_DATE:
            for index, symbol in enumerate(self.market.symbols):
                if symbol in SPLIT_SYMBOLS:
                    actual_quantity[index] *= 0.5
        commissions = np.where(notional > 1e-10,
                               np.minimum(np.maximum(actual_quantity * 0.005, 1.0),
                                          notional * 0.01), 0.0)
        rate = section31_rate(date)
        section31 = float(np.maximum(-quantity * np.nan_to_num(price, nan=0.0), 0).sum()
                          * rate)
        return float(commissions.sum()), section31

    def simulate(self, decisions: pd.DatetimeIndex, start, end, decide):
        config, market = self.config, self.market
        calendar, symbols = market.calendar, market.symbols
        dates = calendar[(calendar >= start) & (calendar < end)]
        if dates.empty:
            raise ValueError("账户区间没有交易日。")
        decisions = pd.DatetimeIndex(decisions)
        if decisions.empty or not decisions.is_unique or not decisions.is_monotonic_increasing:
            raise ValueError("决策日期为空、重复或未排序。")
        decision_set = set(decisions)
        pending = {}
        expiry = [None] * config.holding
        sleeve_shares = np.zeros((config.holding, len(symbols)))
        sleeve_cash = np.full(config.holding, config.capital / config.holding)
        shares = pd.DataFrame(0.0, index=dates, columns=symbols)
        commission = pd.Series(0.0, index=dates)
        section31 = pd.Series(0.0, index=dates)
        records, weights, vintage_shares = [], [], []
        for date in dates:
            position = calendar.get_loc(date)
            price = market.close.loc[date].to_numpy(dtype=float)
            if date in decision_set:
                entry = calendar[position + config.delay]
                exit_date = calendar[position + config.delay + config.holding]
                sleeve = calendar.get_loc(entry) % config.holding
                old_q = sleeve_shares[sleeve]
                if not np.isfinite(price[np.abs(old_q) > 1e-10]).all():
                    raise ValueError("决策时已有持仓缺少收盘价。")
                known_nav = sleeve_cash[sleeve] + old_q @ np.nan_to_num(price, nan=0.0)
                if known_nav <= 0:
                    raise ValueError("决策时子账户净值非正。")
                target, detail = decide(date, old_q.copy(), known_nav, price.copy())
                target = np.asarray(target, dtype=float)
                if target.shape != (len(symbols),) or not np.isfinite(target).all():
                    raise ValueError("目标权重维度或数值无效。")
                pending[entry] = (date, exit_date, sleeve, target, detail)
                weights.append(pd.Series(target, index=symbols, name=date))
            sleeve = position % config.holding
            active = expiry[sleeve] == date
            order = pending.pop(date, None)
            if order is not None:
                signal_date, exit_date, order_sleeve, target, detail = order
                if order_sleeve != sleeve:
                    raise ValueError("执行日与五批子账户编号不一致。")
                if np.abs(sleeve_shares[sleeve]).max() > 1e-10 and not active:
                    raise ValueError("子账户在持有期结束前发生再次交易。")
                old_q = sleeve_shares[sleeve].copy()
                if not np.isfinite(price[(np.abs(old_q) > 1e-10)
                                         | (np.abs(target) > 1e-10)]).all():
                    raise ValueError("交易资产执行日价格缺失。")
                valued_price = np.nan_to_num(price, nan=0.0)
                entry_nav = sleeve_cash[sleeve] + old_q @ valued_price
                if entry_nav <= 0:
                    raise ValueError("执行日子账户净值非正。")
                new_q = np.divide(target * entry_nav, price,
                                  out=np.zeros_like(target),
                                  where=np.isfinite(price) & (price > 0))
                delta = new_q - old_q
                c, s = self.transaction_cost(delta, price, date)
                sleeve_cash[sleeve] = entry_nav - new_q @ valued_price - c - s
                sleeve_shares[sleeve] = new_q
                expiry[sleeve] = exit_date
                commission.loc[date], section31.loc[date] = c, s
                exit_price = market.close.loc[exit_date].to_numpy(dtype=float)
                active_assets = np.abs(target) > 1e-10
                if not np.isfinite(exit_price[active_assets]).all():
                    raise ValueError("持仓到期日资产价格不完整。")
                period_return = target[active_assets] @ (
                    exit_price[active_assets] / price[active_assets] - 1
                )
                records.append({"date": signal_date, "sleeve": sleeve + 1,
                                "entry": date, "exit": exit_date,
                                "entry_capital": entry_nav, **detail,
                                "net": target.sum(), "gross": np.abs(target).sum(),
                                "min_weight": target.min(), "max_weight": target.max(),
                                "actual_turnover": np.abs(delta * price).sum() / entry_nav,
                                "return_5d": period_return,
                                "pnl_5d": entry_nav * period_return})
                vintage_shares.append(pd.Series(new_q, index=symbols, name=signal_date))
            elif active:
                old_q = sleeve_shares[sleeve].copy()
                if not np.isfinite(price[np.abs(old_q) > 1e-10]).all():
                    raise ValueError("平仓日资产价格不完整。")
                c, s = self.transaction_cost(-old_q, price, date)
                sleeve_cash[sleeve] += old_q @ np.nan_to_num(price, nan=0.0) - c - s
                sleeve_shares[sleeve] = 0.0
                expiry[sleeve] = None
                commission.loc[date], section31.loc[date] = c, s
            aggregate = sleeve_shares.sum(axis=0)
            if not np.isfinite(price[np.abs(aggregate) > 1e-10]).all():
                raise ValueError("持仓估值缺少收盘价。")
            shares.loc[date] = aggregate
        if pending or np.abs(sleeve_shares).max() > 1e-8:
            raise ValueError("阶段终点仍有未执行订单或未到期仓位。")
        fees = commission + section31
        daily, trades = self.account_from_shares(shares, fees=fees)
        ledger_equity = sleeve_cash.sum() + sleeve_shares.sum(axis=0) @ np.nan_to_num(
            market.close.loc[dates[-1]].to_numpy(dtype=float), nan=0.0
        )
        records = pd.DataFrame(records).set_index("date")
        if not (np.isclose(daily["equity"].iloc[-1], ledger_equity, atol=1e-6)
                and np.isclose(daily["equity"].iloc[-1] - config.capital,
                               records["pnl_5d"].sum() - fees.sum(), atol=1e-5)):
            raise ValueError("逐日现金、五日持仓损益与费用未对账。")
        account_weights = shares.mul(market.close.reindex(dates)).fillna(0).div(
            daily["equity"], axis=0
        )
        return {"daily": daily, "records": records, "weights": pd.DataFrame(weights),
                "account_weights": account_weights,
                "vintage_shares": pd.DataFrame(vintage_shares),
                "shares": shares, "trades": trades,
                "commission": commission, "section31": section31}

    def optimize(self, mu, sigma, previous, risk, turnover):
        config = self.config
        mu = np.asarray(mu, dtype=float) / config.scale
        sigma = np.asarray(sigma, dtype=float) / config.scale**2
        previous = np.asarray(previous, dtype=float)
        n = len(mu)
        lower, upper = map(float, config.asset_bounds)
        if not (lower < 0 < upper and sigma.shape == (n, n)
                and previous.shape == (n,)):
            raise ValueError("优化器输入维度或仓位边界无效。")
        target_net = sum(config.net_band) / 2
        target_gross = sum(config.gross_band) / 2
        positive_mass = (target_gross + target_net) / 2
        negative_mass = (target_gross - target_net) / 2
        feasible_counts = [count for count in range(1, n)
                           if positive_mass / count <= upper
                           and negative_mass / (n - count) <= -lower]
        if not feasible_counts:
            raise ValueError("单资产边界与净／总敞口不相容。")
        objective_scale = max(np.abs(mu).max(), risk * np.diag(sigma).max(),
                              turnover, 1e-8)

        def objective(w):
            difference = w - previous
            return (-w @ mu + risk * w @ sigma @ w
                    + turnover * np.sqrt(difference**2 + SMOOTH_EPS**2).sum()) / objective_scale

        def gradient(w):
            difference = w - previous
            return (-mu + 2 * risk * sigma @ w
                    + turnover * difference / np.sqrt(difference**2 + SMOOTH_EPS**2)) / objective_scale

        def gross(w):
            return np.sqrt(w**2 + SMOOTH_EPS**2).sum()

        def gross_gradient(w):
            return w / np.sqrt(w**2 + SMOOTH_EPS**2)

        ones = np.ones(n)
        constraints = [
            {"type": "ineq", "fun": lambda w: w.sum() - config.net_band[0],
             "jac": lambda w: ones},
            {"type": "ineq", "fun": lambda w: config.net_band[1] - w.sum(),
             "jac": lambda w: -ones},
            {"type": "ineq", "fun": lambda w: gross(w) - config.gross_band[0],
             "jac": gross_gradient},
            {"type": "ineq", "fun": lambda w: config.gross_band[1] - gross(w),
             "jac": lambda w: -gross_gradient(w)},
        ]

        def initial_weights(long_count, order=None):
            if order is None:
                order = np.argsort(mu, kind="stable")
            long_members = np.zeros(n, dtype=bool)
            long_members[order[-long_count:]] = True
            net = sum(config.net_band) / 2
            gross_target = sum(config.gross_band) / 2
            long_mass = (gross_target + net) / 2
            short_mass = (gross_target - net) / 2
            initial = np.empty(n)
            initial[long_members] = long_mass / long_count
            initial[~long_members] = -short_mass / (n - long_count)
            if initial.min() < lower - config.tolerance or initial.max() > upper + config.tolerance:
                raise ValueError("优化器初始仓位不满足单资产边界。")
            return initial

        signal_count = min(feasible_counts, key=lambda count: abs(count - (n // 2 + 1)))
        signal_start = initial_weights(signal_count)
        neutral_start = initial_weights(signal_count, np.arange(n))
        starts = [] if np.allclose(previous, 0) else [previous]
        starts.append(signal_start)
        if not np.allclose(neutral_start, signal_start, atol=1e-12, rtol=0):
            starts.append(neutral_start)
        solutions, attempts = [], []

        def solve(start, maxiter):
            return minimize(objective, start, jac=gradient, method="SLSQP",
                            bounds=[config.asset_bounds] * n, constraints=constraints,
                            options={"maxiter": maxiter, "ftol": 1e-10})

        def feasible(solution):
            w = solution.x
            return (solution.success and np.isfinite(w).all()
                    and config.net_band[0] - config.tolerance <= w.sum()
                    <= config.net_band[1] + config.tolerance
                    and config.gross_band[0] - config.tolerance <= np.abs(w).sum()
                    <= config.gross_band[1] + config.tolerance
                    and w.min() >= lower - config.tolerance
                    and w.max() <= upper + config.tolerance)

        for start in starts:
            candidate = solve(start, config.solver_maxiter)
            attempts.append(candidate)
            if feasible(candidate):
                solutions.append(candidate)
        retry_used = not solutions
        if retry_used:
            for count in dict.fromkeys((min(feasible_counts), max(feasible_counts), signal_count)):
                candidate = solve(initial_weights(count), config.solver_retry_maxiter)
                attempts.append(candidate)
                if feasible(candidate):
                    solutions.append(candidate)
        if not solutions:
            for attempted in attempts:
                signs = np.where(attempted.x < 0, -1.0, 1.0)
                amplitude_limits = np.where(signs > 0, upper, -lower)
                amplitude_bounds = list(zip(np.zeros(n), amplitude_limits, strict=True))
                rows = np.vstack([signs, -signs, ones, -ones])
                limits = np.array([config.net_band[1], -config.net_band[0],
                                   config.gross_band[1], -config.gross_band[0]])
                if risk == 0 and turnover == 0:
                    rescued = linprog(-signs * mu, A_ub=rows, b_ub=limits,
                                      bounds=amplitude_bounds, method="highs")
                else:
                    linear_constraints = [
                        {"type": "ineq", "fun": lambda a, row=row, limit=limit:
                         limit - row @ a,
                         "jac": lambda a, row=row: -row}
                        for row, limit in zip(rows, limits, strict=True)
                    ]
                    rescued = minimize(lambda a, signs=signs: objective(signs * a),
                                       np.clip(np.abs(attempted.x), 0, amplitude_limits),
                                       jac=lambda a, signs=signs: signs * gradient(signs * a),
                                       method="SLSQP", bounds=amplitude_bounds,
                                       constraints=linear_constraints,
                                       options={"maxiter": config.solver_retry_maxiter,
                                                "ftol": 1e-10})
                if rescued.success:
                    rescued.x = signs * rescued.x
                    if feasible(rescued):
                        solutions.append(rescued)
        if not solutions:
            details = [(item.success, str(item.message),
                        float(item.x.sum()), float(np.abs(item.x).sum()))
                       for item in attempts]
            raise RuntimeError(f"自由多空权重优化没有可用解：{details}")

        chosen = min(solutions, key=lambda result: objective(result.x))
        w = chosen.x
        constraint_error = max(
            0, config.net_band[0] - w.sum(), w.sum() - config.net_band[1],
            config.gross_band[0] - np.abs(w).sum(),
            np.abs(w).sum() - config.gross_band[1],
            lower - w.min(), w.max() - upper)
        if constraint_error > config.tolerance:
            raise RuntimeError("组合权重超过仓位约束容差。")
        return {"weights": w, "converged": True, "solver_retry": retry_used,
                "converged_starts": len(solutions), "attempted_starts": len(attempts),
                "constraint_error": constraint_error,
                "predicted_return": w @ mu, "residual_variance": w @ sigma @ w,
                "decision_turnover": np.abs(w - previous).sum(),
                "largest_weight": np.abs(w).max(),
                "weight_below_0_1pct": int((np.abs(w) < .001).sum()),
                "long_assets": int((w > 1e-8).sum()),
                "short_assets": int((w < -1e-8).sum()),
                "zero_assets": int((np.abs(w) <= 1e-8).sum())}

    def optimized_account(self, predictions: pd.Series, covariances: dict,
                          start, end, risk: float, turnover: float,
                          model_name: str):
        config, market = self.config, self.market
        matrix = predictions.unstack("symbol").reindex(columns=market.symbols)
        possible = self.signal_dates(start, end).intersection(matrix.index)
        required_long = np.ceil((sum(config.gross_band) + sum(config.net_band))
                                / 4 / config.asset_bounds[1])
        required_short = np.ceil((sum(config.gross_band) - sum(config.net_band))
                                 / 4 / -config.asset_bounds[0])
        minimum_assets = int(required_long + required_short)
        eligibility = {}
        for date in possible:
            month = date.normalize().replace(day=1)
            sigma = covariances.get((model_name, month))
            if sigma is None:
                continue
            valid = (market.target.loc[date].notna().to_numpy()
                     & matrix.loc[date].notna().to_numpy()
                     & np.isfinite(np.diag(sigma)))
            if valid.sum() >= minimum_assets and np.isfinite(sigma[np.ix_(valid, valid)]).all():
                eligibility[date] = valid
        decisions = pd.DatetimeIndex(eligibility)

        def decide(date, old_q, known_nav, signal_price):
            previous = old_q * np.nan_to_num(signal_price, nan=0.0) / known_nav
            month = date.normalize().replace(day=1)
            sigma = covariances[model_name, month]
            valid = eligibility[date]
            solution = self.optimize(matrix.loc[date].to_numpy()[valid] * config.scale,
                                     sigma[np.ix_(valid, valid)], previous[valid],
                                     risk, turnover)
            active_weights = solution.pop("weights")
            w = np.zeros(len(market.symbols))
            w[valid] = active_weights
            solution["decision_turnover"] = np.abs(w - previous).sum()
            solution["zero_assets"] = int((np.abs(w) <= 1e-8).sum())
            component = active_weights * (sigma[np.ix_(valid, valid)] @ active_weights)
            contribution_total = abs(component).sum()
            concentration = (abs(component) / contribution_total
                             if contribution_total > 0
                             else np.full(len(component), 1 / len(component)))
            solution["effective_risk_assets"] = 1 / np.square(concentration).sum()
            solution["eligible_assets"] = int(valid.sum())
            return w, solution

        account = self.simulate(decisions, start, end, decide)
        if account["records"]["constraint_error"].max() > config.tolerance:
            raise ValueError("组合权重超过仓位约束容差。")
        return account

    def fee_components(self, account):
        if "commission" in account and "section31" in account:
            return account["commission"], account["section31"]
        shares = account["shares"]
        quantity = shares.diff().fillna(shares.iloc[0])
        traded_value = quantity * self.market.close.reindex(shares.index)
        notional = traded_value.abs()
        actual_quantity = quantity.abs().copy()
        split_columns = [s for s in SPLIT_SYMBOLS if s in actual_quantity]
        actual_quantity.loc[actual_quantity.index < SPLIT_DATE, split_columns] *= 0.5
        commission = np.minimum(
            np.maximum(actual_quantity * 0.005, 1.0), notional * 0.01
        ).where(notional > 1e-10, 0.0).sum(axis=1)
        section_rate = pd.Series([section31_rate(date) for date in shares.index],
                                 index=shares.index)
        section31 = (-traded_value).clip(lower=0).sum(axis=1) * section_rate
        return commission, section31

    def after_fee(self, account, fee):
        daily = account["daily"].copy()
        if not np.allclose(daily["fee"].to_numpy(), fee.reindex(daily.index).to_numpy(),
                           atol=1e-6, rtol=1e-9):
            raise ValueError("费用必须在逐日现金账本中同步入账。")
        return daily, performance(daily)
