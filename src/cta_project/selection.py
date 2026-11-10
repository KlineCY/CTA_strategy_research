"""整体验证期的预测模型与组合参数选择。"""

import numpy as np
import pandas as pd

from .config import Config
from .data import Market
from .models import RollingResult, metrics
from .portfolio import PortfolioEngine, active_period_statistics, effr_cash_sharpe


def compare_models(result: RollingResult, market: Market, config: Config) -> pd.DataFrame:
    rows = []
    end = config.stages["Validation"][1]
    for name, prediction in result.predictions.items():
        truth = market.target.stack().reindex(prediction.index)
        mature = market.label_end.reindex(
            prediction.index.get_level_values("date")
        ).lt(end).to_numpy()
        rows.append({"模型": name, **metrics(truth.loc[mature], prediction.loc[mature])})
    frame = pd.DataFrame(rows).sort_values(["MSE", "MAE"], kind="stable").reset_index(drop=True)
    frame["预测误差门槛合格"] = (
        frame["MSE"].le(frame["MSE"].min() * (1 + config.prediction_error_tolerance))
        & frame["MAE"].le(frame["MAE"].min() * (1 + config.prediction_error_tolerance))
    )
    if not frame["预测误差门槛合格"].any():
        raise ValueError("整体验证期没有模型通过预测误差门槛。")
    return frame


def evaluate_portfolio_point(name: str, lam: float, gam: float,
                             result: RollingResult, engine: PortfolioEngine,
                             start: pd.Timestamp, end: pd.Timestamp) -> dict:
    account = engine.optimized_account(result.predictions[name], result.covariances,
                                       start, end, lam, gam, name)
    commission, section31 = engine.fee_components(account)
    fee = commission + section31
    _, statistics = engine.after_fee(account, fee)
    return {"模型": name, "lambda": lam, "gamma": gam,
            "扣费后累计收益": statistics["累计收益"],
            "扣费后几何年化收益": statistics["几何年化收益"],
            "扣费后最大回撤": statistics["最大回撤"],
            "平均每日换手": statistics["平均每日换手"],
            "总费用美元": fee.sum(),
            "求解器重试次数": int(account["records"]["solver_retry"].sum()),
            "平均多头资产数": account["records"]["long_assets"].mean(),
            "平均空头资产数": account["records"]["short_assets"].mean(),
            "平均零权重资产数": account["records"]["zero_assets"].mean(),
            "最大多头权重": account["records"]["max_weight"].max(),
            "最小空头权重": account["records"]["min_weight"].min()}


def choose_portfolio_params(full_grid: dict[str, pd.DataFrame],
                            model_comparison: pd.DataFrame,
                            config: Config):
    rows = []
    for name, frame in full_grid.items():
        feasible = frame.loc[
            frame["扣费后最大回撤"].ge(-config.max_drawdown_magnitude)
        ].copy()
        if feasible.empty:
            chosen = frame.sort_values(
                ["扣费后最大回撤", "扣费后几何年化收益"],
                ascending=[False, False], kind="stable"
            ).iloc[0]
            status, positive_count = "诊断对照｜不满足回撤约束", 0
        else:
            best = feasible.sort_values(
                ["扣费后几何年化收益", "扣费后最大回撤"],
                ascending=[False, False], kind="stable"
            ).iloc[0]
            zero = frame.loc[frame["gamma"].eq(0)].set_index("lambda")["平均每日换手"]
            positive = feasible.loc[
                feasible["gamma"].gt(0)
                & feasible["扣费后几何年化收益"].ge(
                    best["扣费后几何年化收益"] - config.annual_return_tolerance)
            ].copy()
            positive["同lambda零gamma平均每日换手"] = positive["lambda"].map(zero)
            positive = positive.loc[
                positive["平均每日换手"].lt(positive["同lambda零gamma平均每日换手"])
            ].sort_values(["扣费后几何年化收益", "扣费后最大回撤"],
                          ascending=[False, False], kind="stable")
            chosen = positive.iloc[0] if len(positive) else best
            status, positive_count = "满足回撤约束", len(positive)
        rows.append({"模型": name, "状态": status,
                     "lambda": chosen["lambda"], "gamma": chosen["gamma"],
                     "扣费后几何年化收益": chosen["扣费后几何年化收益"],
                     "扣费后最大回撤": chosen["扣费后最大回撤"],
                     "平均每日换手": chosen["平均每日换手"],
                     "总费用美元": chosen["总费用美元"],
                     "可行组合数": len(feasible), "正gamma候选数": positive_count})
    portfolio_comparison = pd.DataFrame(rows)
    joint = portfolio_comparison.merge(
        model_comparison[["模型", "MSE", "MAE", "预测误差门槛合格"]],
        on="模型", validate="one_to_one"
    )
    joint["回撤门槛合格"] = joint["扣费后最大回撤"].ge(-config.max_drawdown_magnitude)
    joint["综合门槛合格"] = joint["预测误差门槛合格"] & joint["回撤门槛合格"]
    eligible = joint.loc[joint["综合门槛合格"]].sort_values(
        ["扣费后几何年化收益", "扣费后最大回撤"],
        ascending=[False, False], kind="stable"
    )
    if eligible.empty:
        raise ValueError("没有模型同时通过预测误差和组合回撤门槛。")
    selected = eligible.iloc[0]["模型"]
    joint["综合选择结果"] = np.where(joint["模型"].eq(selected), "入选", "未入选")
    return selected, portfolio_comparison, joint


def evaluate_stage(name: str, stage: str, result: RollingResult,
                   engine: PortfolioEngine, config: Config,
                   lam: float, gam: float):
    start, end = config.stages[stage]
    account = engine.optimized_account(result.predictions[name], result.covariances,
                                       start, end, lam, gam, name)
    commission, section31 = engine.fee_components(account)
    daily, stats = engine.after_fee(account, commission + section31)
    prediction = result.predictions[name]
    truth = engine.market.target.stack().reindex(prediction.index)
    mature = engine.market.label_end.reindex(
        prediction.index.get_level_values("date")
    ).lt(end).to_numpy()
    error = metrics(truth.loc[mature], prediction.loc[mature])
    cash = effr_cash_sharpe(daily["equity"], engine.market.effr)
    active = active_period_statistics(account, engine.market.effr)
    active_stats = active["绩效"]
    row = {"阶段": stage, "模型": name, "lambda": lam, "gamma": gam,
           "MSE": error["MSE"], "MAE": error["MAE"],
           "扣费后累计收益": stats["累计收益"],
           "扣费后几何年化收益": stats["几何年化收益"],
           "扣费后最大回撤": stats["最大回撤"],
           "EFFR现金基准Sharpe": cash["EFFR现金基准Sharpe"],
           "年化波动率": stats["年化波动"],
           "有效交易期起始日": active["起始成交日"],
           "空仓准备交易日数": active["空仓准备交易日数"],
           "有效交易期几何年化收益": active_stats["几何年化收益"],
           "有效交易期最大回撤": active_stats["最大回撤"],
           "有效交易期年化波动率": active_stats["年化波动"],
           "有效交易期EFFR现金基准Sharpe": active["EFFR现金基准Sharpe"],
           "平均每日换手": stats["平均每日换手"],
           "平均目标净敞口": account["records"]["net"].mean(),
           "平均目标总敞口": account["records"]["gross"].mean(),
           "实际净敞口最小值": daily["net"].min(),
           "实际净敞口最大值": daily["net"].max(),
           "实际总敞口最小值": daily["gross"].min(),
           "实际总敞口最大值": daily["gross"].max(),
           "未收敛求解次数": int((~account["records"]["converged"]).sum()),
           "总费用美元": float((commission + section31).sum()),
           "平均多头资产数": account["records"]["long_assets"].mean(),
           "平均空头资产数": account["records"]["short_assets"].mean(),
           "平均零权重资产数": account["records"]["zero_assets"].mean()}
    return row, account, daily
