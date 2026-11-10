"""从原始数据到验证选择、三段账户与风险控制的完整入口。"""

from dataclasses import replace
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config

from .config import Config, MODEL_NAMES, PARAMETER_VALUES
from .covariance import rolling_oos_covariances
from .data import load_market
from .models import RollingResult, rolling_model_families
from .portfolio import PortfolioEngine, active_period_statistics
from .risk import RiskEngine
from .selection import (choose_portfolio_params, compare_models,
                        evaluate_portfolio_point, evaluate_stage)
from .signals import SignalEngine
from .storage import check_or_create_run, load_or_compute


TRAIN_ROLL_START = pd.Timestamp("2021-06-01", tz="UTC")


def _read_grid(path, model, lambdas):
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(path)
    except (OSError, UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    required = ("模型", "lambda", "gamma", "扣费后几何年化收益",
                "扣费后最大回撤", "平均每日换手", "总费用美元")
    if not set(required).issubset(frame.columns) or len(frame) != len(lambdas) * len(PARAMETER_VALUES):
        return None
    numeric = frame[list(required[1:])].apply(pd.to_numeric, errors="coerce")
    if (not np.isfinite(numeric.to_numpy()).all()
            or not frame["模型"].eq(model).all()
            or set(numeric["lambda"]) != set(lambdas)
            or set(numeric["gamma"]) != set(PARAMETER_VALUES)
            or not numeric.groupby(["lambda", "gamma"]).size().eq(1).all()):
        return None
    return frame


def _write_grid(path, frame):
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def combine_monthly_results(parts: list[RollingResult]) -> RollingResult:
    if not parts:
        raise ValueError("逐月预测结果为空。")
    names = tuple(parts[0].predictions)
    if any(tuple(part.predictions) != names for part in parts):
        raise ValueError("逐月预测模型列表不一致。")
    predictions = {
        name: pd.concat([part.predictions[name] for part in parts]).sort_index()
        for name in names
    }
    covariances = {key: value for part in parts
                   for key, value in part.covariances.items()}
    return RollingResult(
        predictions, covariances,
        pd.concat([part.folds for part in parts], ignore_index=True),
        pd.concat([part.searches for part in parts], ignore_index=True),
        pd.concat([part.importances for part in parts], ignore_index=True),
        pd.concat([part.skipped for part in parts], ignore_index=True),
    )


class ResearchProject:
    def __init__(self, config: Config):
        self.config = config
        self.market = load_market(config)
        self.signals = SignalEngine(self.market, config)
        self.portfolio = PortfolioEngine(self.market, config)
        self.risk = RiskEngine(self.market, self.signals, self.portfolio, config)

    def predictions(self, stage: str, workers: int = 1,
                    variant: str | None = None):
        config = self.config
        variant = variant or config.vol_state_variant
        if variant not in ("cross_sectional", "level"):
            raise ValueError("VOL_STATE候选须为cross_sectional或level。")
        signals = self.signals if variant == config.vol_state_variant else SignalEngine(
            self.market, replace(config, vol_state_variant=variant)
        )
        start, end = config.stages[stage]
        if stage == "Training":
            start = TRAIN_ROLL_START
        path = config.results_dir / "cache" / f"predictions_{variant}_{stage}.joblib"

        def calculate_stage():
            parts = []
            for month in pd.date_range(start, end, freq="MS", inclusive="left"):
                month_end = min(month + pd.DateOffset(months=1), end)
                month_path = (config.results_dir / "cache"
                              / f"predictions_{variant}_{stage}_{month:%Y%m}.joblib")
                parts.append(load_or_compute(
                    month_path,
                    lambda month=month, month_end=month_end: rolling_model_families(
                        self.market, signals, config, month, month_end,
                        names=MODEL_NAMES, n_jobs=workers,
                    ),
                ))
            return combine_monthly_results(parts)

        result = load_or_compute(path, calculate_stage)
        result.folds.to_csv(config.results_dir / f"逐月模型_{variant}_{stage}.csv", index=False)
        result.searches.to_csv(config.results_dir / f"局部超参数_{variant}_{stage}.csv", index=False)
        result.importances.to_csv(config.results_dir / f"特征系数_{variant}_{stage}.csv", index=False)
        result.skipped.to_csv(config.results_dir / f"缺口排除月份_{variant}_{stage}.csv", index=False)
        if result.folds.empty or not result.folds["最晚成熟标签"].lt(result.folds["月份"]).all():
            raise ValueError(f"{stage}存在未成熟训练标签。")
        if set(result.predictions) != set(MODEL_NAMES):
            raise ValueError(f"{stage}缺少候选模型。")
        return result

    def portfolio_grid(self, validation, workers: int = 1):
        config = self.config
        start, end = config.stages["Validation"]
        output = config.results_dir / "portfolio_grids"
        output.mkdir(parents=True, exist_ok=True)
        full_grid = {}
        for name in MODEL_NAMES:
            name_slug = name.replace(" ", "_")
            path = output / f"{name_slug}_14x14.csv"
            frame = _read_grid(path, name, PARAMETER_VALUES)
            if frame is None:
                batches = []
                for batch_number in range(7):
                    part_path = output / f"{name_slug}_batch_{batch_number + 1}.csv"
                    lambdas = PARAMETER_VALUES[batch_number * 2:(batch_number + 1) * 2]
                    part = _read_grid(part_path, name, lambdas)
                    if part is None:
                        points = [(lam, gam) for lam in lambdas for gam in PARAMETER_VALUES]
                        if workers > 1:
                            with parallel_config(backend="loky", inner_max_num_threads=1):
                                rows = Parallel(n_jobs=workers)(
                                    delayed(evaluate_portfolio_point)(
                                        name, lam, gam, validation, self.portfolio, start, end
                                    ) for lam, gam in points
                                )
                        else:
                            rows = [evaluate_portfolio_point(
                                name, lam, gam, validation, self.portfolio, start, end
                            ) for lam, gam in points]
                        part = pd.DataFrame(rows)
                        _write_grid(part_path, part)
                    batches.append(part)
                frame = pd.concat(batches, ignore_index=True)
                _write_grid(path, frame)
            if (len(frame) != 196
                    or not frame.groupby(["lambda", "gamma"]).size().eq(1).all()
                    or set(frame["lambda"]) != set(PARAMETER_VALUES)
                    or set(frame["gamma"]) != set(PARAMETER_VALUES)):
                raise ValueError(f"{name}组合网格不完整。")
            full_grid[name] = frame
        return full_grid

    def run_full(self, workers: int = 1) -> dict:
        """从本地输入数据执行全部模型与账户研究。"""
        config = self.config
        check_or_create_run(config)
        validation_candidates = {
            variant: self.predictions("Validation", workers, variant)
            for variant in ("cross_sectional", "level")
        }
        first_name = MODEL_NAMES[0]
        common_index = validation_candidates["cross_sectional"].predictions[first_name].index.intersection(
            validation_candidates["level"].predictions[first_name].index
        )
        if common_index.empty:
            raise ValueError("VOL_STATE两种口径没有共同的Validation预测样本。")
        feature_rows = []
        for variant, candidate in validation_candidates.items():
            aligned = RollingResult(
                {name: series.reindex(common_index)
                 for name, series in candidate.predictions.items()},
                candidate.covariances, candidate.folds,
                candidate.searches, candidate.importances, candidate.skipped,
            )
            scores = compare_models(aligned, self.market, config)
            best = scores.iloc[0]
            feature_rows.append({"VOL_STATE口径": variant, "最优预测模型": best["模型"],
                                 "MSE": best["MSE"], "MAE": best["MAE"],
                                 "成熟标签样本数": best["样本数"]})
        feature_choice = pd.DataFrame(feature_rows).sort_values(
            ["MSE", "MAE"], kind="stable"
        ).reset_index(drop=True)
        feature_choice.to_csv(config.results_dir / "Validation波动率特征选择.csv", index=False)
        selected_variant = feature_choice.iloc[0]["VOL_STATE口径"]
        validation = validation_candidates[selected_variant]
        comparison = compare_models(validation, self.market, config)
        comparison.to_csv(config.results_dir / "整体验证期模型误差.csv", index=False)
        grids = self.portfolio_grid(validation, workers)
        selected, choices, joint = choose_portfolio_params(grids, comparison, config)
        choices.to_csv(config.results_dir / "各模型组合参数选择.csv", index=False)
        joint.to_csv(config.results_dir / "整体验证期综合模型选择.csv", index=False)
        stages = {"Training": self.predictions("Training", workers, selected_variant),
                  "Validation": validation}
        indexed = choices.set_index("模型")
        rows, accounts = [], {}

        def evaluate_accounts(stage, result):
            for name in MODEL_NAMES:
                lam, gam = float(indexed.loc[name, "lambda"]), float(indexed.loc[name, "gamma"])
                path = config.results_dir / "cache" / f"account_{stage}_{name.replace(' ', '_')}_{lam:.10g}_{gam:.10g}.joblib"
                row, account, daily = load_or_compute(
                    path, lambda stage=stage, name=name, result=result, lam=lam, gam=gam:
                    evaluate_stage(name, stage, result, self.portfolio, config, lam, gam)
                )
                if stage == "Validation":
                    matching = grids[name].loc[
                        np.isclose(grids[name]["lambda"], lam)
                        & np.isclose(grids[name]["gamma"], gam)
                    ]
                    if (len(matching) != 1
                            or not np.isclose(row["扣费后几何年化收益"],
                                              matching.iloc[0]["扣费后几何年化收益"], atol=1e-9)
                            or not np.isclose(row["扣费后最大回撤"],
                                              matching.iloc[0]["扣费后最大回撤"], atol=1e-9)):
                        raise ValueError("Validation账户与参数网格结果不一致。")
                rows.append(row)
                accounts[stage, name] = account
                pd.DataFrame(rows).to_csv(config.results_dir / "三阶段账户计算进度.csv",
                                          index=False)

        for stage, result in stages.items():
            evaluate_accounts(stage, result)
        validation_base = accounts["Validation", selected]
        validation_start, validation_end = config.stages["Validation"]
        months = pd.date_range(validation_start, validation_end, freq="MS", inclusive="left")
        historical_predictions = pd.concat([
            stages["Training"].predictions[selected],
            validation.predictions[selected],
        ]).sort_index()
        oos_covariances, covariance_windows = rolling_oos_covariances(
            self.market, config, historical_predictions, months, selected
        )
        covariance_windows.to_csv(
            config.results_dir / "Validation样本外残差协方差窗口.csv", index=False
        )
        covariance_accounts = {"训练窗拟合残差": validation_base}
        lam = float(indexed.loc[selected, "lambda"])
        gam = float(indexed.loc[selected, "gamma"])
        validation_accounts = pd.DataFrame(rows)
        baseline = validation_accounts.loc[
            validation_accounts["阶段"].eq("Validation")
            & validation_accounts["模型"].eq(selected)
        ].iloc[0]
        covariance_rows = [{
            "协方差口径": "训练窗拟合残差", "模型": selected,
            "lambda": lam, "gamma": gam,
            "Validation决策日数": len(validation_base["records"]),
            "可用协方差月份数": len(months),
            "扣费后几何年化收益": baseline["扣费后几何年化收益"],
            "扣费后最大回撤": baseline["扣费后最大回撤"],
            "总费用美元": baseline["总费用美元"],
            "同决策日可比": True,
        }]
        if len(oos_covariances) == len(months):
            candidate_result = RollingResult(
                {selected: validation.predictions[selected]}, oos_covariances,
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
            )
            candidate_row, candidate_account, _ = evaluate_stage(
                selected, "Validation", candidate_result,
                self.portfolio, config, lam, gam
            )
            if candidate_account["records"].index.equals(validation_base["records"].index):
                covariance_accounts["既往逐月样本外残差"] = candidate_account
                covariance_rows.append({
                    "协方差口径": "既往逐月样本外残差", "模型": selected,
                    "lambda": lam, "gamma": gam,
                    "Validation决策日数": len(candidate_account["records"]),
                    "可用协方差月份数": len(oos_covariances),
                    "扣费后几何年化收益": candidate_row["扣费后几何年化收益"],
                    "扣费后最大回撤": candidate_row["扣费后最大回撤"],
                    "总费用美元": candidate_row["总费用美元"],
                    "同决策日可比": True,
                })
        if len(covariance_rows) == 1:
            covariance_rows.append({
                "协方差口径": "既往逐月样本外残差", "模型": selected,
                "lambda": lam, "gamma": gam,
                "Validation决策日数": np.nan,
                "可用协方差月份数": len(oos_covariances),
                "扣费后几何年化收益": np.nan,
                "扣费后最大回撤": np.nan,
                "总费用美元": np.nan,
                "同决策日可比": False,
            })
        covariance_comparison = pd.DataFrame(covariance_rows)
        comparable = covariance_comparison.loc[covariance_comparison["同决策日可比"]]
        qualified = comparable.loc[
            comparable["扣费后最大回撤"].ge(-config.max_drawdown_magnitude)
        ].sort_values(["扣费后几何年化收益", "扣费后最大回撤"],
                      ascending=[False, False], kind="stable")
        preferred = qualified.iloc[0]["协方差口径"] if len(qualified) else None
        covariance_comparison["Validation优先候选"] = covariance_comparison[
            "协方差口径"
        ].eq(preferred)
        covariance_comparison.to_csv(
            config.results_dir / "Validation残差协方差对照.csv", index=False
        )
        risk_grid, risk_choice, best_state = self.risk.select_control(validation_base)
        risk_grid.to_csv(config.results_dir / "Validation组合风险控制网格.csv", index=False)
        stages["历史Test"] = self.predictions("历史Test", workers, selected_variant)
        evaluate_accounts("历史Test", stages["历史Test"])
        three_stage = pd.DataFrame(rows)
        three_stage.to_csv(config.results_dir / "七类模型三阶段扣费后表现.csv", index=False)
        risk_rows, risk_accounts = [], {}
        selected_risk_label = ("基础组合" if risk_choice["方案"] == "基础组合"
                               else "状态控制候选")
        for stage in stages:
            base = accounts[stage, selected]
            decisions = self.risk.prepare_decisions(base)
            for label, target, multiplier in (
                ("基础组合", None, 1.0),
                ("状态控制候选", float(best_state["目标年化波动率"]),
                 float(best_state["高波动乘数"])),
            ):
                account = self.risk.controlled_account(base, decisions, target, multiplier)
                risk_accounts[stage, label] = account
                risk_rows.append({"阶段": stage, **self.risk.comparison_row(
                    label, target, multiplier, account)})
            risk_accounts[stage, "最终选定方案"] = risk_accounts[stage, selected_risk_label]
        risk_summary = pd.DataFrame(risk_rows)
        risk_summary["Validation选定"] = risk_summary["方案"].eq(selected_risk_label)
        risk_summary.to_csv(config.results_dir / "三阶段组合风险控制表现.csv", index=False)
        selected_risk_summary = risk_summary.loc[risk_summary["Validation选定"]].copy()
        selected_risk_summary.to_csv(config.results_dir / "三阶段最终选定账户表现.csv", index=False)
        benchmarks, benchmark_rows = {}, []
        for stage in stages:
            strategy = risk_accounts[stage, "最终选定方案"]
            first_entry = strategy["records"]["entry"].min()
            benchmark = self.portfolio.equal_weight_buy_and_hold(
                first_entry, config.stages[stage][1]
            )
            benchmarks[stage] = benchmark
            active = active_period_statistics(strategy, self.market.effr)
            strategy_stats = active["绩效"]
            benchmark_stats = benchmark["statistics"]
            benchmark_rows.extend([
                {"阶段": stage, "方案": "Validation选定策略",
                 "首笔成交日": active["起始成交日"],
                 "末日": strategy["daily"].index[-1],
                 "扣费后累计收益": strategy_stats["累计收益"],
                 "扣费后几何年化收益": strategy_stats["几何年化收益"],
                 "扣费后最大回撤": strategy_stats["最大回撤"],
                 "缺价估值日数": 0},
                {"阶段": stage, "方案": "３７只ETF等权买入并持有",
                 "首笔成交日": benchmark["entry"],
                 "末日": benchmark["exit"],
                 "扣费后累计收益": benchmark_stats["累计收益"],
                 "扣费后几何年化收益": benchmark_stats["几何年化收益"],
                 "扣费后最大回撤": benchmark_stats["最大回撤"],
                 "缺价估值日数": benchmark_stats["缺价日数"]},
            ])
        benchmark_summary = pd.DataFrame(benchmark_rows)
        benchmark_summary.to_csv(config.results_dir / "三阶段策略与等权买入并持有.csv", index=False)
        joblib.dump({"selected_model": selected, "selected_variant": selected_variant,
                     "feature_choice": feature_choice, "choices": choices,
                     "joint": joint, "three_stage": three_stage,
                     "risk_grid": risk_grid, "risk_choice": risk_choice,
                     "covariance_comparison": covariance_comparison,
                     "risk_summary": risk_summary,
                     "selected_risk_summary": selected_risk_summary,
                     "benchmark_summary": benchmark_summary},
                    config.results_dir / "研究摘要.joblib")
        return {"selected_model": selected, "selected_variant": selected_variant,
                "feature_choice": feature_choice,
                "choices": choices, "joint": joint,
                "three_stage": three_stage, "risk_grid": risk_grid,
                 "covariance_comparison": covariance_comparison,
                 "covariance_windows": covariance_windows,
                 "covariance_accounts": covariance_accounts,
                "risk_choice": risk_choice, "risk_summary": risk_summary,
                 "selected_risk_summary": selected_risk_summary,
                 "risk_accounts": risk_accounts,
                 "benchmarks": benchmarks,
                 "benchmark_summary": benchmark_summary}
