"""本地命令行入口。"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config, MODEL_GRID
from .models import rolling_model_families
from .pipeline import ResearchProject
from .reporting import save_figures
from .storage import input_manifest


def smoke(project: ResearchProject):
    config = project.config
    start, _ = config.stages["Validation"]
    end = start + pd.DateOffset(months=1)
    result = rolling_model_families(
        project.market, project.signals, config, start, end,
        names=("Ridge",), grids={"Ridge": MODEL_GRID["Ridge"]}, n_jobs=1,
    )
    actual = result.predictions["Ridge"]
    if actual.empty or not np.isfinite(actual.to_numpy()).all():
        raise AssertionError("样本月预测为空或含非有限值。")
    month = result.folds.iloc[0]["月份"]
    covariance = result.covariances["Ridge", month]
    if (covariance.shape != (len(project.market.symbols), len(project.market.symbols))
            or not np.isfinite(covariance).all()
            or not np.allclose(covariance, covariance.T, atol=1e-8)):
        raise AssertionError("样本月残差协方差无效。")
    account = project.portfolio.optimized_account(
        actual, result.covariances, start, end, 0.005, 0.0025, "Ridge"
    )
    daily = account["daily"]
    if (daily.empty
            or account["records"]["constraint_error"].max() > config.tolerance
            or not np.allclose(daily["equity"], daily["cash"] + daily["asset_value"])
            or not np.isclose(daily["fee"].sum(),
                              account["commission"].sum()
                              + account["section31"].sum())):
        raise AssertionError("目标约束、账户恒等式或成交费核对失败。")
    decisions = project.risk.prepare_decisions(account)
    controlled = project.risk.controlled_account(account, decisions, None, 1.0)
    if not np.allclose(controlled["daily"]["equity"], daily["equity"], atol=1e-6):
        raise AssertionError("执行比例为１时，账户净值不一致。")
    return {"样本预测月": "2024-06", "模型": "Ridge",
            "局部参数": result.folds.iloc[0]["所选参数"],
            "MACD参数": result.folds.iloc[0]["MACD参数"],
            "预测条数": len(actual),
            "协方差维度": covariance.shape[0],
            "最大约束误差": float(account["records"]["constraint_error"].max()),
            "样本月决策日数": len(account["weights"]),
            "样本月累计费用美元": float(daily["fee"].sum()),
            "样本月账户末值美元": float(daily["equity"].iloc[-1])}


def main(argv=None):
    parser = argparse.ArgumentParser(description="37只ETF的CTA研究工程")
    parser.add_argument("command", choices=("validate-data", "smoke", "run"))
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    config = Config.in_project(args.root)
    if args.command == "validate-data":
        project = ResearchProject(config)
        manifest = input_manifest(config)
        print("ETF数量：", len(project.market.symbols))
        print("市场交易日数：", len(project.market.calendar))
        print("行情不全的交易日数：", int(project.market.close.isna().any(axis=1).sum()))
        print("输入文件数：", len(manifest["inputs"]))
        print("输入及代码指纹：", manifest["fingerprint"])
        return 0
    project = ResearchProject(config)
    if args.command == "smoke":
        print(smoke(project))
        return 0
    if args.workers < 1:
        parser.error("--workers须为正整数。")
    result = project.run_full(workers=args.workers)
    save_figures(config, result)
    print("Validation最终模型：", result["selected_model"])
    print("全部结果：", config.results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
