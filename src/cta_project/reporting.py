"""研究结果图；颜色与线型适配深色编辑器。"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import Config, MODEL_NAMES


COLORS = ("#6BBBD3", "#E2BE70", "#C98FBE", "#A1C889",
          "#DEA476", "#B4A5D7", "#D88F8C")


def _style():
    plt.style.use("dark_background")
    plt.rcParams.update({"font.family": ["Times New Roman", "SimSun"],
                         "font.size": 12, "axes.unicode_minus": False,
                         "figure.facecolor": "#15191F", "axes.facecolor": "#15191F",
                         "text.color": "#E7EAF0", "axes.labelcolor": "#E7EAF0",
                         "axes.edgecolor": "#AEB7C2", "xtick.color": "#D5DAE1",
                         "ytick.color": "#D5DAE1", "grid.color": "#74808D",
                         "legend.facecolor": "#1D232B", "legend.edgecolor": "#AEB7C2",
                         "legend.labelcolor": "#E7EAF0"})


def save_figures(config: Config, summary: dict):
    _style()
    folder = config.results_dir / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    comparison = summary["joint"]
    fig, axis = plt.subplots(figsize=(11, 5.5))
    for number, row in comparison.reset_index(drop=True).iterrows():
        axis.scatter(row["MSE"], row["MAE"], s=95,
                     color=COLORS[number],
                     label=("Equal Weights（等信号）" if row["模型"] == "Equal Weights"
                            else row["模型"]))
    axis.set(xlabel="Validation均方误差MSE", ylabel="Validation平均绝对误差MAE",
             title="七模型｜整体验证期预测误差")
    axis.grid(alpha=0.25)
    axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5))
    fig.tight_layout()
    fig.savefig(folder / "validation_model_errors.png", dpi=170)
    plt.close(fig)

    for stage in ("Training", "Validation", "历史Test"):
        fig, axis = plt.subplots(figsize=(13, 5.8))
        for number, name in enumerate(MODEL_NAMES):
            row = summary["choices"].set_index("模型").loc[name]
            lam, gam = float(row["lambda"]), float(row["gamma"])
            path = (config.results_dir / "cache"
                    / f"account_{stage}_{name.replace(' ', '_')}_{lam:.10g}_{gam:.10g}.joblib")
            import joblib
            _metrics, _account, daily = joblib.load(path)
            first_entry = _account["records"]["entry"].min()
            curve = daily.loc[first_entry:, "equity"] / config.capital - 1
            axis.plot(curve.index, curve, color=COLORS[number], linewidth=1.9,
                      linestyle="-",
                      label="Equal Weights（等信号）" if name == "Equal Weights" else name)
        axis.axhline(0, color="#DDDDDD", linewidth=0.8)
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        axis.set(xlabel="日期", ylabel="扣费后累计收益",
                  title=stage + ("事后参数诊断" if stage == "Training" else "")
                  + "｜首笔成交日起七模型累计收益")
        axis.grid(alpha=0.25)
        axis.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=10)
        fig.tight_layout()
        fig.savefig(folder / f"{stage}_seven_models.png", dpi=170)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(12, 5.5))
        selected_label = ("基础组合" if summary["risk_choice"]["方案"] == "基础组合"
                          else "状态控制候选")
        for label, color in (("基础组合", COLORS[1]),
                             ("状态控制候选", COLORS[0])):
            account = summary["risk_accounts"][stage, label]
            first_entry = account["records"]["entry"].min()
            curve = account["net_daily"].loc[first_entry:, "equity"] / config.capital - 1
            display_label = ("Validation选定｜" + label
                             if label == selected_label else "诊断对照｜" + label)
            axis.plot(curve.index, curve, color=color,
                      linewidth=2.5 if label == selected_label else 1.7,
                      linestyle="-", label=display_label)
        axis.axhline(0, color="#FFFFFF", linewidth=0.8)
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        axis.set(xlabel="日期", ylabel="扣费后累计收益",
                  title=stage + "｜首笔成交日起组合风险控制")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(folder / f"{stage}_risk_control.png", dpi=170)
        plt.close(fig)

        fig, axis = plt.subplots(figsize=(12, 5.5))
        selected = summary["risk_accounts"][stage, "最终选定方案"]
        benchmark = summary["benchmarks"][stage]
        first_entry = selected["records"]["entry"].min()
        selected_curve = selected["net_daily"].loc[first_entry:, "equity"] / config.capital - 1
        benchmark_curve = benchmark["equity"].loc[first_entry:] / config.capital - 1
        axis.plot(selected_curve.index, selected_curve, color=COLORS[0],
                  linewidth=2.2, linestyle="-", label="Validation选定策略")
        axis.plot(benchmark_curve.index, benchmark_curve, color=COLORS[1],
                  linewidth=2.0, linestyle="-", label="３７只ETF等权买入并持有")
        axis.axhline(0, color="#DDDDDD", linewidth=0.8)
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        axis.set(xlabel="日期", ylabel="扣费后累计收益",
                 title=stage + "｜首笔成交日起策略与等权买入并持有")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(folder / f"{stage}_buy_hold_benchmark.png", dpi=170)
        plt.close(fig)

    fig, axis = plt.subplots(figsize=(10, 6))
    grid = summary["risk_grid"]
    for label, color in (("基础组合", COLORS[1]),
                         ("状态控制组合", COLORS[0])):
        part = grid.loc[grid["方案"].eq(label)]
        axis.scatter(-part["扣费后最大回撤"], part["扣费后几何年化收益"],
                     s=85 if label == "基础组合" else 48,
                     color=color, label=label)
    axis.axvline(config.max_drawdown_magnitude, color="#FFFFFF", linewidth=1)
    axis.xaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
    axis.set(xlabel="Validation最大回撤幅度",
             ylabel="Validation扣费后几何年化收益",
             title="组合风险控制｜Validation收益与回撤")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(folder / "validation_risk_grid.png", dpi=170)
    plt.close(fig)
    covariance_accounts = summary.get("covariance_accounts", {})
    if "既往逐月样本外残差" in covariance_accounts:
        fig, axis = plt.subplots(figsize=(12, 5.5))
        for label, color in (("训练窗拟合残差", COLORS[1]),
                             ("既往逐月样本外残差", COLORS[0])):
            account = covariance_accounts[label]
            curve = account["daily"]["equity"] / config.capital - 1
            axis.plot(curve.index, curve, color=color, linewidth=2.0,
                      linestyle="-", label=label)
        axis.axhline(0, color="#DDDDDD", linewidth=0.8)
        axis.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
        axis.set(xlabel="日期", ylabel="扣费后累计收益",
                 title="Validation｜同一模型和组合参数的残差协方差对照")
        axis.grid(alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(folder / "validation_covariance_candidate.png", dpi=170)
        plt.close(fig)
    return folder
