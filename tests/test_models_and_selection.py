import numpy as np
import pandas as pd
from types import SimpleNamespace

from cta_project.config import FEATURES
from cta_project.covariance import rolling_oos_covariances
from cta_project.models import RollingResult, prepared_pair
from cta_project.pipeline import combine_monthly_results
import cta_project.pipeline as pipeline_module
from cta_project.selection import choose_portfolio_params, compare_models


def test_standardization_uses_only_train():
    columns = list(FEATURES)
    train = pd.DataFrame(np.arange(18).reshape(3, 6), columns=columns,
                         dtype=float)
    train["target"] = [0.01, 0.02, 0.03]
    apply = train.iloc[[0]].copy()
    apply[columns] = 10_000
    x, y, z, target_mean, target_std = prepared_pair(train, apply)
    assert np.allclose(x.mean().to_numpy(), 0)
    assert (z.to_numpy() > 1_000).all()
    assert target_mean == 0.02
    assert np.isclose(y.mean(), 0)


def test_selection_applies_prediction_gate_before_return(config):
    model = pd.DataFrame({"模型": ["Ridge", "XGBoost"],
                          "MSE": [0.001, 0.002], "MAE": [0.02, 0.03],
                          "预测误差门槛合格": [True, False]})
    base = {"扣费后最大回撤": -0.10, "平均每日换手": 0.10,
            "总费用美元": 100}
    grids = {
        "Ridge": pd.DataFrame([{**base, "lambda": 0.005, "gamma": 0.0025,
                                "扣费后几何年化收益": 0.20},
                               {**base, "lambda": 0.005, "gamma": 0.0,
                                "平均每日换手": 0.15,
                                "扣费后几何年化收益": 0.198}]),
        "XGBoost": pd.DataFrame([{**base, "lambda": 0.005, "gamma": 0.0,
                                  "扣费后几何年化收益": 0.50}]),
    }
    selected, choices, joint = choose_portfolio_params(grids, model, config)
    assert selected == "Ridge"
    assert choices.set_index("模型").loc["Ridge", "gamma"] == 0.0025
    assert joint.set_index("模型").loc["XGBoost", "综合门槛合格"] == False


def test_monthly_result_aggregation_preserves_order():
    months = pd.to_datetime(["2024-06-01", "2024-07-01"], utc=True)
    parts = []
    for number, month in enumerate(months):
        row = pd.DataFrame({"月份": [month], "模型": ["Ridge"]})
        parts.append(RollingResult(
            {"Ridge": pd.Series([float(number)], index=pd.Index([month]))},
            {("Ridge", month): np.eye(2)}, row, row, row,
        ))
    joined = combine_monthly_results(parts)
    assert joined.predictions["Ridge"].tolist() == [0.0, 1.0]
    assert joined.folds["月份"].tolist() == list(months)
    assert len(joined.covariances) == 2


def test_incomplete_portfolio_grid_batch_is_recomputed(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline_module, "MODEL_NAMES", ("Ridge",))
    monkeypatch.setattr(
        pipeline_module, "evaluate_portfolio_point",
        lambda name, lam, gam, *_: {
            "模型": name, "lambda": lam, "gamma": gam,
            "扣费后几何年化收益": 0.02,
            "扣费后最大回撤": -0.03,
            "平均每日换手": 0.1,
            "总费用美元": 1.0,
        },
    )
    folder = tmp_path / "portfolio_grids"
    folder.mkdir()
    (folder / "Ridge_batch_1.csv").write_text("模型,lambda\nRidge,0\n", encoding="utf-8")
    (folder / "Ridge_14x14.csv").write_text("模型,lambda\nRidge,0\n", encoding="utf-8")
    project = object.__new__(pipeline_module.ResearchProject)
    project.config = SimpleNamespace(
        results_dir=tmp_path,
        stages={"Validation": (pd.Timestamp("2024-06-01", tz="UTC"),
                                pd.Timestamp("2025-06-01", tz="UTC"))},
    )
    project.portfolio = object()
    grid = project.portfolio_grid(None)["Ridge"]
    assert len(grid) == 196
    assert len(pd.read_csv(folder / "Ridge_batch_1.csv")) == 28
    assert len(pd.read_csv(folder / "Ridge_14x14.csv")) == 196


def test_validation_errors_exclude_unmatured_labels(config, market):
    end = config.stages["Validation"][1]
    mature = market.calendar[(market.calendar < end)
                             & market.label_end.lt(end).to_numpy()][-2:]
    immature = market.calendar[(market.calendar < end)
                               & market.label_end.ge(end).to_numpy()][0]
    index = pd.MultiIndex.from_tuples(
        [(date, "SPY") for date in (*mature, immature)], names=["date", "symbol"]
    )
    truth = market.target.stack().reindex(index)
    prediction = truth.copy()
    prediction.loc[(immature, "SPY")] = 100.0
    result = RollingResult({"Ridge": prediction}, {}, pd.DataFrame(),
                           pd.DataFrame(), pd.DataFrame())
    compared = compare_models(result, market, config)
    assert compared.loc[0, "样本数"] == 2
    assert compared.loc[0, "MSE"] == 0


def test_volatility_state_variants_keep_same_population(config, market):
    from dataclasses import replace
    from cta_project.signals import SignalEngine

    spec = ("EWMA", 0.2, 10, 1, 7)
    baseline = SignalEngine(market, config).feature_panel(spec)
    level = SignalEngine(market, replace(config, vol_state_variant="level")).feature_panel(spec)
    assert baseline.index.equals(level.index)
    date = pd.Timestamp("2024-06-03", tz="UTC")
    assert level.loc[(date, "SPY"), "VOL_STATE"] > baseline.loc[(date, "SPY"), "VOL_STATE"]


def test_rolling_oos_covariance_uses_only_matured_past_errors(config, market):
    month = pd.Timestamp("2024-08-01", tz="UTC")
    dates = market.calendar[(market.calendar >= pd.Timestamp("2024-06-01", tz="UTC"))
                            & (market.calendar < month)]
    truth = market.target.loc[dates].stack()
    predicted = truth + 0.001 * np.sin(np.arange(len(truth)))
    predicted.index.names = ["date", "symbol"]
    covariances, details = rolling_oos_covariances(
        market, config, predicted, pd.DatetimeIndex([month]), "Ridge"
    )
    assert ("Ridge", month) in covariances
    assert details.loc[0, "最晚成熟标签"] < month
    assert details.loc[0, "完整残差日期数"] >= 20
    assert np.isfinite(np.diag(covariances["Ridge", month])).sum() >= 20
