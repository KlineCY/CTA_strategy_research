"""七模型的逐月拟合、局部选参和残差协方差。"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from .config import Config, FEATURES, MODEL_GRID, MODEL_NAMES
from .data import Market
from .signals import SignalEngine, period


@dataclass
class RollingResult:
    predictions: dict[str, pd.Series]
    covariances: dict[tuple[str, pd.Timestamp], np.ndarray]
    folds: pd.DataFrame
    searches: pd.DataFrame
    importances: pd.DataFrame
    skipped: pd.DataFrame = field(default_factory=pd.DataFrame)


def model_instance(name: str, parameter):
    if name == "Equal Weights":
        return LinearRegression()
    if name == "LASSO":
        return Lasso(alpha=parameter, max_iter=50_000)
    if name == "Ridge":
        return Ridge(alpha=parameter)
    if name == "Elastic Net":
        return ElasticNet(alpha=parameter[0], l1_ratio=parameter[1], max_iter=50_000)
    if name == "Decision Tree":
        return DecisionTreeRegressor(max_depth=parameter[0],
                                     min_samples_leaf=parameter[1],
                                     min_samples_split=parameter[2], random_state=0)
    if name == "Random Forest":
        return RandomForestRegressor(n_estimators=parameter[0],
                                     max_depth=parameter[1],
                                     min_samples_leaf=parameter[2],
                                     max_features=parameter[3],
                                     max_samples=parameter[4], n_jobs=1,
                                     random_state=0)
    if name == "XGBoost":
        return XGBRegressor(n_estimators=parameter[0], max_depth=parameter[1],
                            learning_rate=parameter[2], min_child_weight=parameter[3],
                            subsample=parameter[4], colsample_bytree=parameter[5],
                            reg_lambda=parameter[6], objective="reg:squarederror",
                            tree_method="hist", n_jobs=1, random_state=0)
    raise ValueError(name)


def matrix_for(name: str, frame: pd.DataFrame) -> np.ndarray:
    matrix = frame[list(FEATURES)].to_numpy()
    return matrix.mean(axis=1).reshape(-1, 1) if name == "Equal Weights" else matrix


def metrics(actual: pd.Series, prediction: pd.Series) -> dict:
    valid = pd.concat([actual.rename("y"), prediction.rename("p")], axis=1).dropna()
    return {"MSE": mean_squared_error(valid["y"], valid["p"]),
            "MAE": mean_absolute_error(valid["y"], valid["p"]),
            "R2": r2_score(valid["y"], valid["p"]), "样本数": len(valid)}


def prepared_pair(train: pd.DataFrame, apply: pd.DataFrame):
    mean = train[list(FEATURES)].mean()
    std = train[list(FEATURES)].std().replace(0, 1)
    target_mean, target_std = train["target"].mean(), train["target"].std()
    if not (target_std > 0 and std.notna().all()):
        raise ValueError("训练窗的目标或特征标准差无效。")
    x = (train[list(FEATURES)] - mean) / std
    z = (apply[list(FEATURES)] - mean) / std
    y = (train["target"] - target_mean) / target_std
    return x, y, z, target_mean, target_std


def _score_candidate(name, parameter, x_fit, y_fit, x_check, y_check,
                     target_mean, target_std):
    estimator = model_instance(name, parameter)
    estimator.fit(matrix_for(name, x_fit), y_fit)
    prediction = estimator.predict(matrix_for(name, x_check)) * target_std + target_mean
    return {"parameter": parameter,
            "MSE": mean_squared_error(y_check, prediction),
            "MAE": mean_absolute_error(y_check, prediction)}


def _candidate_results(name, parameters, inputs, n_jobs):
    if name in ("Random Forest", "XGBoost") and n_jobs > 1:
        return Parallel(n_jobs=n_jobs, prefer="processes")(
            delayed(_score_candidate)(name, parameter, *inputs)
            for parameter in parameters
        )
    return [_score_candidate(name, parameter, *inputs) for parameter in parameters]


def _refined_parameters(name, coarse_results):
    if name not in ("Random Forest", "XGBoost"):
        return []
    top = sorted(coarse_results, key=lambda row: (row["MSE"], row["MAE"]))[:4]
    if name == "Random Forest":
        return list(dict.fromkeys(
            (trees, *row["parameter"][1:4], sample)
            for row in top for trees in (200, 400) for sample in (0.7, 1.0)
        ))
    return list(dict.fromkeys(
        (trees, *row["parameter"][1:6], penalty)
        for row in top for trees in (100, 400) for penalty in (0.1, 10.0)
    ))


def rolling_model_families(market: Market, signals: SignalEngine, config: Config,
                           start: pd.Timestamp, end: pd.Timestamp,
                           names: tuple[str, ...] = MODEL_NAMES,
                           n_jobs: int = 1,
                           grids: dict | None = None) -> RollingResult:
    """每月使用前三个月拟合、前一个月局部验证，并预测下一个月。"""
    grids = MODEL_GRID if grids is None else grids
    pieces = {name: [] for name in names}
    covariances, folds, searches, importances, skipped = {}, [], [], [], []
    for month in pd.date_range(start, end, freq="MS", inclusive="left"):
        fit_start = month - pd.DateOffset(months=4)
        local_start = month - pd.DateOffset(months=1)
        spec, _, _ = signals.select_macd(local_start, month,
                                         execution_delay=config.delay)
        panel = signals.feature_panel(tuple(spec))
        train = period(panel, fit_start, local_start, maturity=local_start)
        local = period(panel, local_start, month, maturity=month)
        available = period(panel, fit_start, month, maturity=month)
        apply = period(panel, month, min(month + pd.DateOffset(months=1), end))
        covariance_input = available["target"].unstack("symbol").reindex(
            columns=market.symbols
        )
        covariance_symbols = list(market.symbols)
        while len(covariance_symbols) >= 20:
            complete_residual_dates = covariance_input[covariance_symbols].dropna()
            if len(complete_residual_dates) >= 20:
                break
            missing = covariance_input[covariance_symbols].isna().sum()
            if missing.max() == 0:
                break
            covariance_symbols.remove(missing.idxmax())
        if (min(len(train), len(local), len(apply)) <= 0
                or len(covariance_symbols) < 20
                or len(complete_residual_dates) < 20):
            skipped.append({"月份": month, "原因": "行情缺口影响特征、标签或残差协方差窗口",
                            "拟合样本数": len(train), "局部样本数": len(local),
                            "预测样本数": len(apply),
                            "完整残差日期数": len(complete_residual_dates)})
            continue
        if not available["label_end"].max() < apply.index.get_level_values("date").min():
            raise ValueError("训练标签越过预测月份。")
        x_fit, y_fit, x_local, y_mean, y_std = prepared_pair(train, local)
        x_refit, y_refit, x_apply, refit_mean, refit_std = prepared_pair(available, apply)
        for name in names:
            inputs = (x_fit, y_fit, x_local, local["target"], y_mean, y_std)
            scores = _candidate_results(name, grids[name], inputs, n_jobs)
            scores += _candidate_results(name, _refined_parameters(name, scores), inputs, n_jobs)
            for row in scores:
                searches.append({"月份": month, "模型": name,
                                 "参数": repr(row["parameter"]),
                                 "局部MSE": row["MSE"], "局部MAE": row["MAE"]})
            chosen = min(scores, key=lambda row: (row["MSE"], row["MAE"]))
            estimator = model_instance(name, chosen["parameter"])
            estimator.fit(matrix_for(name, x_refit), y_refit)
            prediction = pd.Series(
                estimator.predict(matrix_for(name, x_apply)) * refit_std + refit_mean,
                index=apply.index, name=name,
            )
            pieces[name].append(prediction)
            fitted = estimator.predict(matrix_for(name, x_refit)) * refit_std + refit_mean
            error = available["target"] - fitted
            error_matrix = error.unstack("symbol").reindex(
                columns=covariance_symbols
            ).dropna()
            if len(error_matrix) < 20:
                raise ValueError("残差样本数不足以估计协方差。")
            sigma = np.full((len(market.symbols), len(market.symbols)), np.nan)
            positions = market.symbols.get_indexer(covariance_symbols)
            sigma[np.ix_(positions, positions)] = (
                LedoitWolf().fit(error_matrix.to_numpy()).covariance_ * config.scale**2
            )
            covariances[name, month] = sigma
            values = getattr(estimator, "coef_",
                             getattr(estimator, "feature_importances_", None))
            if values is not None:
                coefficients = (np.repeat(values[0] / len(FEATURES), len(FEATURES))
                                if name == "Equal Weights" else values)
                importances.append({"月份": month, "模型": name,
                                    **dict(zip(FEATURES, coefficients, strict=True))})
            folds.append({"月份": month, "模型": name,
                          "所选参数": repr(chosen["parameter"]),
                          "局部MSE": chosen["MSE"], "局部MAE": chosen["MAE"],
                          "MACD参数": repr(tuple(spec)),
                          "拟合样本数": len(train), "局部验证样本数": len(local),
                          "重新拟合样本数": len(available), "预测样本数": len(apply),
                          "协方差资产数": len(covariance_symbols),
                          "最晚成熟标签": available["label_end"].max()})
    empty = pd.MultiIndex.from_arrays([[], []], names=["date", "symbol"])
    prediction = {name: pd.concat(series).sort_index() if series
                  else pd.Series(dtype=float, index=empty, name=name)
                  for name, series in pieces.items()}
    return RollingResult(prediction, covariances, pd.DataFrame(folds),
                         pd.DataFrame(searches), pd.DataFrame(importances),
                         pd.DataFrame(skipped))
