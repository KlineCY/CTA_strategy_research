"""使用既往逐月样本外预测误差估计候选残差协方差。"""

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from .config import Config
from .data import Market


def rolling_oos_covariances(market: Market, config: Config,
                            prediction: pd.Series,
                            months: pd.DatetimeIndex,
                            model_name: str):
    """每月仅使用此前四个月且标签已成熟的历史预测残差。"""
    if not prediction.index.is_unique:
        raise ValueError("逐月样本外预测存在重复的资产与日期。")
    dates = prediction.index.get_level_values("date")
    labels = market.label_end.reindex(dates)
    errors = market.target.stack().reindex(prediction.index) - prediction
    results, details = {}, []
    for month in months:
        eligible = ((dates >= month - pd.DateOffset(months=4))
                    & (dates < month)
                    & labels.lt(month).to_numpy())
        observed = errors.loc[eligible]
        matrix = observed.unstack("symbol").reindex(columns=market.symbols)
        symbols = list(market.symbols)
        complete = pd.DataFrame()
        while len(symbols) >= 20:
            complete = matrix[symbols].dropna()
            if len(complete) >= 20:
                break
            missing = matrix[symbols].isna().sum()
            if missing.empty or missing.max() == 0:
                break
            symbols.remove(missing.idxmax())
        if len(symbols) < 20 or len(complete) < 20:
            details.append({"月份": month, "模型": model_name,
                            "协方差资产数": 0, "完整残差日期数": len(complete),
                            "最晚成熟标签": pd.NaT})
            continue
        sigma = np.full((len(market.symbols), len(market.symbols)), np.nan)
        positions = market.symbols.get_indexer(symbols)
        sigma[np.ix_(positions, positions)] = (
            LedoitWolf().fit(complete.to_numpy()).covariance_ * config.scale**2
        )
        mature = labels.loc[eligible].max()
        if not mature < month:
            raise ValueError("样本外残差协方差使用了未成熟标签。")
        results[model_name, month] = sigma
        details.append({"月份": month, "模型": model_name,
                        "协方差资产数": len(symbols),
                        "完整残差日期数": len(complete),
                        "最晚成熟标签": mature})
    return results, pd.DataFrame(details)
