"""研究常量与独立工程的路径配置。"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ("MOM", "MOMRA", "CSMOM", "MACD", "MACD_SMOOTH", "VOL_STATE")
MODEL_NAMES = ("LASSO", "Ridge", "Elastic Net", "Decision Tree",
               "Random Forest", "XGBoost", "Equal Weights")
PARAMETER_VALUES = (0, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.025,
                    0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0)
MODEL_GRID = {
    "Equal Weights": [None],
    "LASSO": [0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0],
    "Ridge": np.logspace(-5, 8, 27).tolist(),
    "Elastic Net": [(alpha, ratio)
                    for alpha in (0.0001, 0.0003, 0.001, 0.003, 0.01,
                                  0.03, 0.1, 0.3, 1.0)
                    for ratio in (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)],
    "Decision Tree": [(depth, leaf, split)
                      for depth in (1, 2, 3, 4, 5, 7, 10, None)
                      for leaf in (2, 5, 10, 20, 40, 80, 160)
                      for split in (2, 20)],
    "Random Forest": [(100, depth, leaf, features, None)
                      for depth in (1, 2, 3, 5, 8, None)
                      for leaf in (2, 5, 10, 20, 40, 80)
                      for features in (0.25, 0.5, 0.75, 1.0)],
    "XGBoost": [(200, depth, rate, child, sample, columns, 1.0)
                for depth in (1, 2, 3, 4)
                for rate in (0.03, 0.1)
                for child in (1, 10)
                for sample in (0.8, 1.0)
                for columns in (0.8, 1.0)],
}


@dataclass(frozen=True)
class Config:
    root: Path
    capital: float = 1_000_000
    mom_window: int = 21
    holding: int = 5
    delay: int = 1
    quantile_window: int = 252
    winsor_window: int = 63
    winsor_limits: tuple[float, float] = (0.01, 0.99)
    net_band: tuple[float, float] = (0.90, 1.00)
    gross_band: tuple[float, float] = (2.80, 3.00)
    asset_bounds: tuple[float, float] = (-0.10, 0.20)
    scale: int = 10_000
    tolerance: float = 1e-5
    solver_maxiter: int = 400
    solver_retry_maxiter: int = 4000
    prediction_error_tolerance: float = 0.005
    max_drawdown_magnitude: float = 0.15
    annual_return_tolerance: float = 0.005
    risk_lookback: int = 63
    high_state_percentile: float = 0.75
    target_volatilities: tuple[float, ...] = (0.04, 0.08, 0.12, 0.16, 0.20)
    high_state_multipliers: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00)
    vol_state_variant: str = "cross_sectional"

    @classmethod
    def in_project(cls, root: Path | None = None) -> "Config":
        return cls((root or Path(__file__).resolve().parents[2]).resolve())

    @property
    def data_dir(self) -> Path:
        return self.root / "data" / "raw"

    @property
    def results_dir(self) -> Path:
        return self.root / "results"

    @property
    def stages(self) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
        dates = pd.to_datetime(
            ["2020-01-01", "2024-06-01", "2025-06-01", "2026-06-01"], utc=True
        )
        return {"Training": (dates[0], dates[1]),
                "Validation": (dates[1], dates[2]),
                "历史Test": (dates[2], dates[3])}
