from pathlib import Path

import pytest

from cta_project.config import Config
from cta_project.data import load_market


@pytest.fixture(scope="session")
def config():
    return Config.in_project(Path(__file__).resolve().parents[1])


@pytest.fixture(scope="session")
def market(config):
    if not list(config.data_dir.glob('*_ohlcv_1d.csv')):
        pytest.skip('请先在data/raw目录放置具备使用权的研究数据。')
    return load_market(config)
