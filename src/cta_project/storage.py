"""输入与代码指纹，以及可安全恢复的实验缓存。"""

from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

import joblib

from .config import Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def input_manifest(config: Config) -> dict:
    raw = sorted(path for path in config.data_dir.iterdir() if path.is_file())
    source = sorted(Path(__file__).parent.glob("*.py"))
    data = {path.name: {"sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in raw}
    code = {path.name: sha256_file(path) for path in source}
    code["pyproject.toml"] = sha256_file(config.root / "pyproject.toml")
    environment = {"python": sys.version.split()[0],
                   "dependencies": {name: version(name) for name in (
                       "numpy", "pandas", "scipy", "scikit-learn", "xgboost",
                       "joblib", "matplotlib", "nbformat")}}
    settings = asdict(config)
    settings["root"] = "."
    payload = {"inputs": data, "source": code, "environment": environment,
               "settings": settings}
    payload["fingerprint"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    return payload


def check_or_create_run(config: Config) -> dict:
    manifest = input_manifest(config)
    config.results_dir.mkdir(parents=True, exist_ok=True)
    path = config.results_dir / "run_manifest.json"
    if path.exists():
        recorded = json.loads(path.read_text(encoding="utf-8"))
        if recorded["fingerprint"] != manifest["fingerprint"]:
            raise RuntimeError("结果目录对应的输入、代码或参数与当前工程不一致。请使用新的结果目录。")
    else:
        path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2,
                                   default=str), encoding="utf-8")
    return manifest


def load_or_compute(path: Path, function):
    if path.exists():
        return joblib.load(path)
    value = function()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)
    return value
