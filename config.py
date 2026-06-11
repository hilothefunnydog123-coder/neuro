"""Configuration loading and the typed config dataclasses.

Values come from ``config.yaml`` by default and can be overridden by keyword
arguments (typically parsed from the command line in ``cli.py``).
"""

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, List, Optional

import yaml


@dataclass
class DataConfig:
    ticker: str = "AAPL"
    start: str = "2015-01-01"
    end: Optional[str] = None
    interval: str = "1d"
    csv: Optional[str] = None  # load from a local CSV instead of downloading
    features: List[str] = field(default_factory=lambda: ["Open", "High", "Low", "Close", "Volume"])
    target: str = "Close"
    # How the model predicts the target:
    #   "logreturn" -> predict day-to-day log returns, rebuild price from the
    #                  last actual price (stationary, recommended).
    #   "price"     -> predict the raw price level (prone to mean-reverting to
    #                  the historical average; only for short/stationary series).
    target_mode: str = "logreturn"


@dataclass
class WindowConfig:
    lookback: int = 60
    horizon: int = 5


@dataclass
class ModelConfig:
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    val_split: float = 0.15
    patience: int = 10
    seed: int = 42
    device: str = "auto"
    resume: bool = False  # continue from existing checkpoint instead of fresh init


@dataclass
class PathsConfig:
    checkpoint: str = "checkpoints/model.pt"
    output_dir: str = "outputs"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)


def _build(cls, data: dict) -> Any:
    """Recursively build a (possibly nested) dataclass from a plain dict."""
    if not is_dataclass(cls):
        return data
    kwargs = {}
    for f in fields(cls):
        if data and f.name in data:
            value = data[f.name]
            if is_dataclass(f.type) or (isinstance(f.type, type) and is_dataclass(f.type)):
                kwargs[f.name] = _build(f.type, value)
            else:
                kwargs[f.name] = value
    return cls(**kwargs)


def load_config(path: str = "config.yaml") -> Config:
    """Load configuration from a YAML file, falling back to defaults."""
    try:
        with open(path, "r") as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raw = {}
    return _build(Config, raw)


def apply_overrides(cfg: Config, overrides: dict) -> Config:
    """Apply a flat dict of overrides (e.g. {"ticker": "MSFT", "epochs": 10}).

    Each key is matched against the fields of every sub-config; ``None`` values
    are ignored so unset CLI flags don't clobber file/default values.
    """
    sections = [cfg.data, cfg.window, cfg.model, cfg.train, cfg.paths]
    for key, value in overrides.items():
        if value is None:
            continue
        for section in sections:
            if hasattr(section, key):
                setattr(section, key, value)
                break
    return cfg
