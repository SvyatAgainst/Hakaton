from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


def _patch_pyarrow_extension_unregister() -> None:
    """Ignore a pyarrow 24 notebook-state bug when an extension type is already absent."""
    try:
        import pyarrow as pa
    except ImportError:
        return

    if getattr(pa, "_hackathon_unregister_patch", False):
        return

    original_unregister = pa.unregister_extension_type

    def safe_unregister(name: str) -> None:
        try:
            original_unregister(name)
        except pa.ArrowKeyError:
            if name != "arrow.py_extension_type":
                raise

    pa.unregister_extension_type = safe_unregister
    pa._hackathon_unregister_patch = True


@dataclass(frozen=True)
class MarketData:
    """Container for wide market feature tables: index is time, columns are tickers."""

    open: pd.DataFrame
    close: pd.DataFrame
    best_bid: pd.DataFrame
    best_ask: pd.DataFrame
    best_bid_size: pd.DataFrame
    best_ask_size: pd.DataFrame
    buy_size: pd.DataFrame
    sell_size: pd.DataFrame
    pr_std: pd.DataFrame


class ParquetFeatureLoader:
    """Loads already aggregated hackathon parquet features."""

    REQUIRED_FILES = {
        "open": "open_.parquet",
        "close": "close_.parquet",
        "best_bid": "best_bid_.parquet",
        "best_ask": "best_ask_.parquet",
        "best_bid_size": "best_bid_size_.parquet",
        "best_ask_size": "best_ask_size_.parquet",
        "buy_size": "buy_size_.parquet",
        "sell_size": "sell_size_.parquet",
        "pr_std": "pr_std_.parquet",
    }

    def __init__(self, data_root: str | Path = "insample_data", timeframe: str = "1min") -> None:
        self.data_root = Path(data_root)
        self.timeframe = timeframe

    @property
    def folder(self) -> Path:
        if self.timeframe in {"1min", "1_min", "1"}:
            return self.data_root / "is_features_1_min_hackaton"
        if self.timeframe in {"5min", "5_min", "5"}:
            return self.data_root / "is_features_5_min_hackaton"
        raise ValueError("timeframe must be one of: '1min', '5min'")

    def read_feature(self, filename: str) -> pd.DataFrame:
        path = self.folder / filename
        if not path.exists():
            raise FileNotFoundError(f"Feature file not found: {path}")
        _patch_pyarrow_extension_unregister()
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        return df.sort_index()

    def load(self) -> MarketData:
        loaded = {
            name: self.read_feature(filename)
            for name, filename in self.REQUIRED_FILES.items()
        }
        return MarketData(**loaded)

