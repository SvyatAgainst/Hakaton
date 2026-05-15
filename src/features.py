from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import MarketData


@dataclass(frozen=True)
class FeatureSet:
    """Prepared features and target used by signal and backtest modules."""

    mid: pd.DataFrame
    spread: pd.DataFrame
    spread_rel: pd.DataFrame
    trade_imbalance: pd.DataFrame
    book_imbalance: pd.DataFrame
    volatility: pd.DataFrame
    liquidity: pd.DataFrame
    target_return: pd.DataFrame
    score: pd.DataFrame


class FeatureBuilder:
    """Builds interpretable microstructure features from wide market tables."""

    def __init__(self, horizon_bars: int = 30, volatility_window: int = 30) -> None:
        self.horizon_bars = horizon_bars
        self.volatility_window = volatility_window

    @staticmethod
    def _safe_divide(numerator: pd.DataFrame, denominator: pd.DataFrame) -> pd.DataFrame:
        denominator = denominator.replace(0, np.nan)
        return numerator / denominator

    def build(self, data: MarketData) -> FeatureSet:
        mid = (data.best_bid + data.best_ask) / 2
        spread = data.best_ask - data.best_bid
        spread_rel = self._safe_divide(spread, mid)

        trade_imbalance = self._safe_divide(
            data.buy_size - data.sell_size,
            data.buy_size + data.sell_size,
        )
        book_imbalance = self._safe_divide(
            data.best_bid_size - data.best_ask_size,
            data.best_bid_size + data.best_ask_size,
        )

        returns = mid.pct_change(fill_method=None)
        volatility = returns.rolling(self.volatility_window, min_periods=5).std()
        liquidity = (data.best_bid_size + data.best_ask_size + data.buy_size + data.sell_size).replace(0, np.nan)
        target_return = mid.shift(-self.horizon_bars) / mid - 1

        score = (
            0.45 * trade_imbalance.fillna(0)
            + 0.35 * book_imbalance.fillna(0)
            - 0.10 * spread_rel.rank(axis=1, pct=True).fillna(0.5)
            - 0.10 * volatility.rank(axis=1, pct=True).fillna(0.5)
        )

        return FeatureSet(
            mid=mid,
            spread=spread,
            spread_rel=spread_rel,
            trade_imbalance=trade_imbalance,
            book_imbalance=book_imbalance,
            volatility=volatility,
            liquidity=liquidity,
            target_return=target_return,
            score=score,
        )

