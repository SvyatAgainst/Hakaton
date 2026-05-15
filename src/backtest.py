from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FeatureSet


@dataclass(frozen=True)
class BacktestResult:
    positions: pd.DataFrame
    turnover: pd.DataFrame
    pnl_mid: pd.Series
    pnl_net: pd.Series
    spread_cost: pd.Series
    impact_cost: pd.Series
    equity_mid: pd.Series
    equity_net: pd.Series
    metrics: pd.Series


class Backtester:
    """Vectorized mid-price backtest with spread and market-impact costs."""

    def __init__(
        self,
        periods_per_year: int = 252 * 390,
        spread_cost_multiplier: float = 0.5,
        impact_alpha: float = 0.0005,
        impact_beta: float = 0.5,
    ) -> None:
        self.periods_per_year = periods_per_year
        self.spread_cost_multiplier = spread_cost_multiplier
        self.impact_alpha = impact_alpha
        self.impact_beta = impact_beta

    def run(self, positions: pd.DataFrame, features: FeatureSet) -> BacktestResult:
        tradable_return = features.target_return.reindex_like(positions).fillna(0.0)

        shifted_positions = positions.shift(1).fillna(0.0)
        pnl_by_asset = shifted_positions * tradable_return
        pnl_mid = pnl_by_asset.sum(axis=1)

        turnover = positions.diff().abs().fillna(positions.abs())
        spread_cost_by_asset = turnover * features.spread_rel.reindex_like(turnover).fillna(0.0) * self.spread_cost_multiplier

        liquidity_rank = features.liquidity.reindex_like(turnover).rank(axis=1, pct=True)
        illiquidity = (1.0 - liquidity_rank).fillna(0.5).clip(lower=0.0, upper=1.0)
        impact_cost_by_asset = self.impact_alpha * (turnover.clip(lower=0.0) ** self.impact_beta) * illiquidity

        spread_cost = spread_cost_by_asset.sum(axis=1)
        impact_cost = impact_cost_by_asset.sum(axis=1)
        pnl_net = pnl_mid - spread_cost - impact_cost

        equity_mid = pnl_mid.cumsum()
        equity_net = pnl_net.cumsum()
        metrics = self._metrics(pnl_mid=pnl_mid, pnl_net=pnl_net, turnover=turnover)

        return BacktestResult(
            positions=positions,
            turnover=turnover,
            pnl_mid=pnl_mid,
            pnl_net=pnl_net,
            spread_cost=spread_cost,
            impact_cost=impact_cost,
            equity_mid=equity_mid,
            equity_net=equity_net,
            metrics=metrics,
        )

    def _metrics(self, pnl_mid: pd.Series, pnl_net: pd.Series, turnover: pd.DataFrame) -> pd.Series:
        def sharpe(pnl: pd.Series) -> float:
            std = pnl.std()
            if std == 0 or np.isnan(std):
                return np.nan
            return float(pnl.mean() / std * np.sqrt(self.periods_per_year))

        drawdown = pnl_net.cumsum() - pnl_net.cumsum().cummax()
        return pd.Series(
            {
                "pnl_mid_total": pnl_mid.sum(),
                "pnl_net_total": pnl_net.sum(),
                "sharpe_mid": sharpe(pnl_mid),
                "sharpe_net": sharpe(pnl_net),
                "hit_rate_net": (pnl_net > 0).mean(),
                "avg_turnover": turnover.sum(axis=1).mean(),
                "max_drawdown_net": drawdown.min(),
            }
        )

