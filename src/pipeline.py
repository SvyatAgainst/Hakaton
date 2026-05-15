from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .backtest import BacktestResult, Backtester
from .data import MarketData, ParquetFeatureLoader
from .execution import ExecutionResult, TWAPExecutor
from .features import FeatureBuilder, FeatureSet
from .trading_signal import CrossSectionalSignal


@dataclass(frozen=True)
class PipelineResult:
    data: MarketData
    features: FeatureSet
    backtest: BacktestResult
    execution: ExecutionResult


class TradingPipeline:
    """End-to-end pipeline: features -> signal -> backtest -> execution."""

    def __init__(
        self,
        data_root: str | Path = "insample_data",
        timeframe: str = "1min",
        signal_horizon_minutes: int = 30,
    ) -> None:
        self.data_root = Path(data_root)
        self.timeframe = timeframe
        self.signal_horizon_minutes = signal_horizon_minutes

    @property
    def bars_per_signal(self) -> int:
        if self.timeframe in {"1min", "1_min", "1"}:
            return self.signal_horizon_minutes
        if self.timeframe in {"5min", "5_min", "5"}:
            return max(1, self.signal_horizon_minutes // 5)
        raise ValueError("timeframe must be one of: '1min', '5min'")

    @property
    def periods_per_year(self) -> int:
        if self.timeframe in {"1min", "1_min", "1"}:
            return 252 * 390
        if self.timeframe in {"5min", "5_min", "5"}:
            return 252 * 78
        raise ValueError("timeframe must be one of: '1min', '5min'")

    def run(
        self,
        execution_ticker: str = "SBER",
        execution_side: str = "buy",
        execution_size: float = 100_000.0,
        execution_slices: int | None = None,
    ) -> PipelineResult:
        loader = ParquetFeatureLoader(data_root=self.data_root, timeframe=self.timeframe)
        data = loader.load()

        features = FeatureBuilder(horizon_bars=self.bars_per_signal).build(data)
        positions = CrossSectionalSignal().make_positions(features)
        backtest = Backtester(periods_per_year=self.periods_per_year).run(positions=positions, features=features)

        ticker = execution_ticker if execution_ticker in features.mid.columns else features.mid.columns[0]
        execution = TWAPExecutor().simulate(
            mid=features.mid[ticker],
            spread=features.spread[ticker],
            liquidity=features.liquidity[ticker],
            side=execution_side,
            order_size=execution_size,
            n_slices=execution_slices or self.bars_per_signal,
        )

        return PipelineResult(
            data=data,
            features=features,
            backtest=backtest,
            execution=execution,
        )

