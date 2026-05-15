from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .data import MarketData, ParquetFeatureLoader
from .features import FeatureBuilder


@dataclass(frozen=True)
class FormalPipelineResult:
    """Strict module outputs from the formal hackathon task."""

    signal_5min: pd.DataFrame
    backtest_baseline: pd.DataFrame
    impact_model: pd.DataFrame
    execution_schedule: pd.DataFrame
    execution_summary: pd.DataFrame
    metrics: pd.Series


class FormalHackathonPipeline:
    """M2-M5 pipeline from the formal task.

    M2 uses 5-minute bars for the signal.
    M3 evaluates the signal on 5-minute bars by mid/open-close returns.
    M4 builds a linear one-minute impact model.
    M5 executes target volume inside each 5-minute bar using 1-minute slices.
    """

    def __init__(
        self,
        data_root: str | Path = "insample_data",
        delta: float = 0.10,
        impact_a: float = 0.03,
        aum_rub: float = 50_000_000,
        x_max: float = 0.30,
        top_n: int | None = None,
        min_avg_volume: float | None = 100_000,
        ema_span: int = 6,
        session_start: str = "10:00",
        session_end: str = "18:30",
    ) -> None:
        self.data_root = Path(data_root)
        self.delta = delta
        self.impact_a = impact_a
        self.aum_rub = aum_rub
        self.x_max = x_max
        self.top_n = top_n
        self.min_avg_volume = min_avg_volume
        self.ema_span = ema_span
        self.session_start = session_start
        self.session_end = session_end

    def run(self, max_execution_bars: int | None = 1000) -> FormalPipelineResult:
        data_5min = self._load_data("5min")
        data_1min = self._load_data("1min")

        universe = self._select_universe(data_5min)
        signal_5min_wide = self._build_signal_5min(data_5min, universe)
        signal_5min = self._wide_to_long(signal_5min_wide, "value")

        backtest_wide, positions_wide = self._backtest_baseline_wide(
            signal_5min_wide=signal_5min_wide,
            data_5min=data_5min,
        )
        backtest_baseline = self._backtest_to_long(positions_wide, backtest_wide)

        impact_model_wide = self._build_impact_model_wide(data_1min, universe)
        impact_model = self._impact_to_long(impact_model_wide, data_1min)

        execution_schedule, execution_summary = self._build_execution(
            positions_wide=positions_wide,
            pnl_mid_wide=backtest_wide,
            data_1min=data_1min,
            impact_model_wide=impact_model_wide,
            max_execution_bars=max_execution_bars,
        )
        metrics = self._metrics(backtest_baseline, execution_summary)

        return FormalPipelineResult(
            signal_5min=signal_5min,
            backtest_baseline=backtest_baseline,
            impact_model=impact_model,
            execution_schedule=execution_schedule,
            execution_summary=execution_summary,
            metrics=metrics,
        )

    def _load_data(self, timeframe: str) -> MarketData:
        data = ParquetFeatureLoader(data_root=self.data_root, timeframe=timeframe).load()
        return self._filter_session(data)

    def _filter_session(self, data: MarketData) -> MarketData:
        values = {
            field: self._filter_frame(getattr(data, field))
            for field in data.__dataclass_fields__
        }
        return MarketData(**values)

    def _filter_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        return frame.between_time(self.session_start, self.session_end)

    def _select_universe(self, data: MarketData) -> list[str]:
        market_volume = self._market_volume(data).mean()
        book_liquidity = (data.best_bid_size + data.best_ask_size).mean()
        liquidity = (market_volume + book_liquidity).dropna().sort_values(ascending=False)

        if self.min_avg_volume is not None:
            liquid_names = market_volume[market_volume >= self.min_avg_volume].dropna().index
            liquidity = liquidity.loc[liquidity.index.intersection(liquid_names)]

        if self.top_n is not None:
            liquidity = liquidity.head(self.top_n)

        if liquidity.empty:
            raise ValueError("No instruments passed the liquidity filter. Lower min_avg_volume or set top_n.")

        return liquidity.index.tolist()

    def _build_signal_5min(self, data: MarketData, universe: list[str]) -> pd.DataFrame:
        features = FeatureBuilder(horizon_bars=1, volatility_window=6).build(data)
        trade_ema = features.trade_imbalance[universe].ewm(span=self.ema_span, min_periods=1).mean()
        book_ema = features.book_imbalance[universe].ewm(span=self.ema_span, min_periods=1).mean()
        trend_ema = features.mid[universe].pct_change(fill_method=None).ewm(span=self.ema_span, min_periods=1).mean()

        trend_rank = trend_ema.rank(axis=1, pct=True).sub(0.5).fillna(0.0)
        spread_penalty = features.spread_rel[universe].rank(axis=1, pct=True).sub(0.5).fillna(0.0)
        volatility_penalty = features.volatility[universe].rank(axis=1, pct=True).sub(0.5).fillna(0.0)

        raw_score = (
            0.45 * trade_ema.fillna(0.0)
            + 0.35 * book_ema.fillna(0.0)
            + 0.20 * trend_rank
            - 0.10 * spread_penalty
            - 0.10 * volatility_penalty
        ).replace([np.inf, -np.inf], np.nan)
        valid_signal = trade_ema.notna() | book_ema.notna() | trend_ema.notna()
        centered = raw_score.sub(raw_score.median(axis=1), axis=0)
        scale = centered.abs().median(axis=1).replace(0, np.nan)
        signal = np.tanh(centered.div(scale, axis=0)).clip(-1, 1)
        return signal.where(valid_signal)

    def _backtest_baseline_wide(
        self,
        signal_5min_wide: pd.DataFrame,
        data_5min: MarketData,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        signal = signal_5min_wide.where(signal_5min_wide.abs() >= self.delta, 0.0)
        positions = self._dollar_neutral_positions(signal.shift(1).fillna(0.0))

        open_raw = getattr(data_5min, "open", None)
        if open_raw is None:
            open_raw = ParquetFeatureLoader(data_root=self.data_root, timeframe="5min").read_feature("open_.parquet")
        open_px = open_raw.reindex_like(positions).replace(0, np.nan)
        close_px = data_5min.close.reindex_like(positions)
        bar_return = close_px / open_px - 1
        pnl_mid = positions * bar_return
        return pnl_mid.fillna(0.0), positions

    @staticmethod
    def _dollar_neutral_positions(signal: pd.DataFrame) -> pd.DataFrame:
        longs = signal.clip(lower=0.0)
        shorts = (-signal.clip(upper=0.0))

        long_sum = longs.sum(axis=1).replace(0, np.nan)
        short_sum = shorts.sum(axis=1).replace(0, np.nan)
        has_both_sides = long_sum.notna() & short_sum.notna()

        long_weights = longs.div(long_sum, axis=0).fillna(0.0) * 0.5
        short_weights = shorts.div(short_sum, axis=0).fillna(0.0) * 0.5
        positions = long_weights - short_weights
        return positions.where(has_both_sides, 0.0)

    def _build_impact_model_wide(self, data_1min: MarketData, universe: list[str]) -> pd.DataFrame:
        volume_mkt = self._market_volume(data_1min)[universe]
        volatility_rank = data_1min.pr_std[universe].rank(axis=1, pct=True).fillna(0.5)
        a = self.impact_a * (0.75 + 0.5 * volatility_rank)
        return a.clip(0.01, 0.05).where(volume_mkt.notna())

    def _build_execution(
        self,
        positions_wide: pd.DataFrame,
        pnl_mid_wide: pd.DataFrame,
        data_1min: MarketData,
        impact_model_wide: pd.DataFrame,
        max_execution_bars: int | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        volume_1min = self._market_volume(data_1min)
        price_1min = ((data_1min.best_bid + data_1min.best_ask) / 2).fillna(data_1min.close)

        schedules: list[pd.DataFrame] = []
        summaries: list[dict[str, float | int | str]] = []

        active_bars = positions_wide[positions_wide.abs().sum(axis=1) > 0]
        if max_execution_bars is not None:
            active_bars = active_bars.head(max_execution_bars)

        for bar_end_ts, row in active_bars.iterrows():
            start_ts = bar_end_ts - pd.Timedelta(minutes=5)
            slice_index = price_1min.loc[(price_1min.index > start_ts) & (price_1min.index <= bar_end_ts)].index
            if slice_index.empty:
                continue

            for seccode, weight in row[row != 0].items():
                price_slices = price_1min.loc[slice_index, seccode].dropna()
                if price_slices.empty:
                    continue
                volume_slices = volume_1min.loc[price_slices.index, seccode].replace(0, np.nan).dropna()
                common_index = price_slices.index.intersection(volume_slices.index)
                if common_index.empty:
                    continue

                price_slices = price_slices.loc[common_index]
                volume_slices = volume_slices.loc[common_index]
                a_slices = impact_model_wide.loc[common_index, seccode].fillna(self.impact_a)

                price_ref = price_slices.iloc[0]
                q_total = abs(weight) * self.aum_rub / price_ref
                q_total = min(q_total, float(self.x_max * volume_slices.sum()))
                if q_total <= 0:
                    continue

                q_abs = self._impact_optimal_schedule(q_total, volume_slices, a_slices)
                direction = 1 if weight > 0 else -1
                q_signed = direction * q_abs
                participation = (q_abs / volume_slices).clip(upper=self.x_max)
                impact_cost_rel = a_slices * participation
                fill_price = price_slices + direction * impact_cost_rel * price_slices
                cash_abs = q_abs * fill_price

                schedule = pd.DataFrame(
                    {
                        "bar_end_ts_5min": bar_end_ts.value,
                        "bar_end_ts_1min": common_index.view("int64"),
                        "seccode": seccode,
                        "q_slice": q_signed.to_numpy(),
                        "participation_rate": participation.to_numpy(),
                        "impact_cost_rel": impact_cost_rel.to_numpy(),
                    },
                    index=common_index,
                )
                schedules.append(schedule)

                is_vwap = float((a_slices * participation**2 * volume_slices * price_slices).sum())
                twap_q = pd.Series(q_total / len(common_index), index=common_index)
                twap_participation = twap_q / volume_slices
                is_twap = float((a_slices * twap_participation**2 * volume_slices * price_slices).sum())
                twap_fill = float(
                    ((price_slices + direction * a_slices * twap_participation * price_slices) * twap_q).sum()
                    / twap_q.sum()
                )

                pnl_mid = float(pnl_mid_wide.loc[bar_end_ts, seccode] * self.aum_rub)
                summaries.append(
                    {
                        "bar_end_ts_5min": bar_end_ts.value,
                        "seccode": seccode,
                        "Q_executed": float(q_abs.sum()),
                        "vwap_fill": float(cash_abs.sum() / q_abs.sum()),
                        "twap_bench": twap_fill,
                        "implementation_shortfall": is_vwap,
                        "implementation_shortfall_twap": is_twap,
                        "is_reduction": (is_twap - is_vwap) / is_twap if is_twap > 0 else np.nan,
                        "pnl_mid": pnl_mid,
                        "pnl_net": pnl_mid - is_vwap,
                    }
                )

        execution_schedule = pd.concat(schedules, ignore_index=True) if schedules else self._empty_schedule()
        execution_summary = pd.DataFrame(summaries) if summaries else self._empty_summary()
        return execution_schedule, execution_summary

    def _impact_optimal_schedule(
        self,
        q_total: float,
        volume_slices: pd.Series,
        a_slices: pd.Series,
    ) -> pd.Series:
        q_abs = pd.Series(0.0, index=volume_slices.index)
        cap = self.x_max * volume_slices
        weights = (volume_slices / a_slices.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        remaining = min(q_total, float(cap.sum()))
        free_index = volume_slices.index

        while remaining > 1e-9 and len(free_index) > 0:
            free_weights = weights.loc[free_index]
            if free_weights.sum() <= 0:
                allocation = pd.Series(remaining / len(free_index), index=free_index)
            else:
                allocation = remaining * free_weights / free_weights.sum()

            over_cap = allocation > cap.loc[free_index]
            if not over_cap.any():
                q_abs.loc[free_index] = allocation
                break

            capped_index = allocation[over_cap].index
            q_abs.loc[capped_index] = cap.loc[capped_index]
            remaining -= float(cap.loc[capped_index].sum())
            free_index = free_index.difference(capped_index)

        return q_abs

    @staticmethod
    def _market_volume(data: MarketData) -> pd.DataFrame:
        return (data.buy_size + data.sell_size).replace(0, np.nan)

    @staticmethod
    def _wide_to_long(frame: pd.DataFrame, value_name: str) -> pd.DataFrame:
        result = frame.stack().rename(value_name).reset_index()
        result.columns = ["bar_end_dt", "seccode", value_name]
        result.insert(0, "bar_end_ts", result["bar_end_dt"].astype("int64"))
        return result.drop(columns=["bar_end_dt"])

    def _backtest_to_long(self, positions: pd.DataFrame, pnl_mid: pd.DataFrame) -> pd.DataFrame:
        records = []
        cum_pnl = pnl_mid.cumsum()
        for name, frame in {"pos": positions, "pnl_mid": pnl_mid, "cum_pnl_mid": cum_pnl}.items():
            part = self._wide_to_long(frame, name)
            records.append(part)

        result = records[0]
        for part in records[1:]:
            result = result.merge(part, on=["bar_end_ts", "seccode"], how="left")
        return result

    def _impact_to_long(self, impact_a: pd.DataFrame, data_1min: MarketData) -> pd.DataFrame:
        volume_mkt = self._market_volume(data_1min).reindex_like(impact_a)
        a_long = self._wide_to_long(impact_a, "a")
        volume_long = self._wide_to_long(volume_mkt, "volume_mkt")
        result = volume_long.merge(a_long, on=["bar_end_ts", "seccode"], how="left")
        result["volume_mkt"] = result["volume_mkt"].fillna(0).astype("int64")
        return result

    @staticmethod
    def _empty_schedule() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "bar_end_ts_5min",
                "bar_end_ts_1min",
                "seccode",
                "q_slice",
                "participation_rate",
                "impact_cost_rel",
            ]
        )

    @staticmethod
    def _empty_summary() -> pd.DataFrame:
        return pd.DataFrame(
            columns=[
                "bar_end_ts_5min",
                "seccode",
                "Q_executed",
                "vwap_fill",
                "twap_bench",
                "implementation_shortfall",
                "implementation_shortfall_twap",
                "is_reduction",
                "pnl_mid",
                "pnl_net",
            ]
        )

    @staticmethod
    def _metrics(backtest_baseline: pd.DataFrame, execution_summary: pd.DataFrame) -> pd.Series:
        pnl_mid = backtest_baseline.groupby("bar_end_ts")["pnl_mid"].sum()
        pnl_net = execution_summary.groupby("bar_end_ts_5min")["pnl_net"].sum()
        is_reduction = execution_summary["is_reduction"].replace([np.inf, -np.inf], np.nan)

        sharpe_net = np.nan
        if len(pnl_net) > 1 and pnl_net.std() != 0:
            sharpe_net = float(pnl_net.mean() / pnl_net.std() * np.sqrt(252 * 78))

        return pd.Series(
            {
                "signal_rows": len(backtest_baseline),
                "execution_rows": len(execution_summary),
                "pnl_mid_total": pnl_mid.sum(),
                "pnl_net_total": pnl_net.sum(),
                "sharpe_net": sharpe_net,
                "hit_rate_mid": (pnl_mid > 0).mean(),
                "avg_is_reduction": is_reduction.mean(),
            }
        )

