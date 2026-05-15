from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExecutionResult:
    schedule: pd.DataFrame
    summary: pd.Series


class TWAPExecutor:
    """Simple execution simulator: split parent order into equal time slices."""

    def __init__(self, impact_alpha: float = 0.0005, impact_beta: float = 0.5) -> None:
        self.impact_alpha = impact_alpha
        self.impact_beta = impact_beta

    def simulate(
        self,
        mid: pd.Series,
        spread: pd.Series,
        liquidity: pd.Series,
        side: str = "buy",
        order_size: float = 100_000.0,
        n_slices: int = 30,
    ) -> ExecutionResult:
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        if n_slices <= 0:
            raise ValueError("n_slices must be positive")

        frame = pd.DataFrame({"mid": mid, "spread": spread, "liquidity": liquidity}).dropna().head(n_slices)
        if frame.empty:
            raise ValueError("No valid market data for TWAP simulation")

        signed_direction = 1 if side == "buy" else -1
        frame["slice_qty"] = order_size / len(frame)
        participation = (frame["slice_qty"] / frame["liquidity"].replace(0, np.nan)).fillna(0.0)
        frame["impact_per_share"] = frame["mid"] * self.impact_alpha * (participation.clip(lower=0.0) ** self.impact_beta)
        frame["spread_cost_per_share"] = frame["spread"] / 2
        frame["exec_price"] = frame["mid"] + signed_direction * (
            frame["spread_cost_per_share"] + frame["impact_per_share"]
        )
        frame["cash"] = frame["exec_price"] * frame["slice_qty"]

        arrival_price = frame["mid"].iloc[0]
        avg_exec_price = frame["cash"].sum() / frame["slice_qty"].sum()
        implementation_shortfall = signed_direction * (avg_exec_price - arrival_price) / arrival_price

        summary = pd.Series(
            {
                "side": side,
                "order_size": order_size,
                "n_slices": len(frame),
                "arrival_price": arrival_price,
                "avg_exec_price": avg_exec_price,
                "implementation_shortfall": implementation_shortfall,
                "total_cash": frame["cash"].sum(),
            }
        )
        return ExecutionResult(schedule=frame, summary=summary)

