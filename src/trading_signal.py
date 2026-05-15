from __future__ import annotations

import numpy as np
import pandas as pd

from .features import FeatureSet


class CrossSectionalSignal:
    """Turns feature scores into long/short/flat portfolio weights."""

    def __init__(self, long_quantile: float = 0.80, short_quantile: float = 0.20, gross_exposure: float = 1.0) -> None:
        if not 0 < short_quantile < long_quantile < 1:
            raise ValueError("Expected 0 < short_quantile < long_quantile < 1")
        self.long_quantile = long_quantile
        self.short_quantile = short_quantile
        self.gross_exposure = gross_exposure

    def predict_score(self, features: FeatureSet) -> pd.DataFrame:
        return features.score

    def make_positions(self, features: FeatureSet) -> pd.DataFrame:
        score = self.predict_score(features)
        ranks = score.rank(axis=1, pct=True)

        raw_positions = pd.DataFrame(0.0, index=score.index, columns=score.columns)
        raw_positions = raw_positions.mask(ranks >= self.long_quantile, 1.0)
        raw_positions = raw_positions.mask(ranks <= self.short_quantile, -1.0)
        raw_positions = raw_positions.where(features.target_return.notna())

        # Normalize by active names so every timestamp has comparable portfolio risk.
        active_count = raw_positions.abs().sum(axis=1).replace(0, np.nan)
        return raw_positions.div(active_count, axis=0).fillna(0.0) * self.gross_exposure

