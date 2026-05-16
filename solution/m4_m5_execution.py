import numpy as np
import pandas as pd

class MarketImpactModel:
    def __init__(self, a_coef=0.03):
        self.a = a_coef

    def get_participation_rate(self, my_volume, market_volume):
        if market_volume <= 0 or pd.isna(market_volume):
            return 0.0
        return abs(my_volume) / market_volume

    def calculate_impact_cost(self, x):
        return self.a * x

    def get_execution_price(self, ideal_price, my_volume, market_volume):
        if my_volume == 0 or market_volume <= 0 or pd.isna(market_volume):
            return ideal_price, 0.0
        x = self.get_participation_rate(my_volume, market_volume)
        cost_fraction = self.calculate_impact_cost(x)
        direction = np.sign(my_volume)
        real_price = ideal_price * (1 + (direction * cost_fraction))
        return real_price, cost_fraction

class TWAPExecutor:
    def __init__(self, impact_model: MarketImpactModel, num_slices=30):
        self.impact_model = impact_model
        self.K = num_slices

    def execute_order(self, target_volume, minute_bars_data):
        """
        minute_bars_data: список словарей [{'price': ..., 'market_volume': ...}, ...]
        """
        if target_volume == 0 or len(minute_bars_data) == 0:
            return {"vwap_fill": 0.0, "total_is": 0.0, "executed_volume": 0.0}

        slice_volume = target_volume / self.K
        total_value_traded = 0.0  
        total_is = 0.0            
        executed_abs_volume = 0.0 

        for k_data in minute_bars_data:
            ideal_price = k_data['price']
            market_vol_1min = k_data['market_volume']
            
            real_price, cost_fraction = self.impact_model.get_execution_price(
                ideal_price=ideal_price, 
                my_volume=slice_volume, 
                market_volume=market_vol_1min
            )
            
            slice_is = abs(slice_volume) * ideal_price * cost_fraction
            
            total_value_traded += abs(slice_volume) * real_price
            total_is += slice_is
            executed_abs_volume += abs(slice_volume)

        vwap_fill = total_value_traded / executed_abs_volume if executed_abs_volume > 0 else 0.0

        return {
            "vwap_fill": vwap_fill,
            "total_is": total_is,
            "target_volume": target_volume
        }