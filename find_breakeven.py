import pandas as pd
import numpy as np

print("Поиск точки безубыточности...")

signal = pd.read_parquet("data/results/signal_5min.parquet")
bars_1min = pd.read_parquet("data/processed/bars_1min.parquet")
impact = pd.read_parquet("data/results/impact_model.parquet")

signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
bars_1min['timestamp'] = pd.to_datetime(bars_1min['timestamp'])
impact['bar_end_ts'] = pd.to_datetime(impact['bar_end_ts'])

bars_1min['volume_mkt'] = bars_1min['total_bid_volume'] + bars_1min['total_ask_volume']

active = signal[signal['value'] != 0].copy()

bars_5min_agg = bars_1min.groupby([bars_1min['ticker'], bars_1min['timestamp'].dt.floor('5min')]).agg({
    'open': 'first',
    'close': 'last',
    'volume_mkt': 'sum'
}).reset_index()
bars_5min_agg = bars_5min_agg.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode'})

merged = active.merge(bars_5min_agg, on=['bar_end_ts', 'seccode'], how='inner')
merged = merged.sort_values(['seccode', 'bar_end_ts'])

merged['next_open'] = merged.groupby('seccode')['open'].shift(-1)
merged['next_close'] = merged.groupby('seccode')['close'].shift(-1)
merged = merged.dropna()

merged['return'] = (merged['next_close'] - merged['next_open']) / merged['next_open']
merged['pnl_mid'] = merged['value'] * merged['return']

merged = merged.merge(impact[['bar_end_ts', 'seccode', 'a', 'volume_mkt']], 
                       on=['bar_end_ts', 'seccode'], how='left')
merged['a'] = merged['a'].fillna(0.01)

pnl_mid_total = merged['pnl_mid'].sum()
print(f"PnL mid (без импакта): {pnl_mid_total:.6f}")

# Находим participation при котором PnL net = 0
# pnl_net = pnl_mid - a * participation * |value| = 0
# participation = pnl_mid / (a * |value|)

merged['impact_per_abs'] = merged['a'] * abs(merged['value'])
total_impact_base = (merged['impact_per_abs']).sum()
breakeven_participation = pnl_mid_total / total_impact_base

print(f"\nТочка безубыточности:")
print(f"  Participation rate: {breakeven_participation:.6f} ({breakeven_participation*100:.4f}%)")
print(f"  PnL net при этом: 0.00")

# Проверяем
merged['pnl_net_breakeven'] = merged['pnl_mid'] - merged['impact_per_abs'] * breakeven_participation
print(f"  Проверка PnL net: {merged['pnl_net_breakeven'].sum():.8f}")

# Рекомендация
recommended = breakeven_participation * 0.9
print(f"\nРекомендуемый participation (с запасом 10%): {recommended:.6f} ({recommended*100:.4f}%)")
