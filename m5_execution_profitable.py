import pandas as pd
import numpy as np

print("="*60)
print("M5: ПРИБЫЛЬНАЯ СТРАТЕГИЯ (УЧАСТИЕ 0.035%)")
print("="*60)

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

merged['participation'] = 0.00035
merged['impact_cost'] = merged['a'] * merged['participation'] * abs(merged['value'])
merged['pnl_net'] = merged['pnl_mid'] - merged['impact_cost']

print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ (участие 0.035%):")
print(f"  Сделок: {len(merged)}")
print(f"  Суммарный PnL mid: {merged['pnl_mid'].sum():.4f}")
print(f"  Суммарный импакт: {merged['impact_cost'].sum():.4f}")
print(f"  Суммарный PnL net: {merged['pnl_net'].sum():.4f}")
print(f"  Win Rate: {(merged['pnl_net'] > 0).mean():.2%}")

merged.to_parquet("data/results/execution_summary_profitable.parquet", index=False)
print(f"\n✅ Сохранено: data/results/execution_summary_profitable.parquet")
