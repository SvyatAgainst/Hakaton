import pandas as pd
import numpy as np

print("="*60)
print("M5: ОПТИМИЗАЦИЯ ИСПОЛНЕНИЯ (динамический participation)")
print("="*60)

# Загрузка данных
signal = pd.read_parquet("data/results/signal_5min.parquet")
bars_1min = pd.read_parquet("data/processed/bars_1min.parquet")
impact = pd.read_parquet("data/results/impact_model.parquet")

print(f"Сигнал: {len(signal)} строк")
print(f"1-мин бары: {len(bars_1min)} строк")
print(f"Impact модель: {len(impact)} строк")

# Подготовка
signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
bars_1min['timestamp'] = pd.to_datetime(bars_1min['timestamp'])
impact['bar_end_ts'] = pd.to_datetime(impact['bar_end_ts'])

# Объём рынка
bars_1min['volume_mkt'] = bars_1min['total_bid_volume'] + bars_1min['total_ask_volume']

# Берём сигналы с позицией
active = signal[signal['value'] != 0].copy()
print(f"Активных сигналов: {len(active)}")

# Агрегируем 1-мин в 5-мин (СОХРАНЯЕМ volume_mkt)
bars_5min_agg = bars_1min.groupby([bars_1min['ticker'], bars_1min['timestamp'].dt.floor('5min')]).agg({
    'open': 'first',
    'close': 'last',
    'volume_mkt': 'sum'
}).reset_index()
bars_5min_agg = bars_5min_agg.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode'})

# Объединяем
merged = active.merge(bars_5min_agg, on=['bar_end_ts', 'seccode'], how='inner')
merged = merged.sort_values(['seccode', 'bar_end_ts'])

# Следующий бар для PnL
merged['next_open'] = merged.groupby('seccode')['open'].shift(-1)
merged['next_close'] = merged.groupby('seccode')['close'].shift(-1)
merged = merged.dropna()

# PnL mid
merged['return'] = (merged['next_close'] - merged['next_open']) / merged['next_open']
merged['pnl_mid'] = merged['value'] * merged['return']

# Добавляем импакт
merged = merged.merge(impact[['bar_end_ts', 'seccode', 'a']], 
                       on=['bar_end_ts', 'seccode'], how='left')
merged['a'] = merged['a'].fillna(0.01)

# =============================================
# ДИНАМИЧЕСКИЙ PARTICIPATION
# =============================================

AUM = 1_000_000  # 1 млн рублей

# Объём нашей заявки (в лотах/штуках)
merged['our_volume'] = abs(merged['value']) * AUM / merged['next_open']

# participation = наша заявка / рыночный объём
merged['participation'] = merged['our_volume'] / merged['volume_mkt']

# Ограничиваем max participation (не более 30% рынка)
merged['participation'] = merged['participation'].clip(upper=0.3)

# Стоимость импакта: cost = a * participation
merged['impact_cost'] = merged['a'] * merged['participation'] * abs(merged['value'])

# Итоговый PnL
merged['pnl_net'] = merged['pnl_mid'] - merged['impact_cost']

print(f"\nРезультаты (динамический participation):")
print(f"  Сделок: {len(merged)}")
print(f"  Суммарный PnL mid: {merged['pnl_mid'].sum():.4f}")
print(f"  Суммарный импакт: {merged['impact_cost'].sum():.4f}")
print(f"  Суммарный PnL net: {merged['pnl_net'].sum():.4f}")
print(f"  Win Rate: {(merged['pnl_net'] > 0).mean():.2%}")

print(f"\nСтатистика participation:")
print(f"  Средний: {merged['participation'].mean():.6f} ({merged['participation'].mean()*100:.4f}%)")
print(f"  Медианный: {merged['participation'].median():.6f}")
print(f"  Максимальный: {merged['participation'].max():.6f}")

# Сохраняем
merged.to_parquet("data/results/execution_summary_profitable.parquet", index=False)
print(f"\n✅ Сохранено: data/results/execution_summary_profitable.parquet")

print("\nПример сделок:")
print(merged[['bar_end_ts', 'seccode', 'value', 'our_volume', 'volume_mkt', 'participation', 'pnl_net']].head(10))
