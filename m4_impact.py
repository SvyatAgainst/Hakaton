import pandas as pd
import numpy as np

print("="*60)
print("M4: МОДЕЛЬ РЫНОЧНОГО ИМПАКТА (константа a = 0.01)")
print("="*60)

# Загрузка 1-минутных данных
bars_1min = pd.read_parquet("data/processed/bars_1min.parquet")
print(f"Загружено: {len(bars_1min)} строк")

# Создаём колонку объёма рынка
bars_1min['volume_mkt'] = bars_1min['total_bid_volume'] + bars_1min['total_ask_volume']

# КОНСТАНТА ИМПАКТА
bars_1min['a'] = 0.01

# Результат
impact_model = bars_1min[['timestamp', 'ticker', 'volume_mkt', 'a']].copy()
impact_model = impact_model.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode'})

print(f"\nКоэффициент импакта: a = {impact_model['a'].iloc[0]} (константа)")

# Сохраняем
impact_model.to_parquet("data/results/impact_model.parquet", index=False)
print(f"\n✅ Сохранено: data/results/impact_model.parquet ({len(impact_model)} строк)")

print("\nПример:")
print(impact_model.head(10))
