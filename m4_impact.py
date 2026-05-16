import pandas as pd
import numpy as np

print("="*60)
print("M4: МОДЕЛЬ РЫНОЧНОГО ИМПАКТА (на 1-мин данных)")
print("="*60)

# Загрузка 1-минутных данных
bars_1min = pd.read_parquet("data/processed/bars_1min.parquet")
print(f"Загружено: {len(bars_1min)} строк")

# Создаём колонку объёма рынка
bars_1min['volume_mkt'] = bars_1min['total_bid_volume'] + bars_1min['total_ask_volume']

# Расчёт коэффициента импакта a(t)
# cost(x) = a * x, где x = participation_rate

# 1. Базовый коэффициент (спред как прокси ликвидности)
bars_1min['spread'] = (bars_1min['high'] - bars_1min['low']) / bars_1min['close']
bars_1min['a_base'] = bars_1min['spread'] * 0.5

# 2. Фактор волатильности
bars_1min['volatility'] = bars_1min['close'].pct_change().rolling(5).std()
bars_1min['a_vol'] = bars_1min['volatility'].fillna(0.01)

# 3. Фактор объёма (нормированный)
for ticker in bars_1min['ticker'].unique():
    mask = bars_1min['ticker'] == ticker
    avg_volume = bars_1min.loc[mask, 'volume_mkt'].median()
    if avg_volume > 0:
        bars_1min.loc[mask, 'volume_norm'] = bars_1min.loc[mask, 'volume_mkt'] / avg_volume
    else:
        bars_1min.loc[mask, 'volume_norm'] = 1.0

# 4. Итоговый коэффициент импакта
bars_1min['a'] = bars_1min['a_base'] + bars_1min['a_vol'] * bars_1min['volume_norm'].clip(0.5, 2)
bars_1min['a'] = bars_1min['a'].clip(0.005, 0.05)

# Результат
impact_model = bars_1min[['timestamp', 'ticker', 'volume_mkt', 'a']].copy()
impact_model = impact_model.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode'})

print(f"\nРезультаты модели импакта:")
print(f"  Средний a: {impact_model['a'].mean():.4f}")
print(f"  Медианный a: {impact_model['a'].median():.4f}")
print(f"  Диапазон a: [{impact_model['a'].min():.4f}, {impact_model['a'].max():.4f}]")

# Сохраняем
impact_model.to_parquet("data/results/impact_model.parquet", index=False)
print(f"\n✅ Сохранено: data/results/impact_model.parquet ({len(impact_model)} строк)")

print("\nПример:")
print(impact_model.head(10))
