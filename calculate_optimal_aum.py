import pandas as pd
import numpy as np

print("="*60)
print("ДОПОЛНИТЕЛЬНАЯ ЗАДАЧА: ОПТИМАЛЬНЫЙ AUM")
print("="*60)

# Загрузка данных
signal = pd.read_parquet("data/results/signal_5min.parquet")
backtest = pd.read_parquet("data/results/backtest_baseline.parquet")
impact = pd.read_parquet("data/results/impact_model.parquet")
bars_5min = pd.read_parquet("data/processed/bars_5min.parquet")

print(f"Сигнал: {len(signal)} строк")
print(f"Бэктест: {len(backtest)} строк")
print(f"Impact модель: {len(impact)} строк")
print(f"5-мин бары: {len(bars_5min)} строк")

# 1. Оценка α (ожидаемая доходность сигнала)
print("\n" + "="*60)
print("1. ОЦЕНКА α (доходность сигнала)")
print("="*60)

# Берём только ненулевые сигналы
backtest_nonzero = backtest[backtest['pos'] != 0].copy()
backtest_nonzero['bar_end_ts'] = pd.to_datetime(backtest_nonzero['bar_end_ts'], unit='s')

# Средняя доходность на сделку
alpha_mean = backtest_nonzero['pnl_mid'].mean()
alpha_median = backtest_nonzero['pnl_mid'].median()
alpha_total = backtest_nonzero['pnl_mid'].sum()

print(f"Средняя доходность на сделку (α): {alpha_mean:.6f} ({alpha_mean*100:.4f}%)")
print(f"Медианная доходность: {alpha_median:.6f}")
print(f"Суммарная доходность: {alpha_total:.4f}")

# 2. Оценка ADV (среднедневной объём)
print("\n" + "="*60)
print("2. ОЦЕНКА ADV (среднедневной объём)")
print("="*60)

bars_5min['date'] = pd.to_datetime(bars_5min['timestamp']).dt.date
bars_5min['volume_mkt'] = bars_5min['total_bid_volume'] + bars_5min['total_ask_volume']

# Дневные объёмы по каждому инструменту
daily_volume = bars_5min.groupby(['ticker', 'date'])['volume_mkt'].sum()
adv = daily_volume.groupby('ticker').mean()

print(f"Средний ADV по всем инструментам: {adv.mean():.2f}")
print(f"Медианный ADV: {adv.median():.2f}")

# 3. Модель импакта a (уже константа 0.01)
print("\n" + "="*60)
print("3. КОЭФФИЦИЕНТ ИМПАКТА a")
print("="*60)

a_const = 0.01
print(f"a = {a_const} (константа)")

# 4. Расчёт оптимального AUM для каждого инструмента
print("\n" + "="*60)
print("4. ОПТИМАЛЬНЫЙ AUM ПО ИНСТРУМЕНТАМ")
print("="*60)

# Берём только инструменты, по которым есть сделки
tickers_with_trades = backtest_nonzero['seccode'].unique()
print(f"Инструментов со сделками: {len(tickers_with_trades)}")

results = []

for ticker in tickers_with_trades:
    # α для конкретного инструмента
    ticker_trades = backtest_nonzero[backtest_nonzero['seccode'] == ticker]
    alpha_ticker = ticker_trades['pnl_mid'].mean()
    
    # ADV для инструмента
    adv_ticker = adv.get(ticker, 0)
    
    if adv_ticker > 0 and alpha_ticker > 0:
        # X* = α * ADV / (2a)
        optimal_aum = alpha_ticker * adv_ticker / (2 * a_const)
        
        results.append({
            'ticker': ticker,
            'alpha': alpha_ticker,
            'ADV': adv_ticker,
            'optimal_AUM': optimal_aum,
            'trades_count': len(ticker_trades)
        })

# Сортируем по оптимальному AUM
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('optimal_AUM', ascending=False)

print(f"\nТОП-10 ИНСТРУМЕНТОВ ПО ОПТИМАЛЬНОМУ AUM:")
print(results_df.head(10).to_string(index=False))

# 5. Общая оценка
print("\n" + "="*60)
print("5. ИТОГОВАЯ ОЦЕНКА")
print("="*60)

total_optimal_aum = results_df['optimal_AUM'].sum()
avg_alpha = results_df['alpha'].mean()
avg_adv = results_df['ADV'].mean()

print(f"Суммарный оптимальный AUM по всем инструментам: {total_optimal_aum:,.2f} ₽")
print(f"Средний α: {avg_alpha:.6f} ({avg_alpha*100:.4f}%)")
print(f"Средний ADV: {avg_adv:.2f}")
print(f"\nРекомендуемый AUM для портфеля: {total_optimal_aum:,.0f} ₽")

# Сохраняем результаты
results_df.to_csv("reports/optimal_aum_by_ticker.csv", index=False)
print(f"\n✅ Сохранено: reports/optimal_aum_by_ticker.csv")

# Визуализация
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 6))
top10 = results_df.head(10)
colors = ['green' if x > 0 else 'red' for x in top10['optimal_AUM']]
ax.barh(range(len(top10)), top10['optimal_AUM'].values, color=colors)
ax.set_yticks(range(len(top10)))
ax.set_yticklabels(top10['ticker'].values)
ax.set_xlabel('Оптимальный AUM (рубли)')
ax.set_title('Топ-10 инструментов по оптимальному объёму портфеля')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('reports/plots/optimal_aum.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Сохранён график: reports/plots/optimal_aum.png")

print("\n" + "="*60)
print("ДОПОЛНИТЕЛЬНАЯ ЗАДАЧА ВЫПОЛНЕНА!")
print("="*60)
