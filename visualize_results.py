import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Настройка стиля
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*60)
print("ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ СТРАТЕГИИ")
print("="*60)

# Создаём папку для графиков
import os
os.makedirs('reports/plots', exist_ok=True)

# Загрузка данных
signal = pd.read_parquet("data/results/signal_5min.parquet")
backtest = pd.read_parquet("data/results/backtest_baseline.parquet")
execution = pd.read_parquet("data/results/execution_summary_profitable.parquet")

# Преобразуем timestamp
signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
backtest['bar_end_ts'] = pd.to_datetime(backtest['bar_end_ts'], unit='s')
execution['bar_end_ts'] = pd.to_datetime(execution['bar_end_ts'])

print("✅ Данные загружены")

# 1. Распределение сигналов
print("\n1. Распределение сигналов...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

signal_nonzero = signal[signal['value'] != 0]
axes[0].hist(signal_nonzero['value'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('Значение сигнала')
axes[0].set_ylabel('Частота')
axes[0].set_title('Распределение торговых сигналов')
axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)

long_count = (signal_nonzero['value'] > 0).sum()
short_count = (signal_nonzero['value'] < 0).sum()
axes[1].pie([long_count, short_count], labels=['Long', 'Short'], 
            autopct='%1.1f%%', colors=['green', 'red'], explode=(0.05, 0))
axes[1].set_title('Соотношение Long/Short')

plt.tight_layout()
plt.savefig('reports/plots/signal_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ signal_distribution.png")

# 2. Кумулятивный PnL
print("\n2. Кумулятивный PnL...")
fig, ax = plt.subplots(figsize=(12, 6))

backtest_grouped = backtest.groupby('bar_end_ts')['pnl_mid'].sum().cumsum()
ax.plot(backtest_grouped.index, backtest_grouped.values, linewidth=1.5, color='blue')
ax.fill_between(backtest_grouped.index, 0, backtest_grouped.values, alpha=0.3, color='blue')
ax.set_xlabel('Время')
ax.set_ylabel('Кумулятивный PnL')
ax.set_title('Кумулятивная прибыль (без учёта импакта)')
ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('reports/plots/cumulative_pnl.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ cumulative_pnl.png")

# 3. Топ-10 инструментов
print("\n3. Топ-10 инструментов...")
fig, ax = plt.subplots(figsize=(10, 6))

ticker_pnl = execution.groupby('seccode')['pnl_net'].sum().sort_values(ascending=False).head(10)
colors = ['green' if x > 0 else 'red' for x in ticker_pnl.values]
ax.barh(range(len(ticker_pnl)), ticker_pnl.values, color=colors)
ax.set_yticks(range(len(ticker_pnl)))
ax.set_yticklabels(ticker_pnl.index)
ax.set_xlabel('PnL net')
ax.set_title('Топ-10 инструментов по прибыли')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.tight_layout()
plt.savefig('reports/plots/top_instruments.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ top_instruments.png")

# 4. Распределение PnL по сделкам
print("\n4. Распределение PnL...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pnl_values = execution['pnl_net'].dropna().values
axes[0].hist(pnl_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
axes[0].set_xlabel('PnL на сделку')
axes[0].set_ylabel('Частота')
axes[0].set_title('Распределение PnL по сделкам')
axes[0].axvline(x=0, color='red', linestyle='--', alpha=0.5)
axes[0].axvline(x=pnl_values.mean(), color='blue', linestyle='--', alpha=0.5, label=f'Среднее: {pnl_values.mean():.6f}')
axes[0].legend()

profit_data = execution[execution['pnl_net'] > 0]['pnl_net']
loss_data = execution[execution['pnl_net'] < 0]['pnl_net']
bp = axes[1].boxplot([profit_data, loss_data], labels=['Прибыльные', 'Убыточные'], patch_artist=True)
bp['boxes'][0].set_facecolor('green')
bp['boxes'][1].set_facecolor('red')
axes[1].set_ylabel('PnL')
axes[1].set_title('Сравнение сделок')

plt.tight_layout()
plt.savefig('reports/plots/pnl_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ pnl_distribution.png")

# 5. Win Rate по дням
print("\n5. Win Rate по дням...")
fig, ax = plt.subplots(figsize=(12, 5))

execution['date'] = execution['bar_end_ts'].dt.date
daily_winrate = execution.groupby('date').apply(lambda x: (x['pnl_net'] > 0).mean())
ax.bar(range(len(daily_winrate)), daily_winrate.values, alpha=0.7, color='steelblue')
ax.set_xlabel('День')
ax.set_ylabel('Win Rate')
ax.set_title('Ежедневный Win Rate')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Случайный уровень (50%)')
ax.axhline(y=daily_winrate.mean(), color='blue', linestyle='--', alpha=0.5, label=f'Средний: {daily_winrate.mean():.1%}')
ax.legend()
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('reports/plots/daily_winrate.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ daily_winrate.png")

# 6. График сигнала во времени
print("\n6. Сигнал во времени (пример для одного инструмента)...")
fig, ax = plt.subplots(figsize=(12, 5))

# Берём один инструмент
ticker_example = 'SBER'
signal_ticker = signal[signal['seccode'] == ticker_example].copy()
if len(signal_ticker) > 0:
    ax.plot(signal_ticker['bar_end_ts'], signal_ticker['value'], linewidth=0.5, color='steelblue')
    ax.fill_between(signal_ticker['bar_end_ts'], 0, signal_ticker['value'], alpha=0.3, 
                     where=(signal_ticker['value'] > 0), color='green', label='Long сигнал')
    ax.fill_between(signal_ticker['bar_end_ts'], 0, signal_ticker['value'], alpha=0.3,
                     where=(signal_ticker['value'] < 0), color='red', label='Short сигнал')
    ax.set_xlabel('Время')
    ax.set_ylabel('Сигнал')
    ax.set_title(f'Сигналы для инструмента {ticker_example}')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    plt.savefig('reports/plots/signal_over_time.png', dpi=150, bbox_inches='tight')
    print("  ✅ signal_over_time.png")
else:
    print(f"  ⚠️ Инструмент {ticker_example} не найден")

plt.close()

print("\n" + "="*60)
print("✅ ГРАФИКИ СОЗДАНЫ!")
print("   Папка: reports/plots/")
print("="*60)
