import pandas as pd
import numpy as np

# Загрузка данных
filenames = {
    "open": "./data/open_.parquet",
    "close": "./data/close_.parquet",
    "high": "./data/high_.parquet",
    "low": "./data/low_.parquet",
    "buy_size": "./data/buy_size_.parquet",
    "sell_size": "./data/sell_size_.parquet",
    "total_ask_volume": "./data/total_ask_volume_.parquet",
    "total_bid_volume": "./data/total_bid_volume_.parquet"
}

dataframes = {
    key: pd.read_parquet(path) 
    for key, path in filenames.items()
}

# Создаём MultiIndex DataFrame
df_3d = pd.concat(list(dataframes.values()), axis=1, keys=list(dataframes.keys()))
df_swap = df_3d.swaplevel(0, 1, axis=1).sort_index(axis=1)

print(df_swap)

# РАСПРЕДЕЛЕНИЕ ПО РАЗНЫМ УРОВНЯМ

print("\n" + "="*60)
print("РАСПРЕДЕЛЕНИЕ ПО ТИКЕРАМ")
print("="*60)

ticker_counts = {}
for ticker in df_swap.columns.levels[0]:
    if 'close' in df_swap[ticker].columns:
        ticker_counts[ticker] = df_swap[ticker]['close'].count()
    else:
        ticker_counts[ticker] = 0

ticker_sorted = dict(sorted(ticker_counts.items(), key=lambda x: x[1], reverse=True))
print(f"Всего тикеров: {len(ticker_sorted)}")
print("\nТоп-10 тикеров по количеству данных:")
for i, (ticker, count) in enumerate(list(ticker_sorted.items())[:10]):
    print(f"  {i+1}. {ticker}: {count} записей")

print("\nРаспределение тикеров по количеству данных:")
bins = [0, 1000, 2000, 3000, 4000, 5000, 6000]
labels = ['1-1000', '1001-2000', '2001-3000', '3001-4000', '4001-5000', '5001+']
ticker_dist = pd.cut(pd.Series(ticker_counts), bins=bins, labels=labels)
print(ticker_dist.value_counts().sort_index())

print("\n" + "="*60)
print("РАСПРЕДЕЛЕНИЕ ПО ПАРАМЕТРАМ")
print("="*60)

param_counts = {}
for param in df_swap.columns.levels[1]:
    param_data = df_swap.xs(param, axis=1, level=1)
    param_counts[param] = param_data.count().sum()

for param, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {param}: {count} записей")

print("\n" + "="*60)
print("РАСПРЕДЕЛЕНИЕ ПО ВРЕМЕНИ")
print("="*60)

if isinstance(df_swap.index, pd.DatetimeIndex):
    hour_dist = df_swap.index.hour.value_counts().sort_index()
    print("\nПо часам:")
    for hour, count in hour_dist.items():
        print(f"  {hour:02d}:00 - {count} записей")
    
    date_counts = pd.Series(df_swap.index.date).value_counts().sort_index()
    print(f"\nПо дням (всего {len(date_counts)} дней):")
    for date, count in list(date_counts.items())[:10]:
        print(f"  {date}: {count} записей")

print("\n" + "="*60)
print("СТАТИСТИКА ПО ЦЕНАМ ДЛЯ ТОП-5 ТИКЕРОВ")
print("="*60)

top5 = list(ticker_sorted.keys())[:5]
for ticker in top5:
    print(f"\n{ticker}:")
    ticker_data = df_swap[ticker]
    for param in ['open', 'high', 'low', 'close']:
        if param in ticker_data.columns:
            data = ticker_data[param].dropna()
            if len(data) > 0:
                print(f"  {param}: mean={data.mean():.2f}, min={data.min():.2f}, max={data.max():.2f}")

print("\n" + "="*60)
print("СТАТИСТИКА ПО ОБЪЁМАМ")
print("="*60)

volume_params = ['buy_size', 'sell_size', 'total_ask_volume', 'total_bid_volume']
for param in volume_params:
    if param in df_swap.columns.levels[1]:
        param_data = df_swap.xs(param, axis=1, level=1)
        values = param_data.values.flatten()
        values = values[~np.isnan(values)]
        if len(values) > 0:
            print(f"\n{param}:")
            print(f"  mean: {np.mean(values):.2e}")
            print(f"  median: {np.median(values):.2e}")
            print(f"  max: {np.max(values):.2e}")
            print(f"  nonzero: {(values > 0).sum()} / {len(values)}")

print("\n" + "="*60)
print("ПРОЦЕНТ ПРОПУСКОВ ДЛЯ ТОП-5 ТИКЕРОВ")
print("="*60)

for ticker in top5:
    ticker_data = df_swap[ticker]
    total_cells = ticker_data.shape[0] * ticker_data.shape[1]
    null_cells = ticker_data.isnull().sum().sum()
    null_pct = (null_cells / total_cells) * 100
    print(f"  {ticker}: {null_pct:.1f}% пропусков")

print("\n" + "="*60)
print("КОРРЕЛЯЦИЯ ПАРАМЕТРОВ (SBER)")
print("="*60)

if 'SBER' in df_swap.columns.levels[0]:
    sber_data = df_swap['SBER'][['open', 'high', 'low', 'close']].dropna()
    if len(sber_data) > 0:
        corr = sber_data.corr()
        print(corr.round(4))
else:
    print("SBER не найден, использую первый доступный тикер")
    first_ticker = top5[0]
    ticker_data = df_swap[first_ticker][['open', 'high', 'low', 'close']].dropna()
    corr = ticker_data.corr()
    print(corr.round(4))

# Сохранение
df_swap.to_parquet("market_data_multiindex.parquet")
print("\nДанные сохранены в market_data_multiindex.parquet")