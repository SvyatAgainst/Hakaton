"""
Генерация 30-минутных торговых сигналов (от -1 до +1) на основе скользящих средних.
Вход: market_data_multiindex.parquet  (создаётся парсингом)
Выход: signals.parquet
"""
import pandas as pd
import numpy as np

# ===============================
# 1. Загрузка данных из готового файла
# ===============================
DATA_FILE = "market_data_multiindex.parquet"
df_swap = pd.read_parquet(DATA_FILE)

print(f"Загружено: {df_swap.shape[0]} строк, {len(df_swap.columns.levels[0])} тикеров")

# ===============================
# 2. Ресемплинг 5 мин → 30 мин
# ===============================
def resample_close_to_30min(close_5min: pd.Series) -> pd.Series:
    """Последнее значение close в 30-минутном окне."""
    return close_5min.resample('30min').last()

# ===============================
# 3. Расчёт взвешенного сигнала (-1…+1)
# ===============================
def compute_weighted_signal(close_30min: pd.Series) -> pd.Series:
    """
    Аргументы:
        close_30min – 30-минутные цены закрытия (pd.Series с DatetimeIndex)
    Возвращает:
        сигнал от -1 (сильный SELL) до +1 (сильный BUY)
    """
    # Скользящие средние
    ma_fast = close_30min.rolling(1).mean()   # 10 * 30 мин = 5 часов
    ma_slow = close_30min.rolling(5).mean()   # 1500 мин = 25 часов

    # Дополнительные MA для множественного сигнала
    periods = [5, 10, 20, 50, 100, 200]
    mas = {p: close_30min.rolling(p).mean() for p in periods}

    # --- Сигнал 1: пересечение fast/slow ---
    signal_cross = pd.Series(0.0, index=close_30min.index)
    signal_cross[ma_fast > ma_slow] = 1.0
    signal_cross[ma_fast < ma_slow] = -1.0

    # --- Сигнал 2: множественный MA (доля MA, которые выше/ниже цены) ---
    signals_multi = pd.DataFrame(index=close_30min.index)
    for p, ma in mas.items():
        col = f'MA_{p}'
        signals_multi[col] = 0.0
        signals_multi.loc[close_30min > ma, col] = 1.0
        signals_multi.loc[close_30min < ma, col] = -1.0
    signal_multi = signals_multi.mean(axis=1)

    # --- Сигнал 3: наклон MA(20) ---
    ma_20 = mas[20]
    slope = ma_20.diff(5)
    max_abs = slope.abs().max()
    if max_abs and max_abs > 0:
        signal_slope = (slope / max_abs).clip(-1, 1).fillna(0)
    else:
        signal_slope = pd.Series(0.0, index=close_30min.index)

    # --- Сигнал 4: отклонение от MA(50) (перекупленность/перепроданность) ---
    ma_50 = mas[50]
    deviation = (close_30min - ma_50) / ma_50 * 100
    low = deviation.quantile(0.05)
    high = deviation.quantile(0.95)
    if high > low:
        signal_dist = (2 * (deviation - low) / (high - low) - 1).clip(-1, 1).fillna(0)
    else:
        signal_dist = pd.Series(0.0, index=close_30min.index)
    signal_dist = -signal_dist   # перекупленность → SELL

    # --- Взвешенное объединение ---
    weighted = (0.35 * signal_cross +
                0.25 * signal_multi +
                0.20 * signal_slope +
                0.20 * signal_dist)
    return weighted

# ===============================
# 4. Обработка всех тикеров (без фрагментации)
# ===============================
signals_dict = {}  # ticker -> pd.Series

for ticker in df_swap.columns.levels[0]:
    if 'close' not in df_swap[ticker].columns:
        continue

    close_5min = df_swap[ticker]['close'].dropna()
    if len(close_5min) < 30:
        continue

    close_30 = resample_close_to_30min(close_5min).dropna()
    if len(close_30) < 50:   # необходимо для MA(50)
        continue

    try:
        sig = compute_weighted_signal(close_30)
        signals_dict[ticker] = sig
    except Exception as e:
        print(f"Ошибка для {ticker}: {e}")

# Однократное объединение всех сигналов
signals_30min = pd.concat(signals_dict, axis=1, sort=True)
# Удаляем строки, где все значения NaN
signals_30min = signals_30min.dropna(how='all')

print(f"\nСгенерированы сигналы: {signals_30min.shape[0]} временных меток для {signals_30min.shape[1]} тикеров")
# ===============================
# 5. Сохранение в signals.parquet
# ===============================
OUTPUT_FILE = "signals.parquet"
signals_30min.to_parquet(OUTPUT_FILE)
print(f"Сигналы сохранены в {OUTPUT_FILE}")

# ===============================
# 6. Быстрый просмотр (топ-5 тикеров)
# ===============================
print("\nПример последних 5 сигналов (первые 5 тикеров):")
print(signals_30min.iloc[-5:, :5].round(3))