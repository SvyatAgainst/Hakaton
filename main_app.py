#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Единый пайплайн: от сырых 1-минутных данных (OHLCV + объёмы) до сигнала,
бэктеста, модели импакта, оптимизации исполнения и визуализации.

Использование:
    python full_pipeline.py --data_dir ./data --output_dir ./results

Требуемые входные файлы (в --data_dir):
    - open_.parquet
    - close_.parquet
    - high_.parquet
    - low_.parquet
    - buy_size_.parquet
    - sell_size_.parquet
    - total_ask_volume_.parquet
    - total_bid_volume_.parquet

Все файлы имеют структуру: индекс (datetime) x колонки (тикеры).
"""

import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from fastapi import FastAPI

app = FastAPI()

warnings.filterwarnings("ignore")
sns.set_style("darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)


# ----------------------------------------------------------------------
# 1. ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ
# ----------------------------------------------------------------------
def load_raw_data(data_dir: str) -> pd.DataFrame:
    """
    Загружает 8 parquet-файлов и объединяет в long-формат:
    timestamp, ticker, open, high, low, close, buy_size, sell_size,
    total_ask_volume, total_bid_volume.
    """
    filenames = {
        "open": "open_.parquet",
        "close": "close_.parquet",
        "high": "high_.parquet",
        "low": "low_.parquet",
        "buy_size": "buy_size_.parquet",
        "sell_size": "sell_size_.parquet",
        "total_ask_volume": "total_ask_volume_.parquet",
        "total_bid_volume": "total_bid_volume_.parquet",
    }

    data_frames = {}
    for name, fname in filenames.items():
        path = os.path.join(data_dir, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Не найден файл: {path}")
        df = pd.read_parquet(path)

        # Приводим к формату: индекс = timestamp, колонки = тикеры
        if isinstance(df.index, pd.MultiIndex):
            # MultiIndex (timestamp, ticker) -> wide формат
            df = df.reset_index()
            # Определяем имена уровней
            level_names = df.columns[:df.index.nlevels]
            if 'timestamp' not in level_names or 'ticker' not in level_names:
                # Предполагаем порядок: сначала timestamp, потом ticker
                df.columns = ['timestamp', 'ticker', name]
            else:
                df = df.pivot(index='timestamp', columns='ticker', values=name)
        else:
            # Индекс — timestamp, колонки — тикеры
            pass  # уже подходящий формат

        # Убедимся, что индекс — timestamp
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'timestamp' in df.columns:
                df = df.set_index('timestamp')
            else:
                df.index = pd.to_datetime(df.index)

        data_frames[name] = df

    # Объединяем все параметры по колонкам (мультииндекс: параметр, тикер)
    combined = pd.concat(data_frames, axis=1, keys=list(data_frames.keys()))
    # Меняем уровни: сначала тикер, потом параметр
    combined = combined.swaplevel(0, 1, axis=1).sort_index(axis=1)
    # Превращаем в long формат
    result = combined.stack(level=0).reset_index()
    result.rename(columns={'level_0': 'timestamp', 'level_1': 'ticker'}, inplace=True)
    result['timestamp'] = pd.to_datetime(result['timestamp'])
    result.sort_values(['ticker', 'timestamp'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    return result


def resample_to_5min(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Агрегирует 1-минутные бары в 5-минутные.
    """
    df = bars_1min.set_index("timestamp")
    resampled = df.groupby("ticker").resample("5min").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        buy_size=("buy_size", "sum"),
        sell_size=("sell_size", "sum"),
        total_ask_volume=("total_ask_volume", "sum"),
        total_bid_volume=("total_bid_volume", "sum"),
    ).reset_index()
    resampled.rename(columns={"level_1": "timestamp"}, inplace=True)
    return resampled


# ----------------------------------------------------------------------
# 2. ГЕНЕРАЦИЯ СИГНАЛА (MEAN REVERSION)
# ----------------------------------------------------------------------
def generate_signal(bars_5min: pd.DataFrame,
                    lookback: int = 30,
                    threshold: float = 2.0) -> pd.DataFrame:
    """
    Mean reversion сигнал с использованием Z-score и -tanh.
    Возвращает DataFrame с колонками: bar_end_ts, seccode, value.
    """
    df = bars_5min.sort_values(["ticker", "timestamp"]).copy()
    signals = []

    for ticker in df["ticker"].unique():
        ticker_df = df[df["ticker"] == ticker].copy()
        if len(ticker_df) < 50:
            continue

        ticker_df["ma"] = ticker_df["close"].rolling(lookback).mean()
        ticker_df["std"] = ticker_df["close"].rolling(lookback).std()
        ticker_df["z_score"] = (ticker_df["close"] - ticker_df["ma"]) / ticker_df["std"]
        ticker_df["value"] = -np.tanh(ticker_df["z_score"] / threshold)
        ticker_df.loc[ticker_df["value"].abs() < 0.1, "value"] = 0.0
        ticker_df["value"] = ticker_df["value"].fillna(0.0)
        ticker_df["value"] = ticker_df["value"].shift(1).fillna(0.0)   # избегаем look-ahead

        signals.append(ticker_df[["timestamp", "ticker", "value"]].copy())

    if not signals:
        return pd.DataFrame(columns=["bar_end_ts", "seccode", "value"])

    result = pd.concat(signals, ignore_index=True)
    result.rename(columns={"timestamp": "bar_end_ts", "ticker": "seccode"}, inplace=True)
    return result


# ----------------------------------------------------------------------
# 3. БЭКТЕСТ БЕЗ ИМПАКТА (MID PRICE)
# ----------------------------------------------------------------------
def backtest_signal(signal_5min: pd.DataFrame,
                    bars_5min: pd.DataFrame,
                    threshold: float = 0.2,
                    aum: float = 1_000_000) -> pd.DataFrame:
    """
    Бэктест сигнала по mid-цене (close). Возвращает детализированный DataFrame
    с PnL в денежном выражении.
    """
    bars = bars_5min.copy()
    bars["timestamp"] = pd.to_datetime(bars["timestamp"])
    bars["mid"] = bars["close"]

    signal = signal_5min.copy()
    signal["bar_end_ts"] = pd.to_datetime(signal["bar_end_ts"])

    merged = signal.merge(bars,
                          left_on=["bar_end_ts", "seccode"],
                          right_on=["timestamp", "ticker"],
                          how="inner")
    merged = merged.sort_values(["seccode", "bar_end_ts"])

    merged["pos"] = merged["value"]
    merged.loc[merged["value"].abs() < threshold, "pos"] = 0

    merged["return"] = merged.groupby("seccode")["close"].pct_change().shift(-1)
    merged["pnl_mid"] = merged["pos"] * merged["return"] * aum   # денежное выражение
    merged["cum_pnl_mid"] = merged.groupby("seccode")["pnl_mid"].cumsum()

    result = merged[["bar_end_ts", "seccode", "pos", "pnl_mid", "cum_pnl_mid"]].copy()
    return result


# ----------------------------------------------------------------------
# 4. МОДЕЛЬ РЫНОЧНОГО ИМПАКТА (константа)
# ----------------------------------------------------------------------
def create_impact_model(bars_1min: pd.DataFrame, base_impact: float = 0.01) -> pd.DataFrame:
    """
    Создаёт модель импакта с постоянным коэффициентом a.
    Возвращает DataFrame с колонками: bar_end_ts, seccode, a.
    """
    df = bars_1min.copy()
    df["volume_mkt"] = df["total_bid_volume"] + df["total_ask_volume"]
    df["a"] = base_impact
    result = df[["timestamp", "ticker", "a"]].copy()
    result.rename(columns={"timestamp": "bar_end_ts", "ticker": "seccode"}, inplace=True)
    return result


# ----------------------------------------------------------------------
# 5. ОПТИМИЗАЦИЯ ИСПОЛНЕНИЯ (динамический participation)
# ----------------------------------------------------------------------
def optimize_execution_profitable(signal_5min: pd.DataFrame,
                                  bars_1min: pd.DataFrame,
                                  impact_model: pd.DataFrame,
                                  aum: float = 1_000_000,
                                  participation_max: float = 0.3) -> pd.DataFrame:
    """
    Исполнение сигналов с динамическим участием в рынке.
    Возвращает DataFrame со сделками и PnL net.
    """
    print("=" * 60)
    print("M5: ОПТИМИЗАЦИЯ ИСПОЛНЕНИЯ (динамический participation)")
    print("=" * 60)

    signal = signal_5min.copy()
    signal["bar_end_ts"] = pd.to_datetime(signal["bar_end_ts"])
    bars_1min["timestamp"] = pd.to_datetime(bars_1min["timestamp"])
    impact_model["bar_end_ts"] = pd.to_datetime(impact_model["bar_end_ts"])

    # Объём рынка на 1-минутках
    bars_1min["volume_mkt"] = bars_1min["total_bid_volume"] + bars_1min["total_ask_volume"]

    # Агрегируем 1-минутки в 5-минутные бары (сохраняем open, close, volume_mkt)
    bars_5min_agg = bars_1min.groupby([bars_1min["ticker"], bars_1min["timestamp"].dt.floor("5min")]).agg({
        "open": "first",
        "close": "last",
        "volume_mkt": "sum"
    }).reset_index()
    bars_5min_agg = bars_5min_agg.rename(columns={"timestamp": "bar_end_ts", "ticker": "seccode"})

    # Берём только активные сигналы (value != 0)
    active = signal[signal["value"] != 0].copy()
    print(f"Активных сигналов: {len(active)}")

    if len(active) == 0:
        return pd.DataFrame()

    # Объединяем сигналы с 5-минутными данными
    merged = active.merge(bars_5min_agg, on=["bar_end_ts", "seccode"], how="inner")
    merged = merged.sort_values(["seccode", "bar_end_ts"])

    # Берём следующий бар для расчёта доходности
    merged["next_open"] = merged.groupby("seccode")["open"].shift(-1)
    merged["next_close"] = merged.groupby("seccode")["close"].shift(-1)
    merged = merged.dropna()

    # PnL mid (денежное выражение)
    merged["return"] = (merged["next_close"] - merged["next_open"]) / merged["next_open"]
    merged["pnl_mid"] = merged["value"] * merged["return"] * aum

    # Добавляем коэффициент импакта
    merged = merged.merge(impact_model[["bar_end_ts", "seccode", "a"]],
                          on=["bar_end_ts", "seccode"], how="left")
    merged["a"] = merged["a"].fillna(0.01)

    # Динамический participation
    merged["our_volume"] = abs(merged["value"]) * aum / merged["next_open"]   # в штуках
    merged["participation"] = merged["our_volume"] / merged["volume_mkt"]
    merged["participation"] = merged["participation"].clip(upper=participation_max)

    # Стоимость импакта в деньгах
    merged["impact_cost"] = merged["a"] * merged["participation"] * abs(merged["value"]) * aum

    # Итоговый PnL
    merged["pnl_net"] = merged["pnl_mid"] - merged["impact_cost"]

    print(f"\nРезультаты исполнения:")
    print(f"  Сделок: {len(merged)}")
    print(f"  Суммарный PnL mid: {merged['pnl_mid'].sum():.2f} руб.")
    print(f"  Суммарный импакт: {merged['impact_cost'].sum():.2f} руб.")
    print(f"  Суммарный PnL net: {merged['pnl_net'].sum():.2f} руб.")
    print(f"  Win Rate (net): {(merged['pnl_net'] > 0).mean():.2%}")

    print(f"\nСтатистика participation:")
    print(f"  Средний: {merged['participation'].mean():.4%}")
    print(f"  Медианный: {merged['participation'].median():.4%}")
    print(f"  Максимальный: {merged['participation'].max():.4%}")

    return merged


# ----------------------------------------------------------------------
# 6. ВИЗУАЛИЗАЦИЯ
# ----------------------------------------------------------------------
def plot_results(backtest_mid: pd.DataFrame,
                 exec_summary: pd.DataFrame,
                 signal: pd.DataFrame,
                 output_dir: str):
    """Строит и сохраняет основные графики."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Кумулятивный PnL (mid и net)
    fig, ax = plt.subplots(figsize=(12, 6))
    # Суммируем PnL по времени для mid
    mid_pnl = backtest_mid.groupby("bar_end_ts")["pnl_mid"].sum().cumsum()
    mid_pnl.index = pd.to_datetime(mid_pnl.index)
    mid_pnl.plot(ax=ax, label="Mid PnL (без импакта)", color="blue")

    if len(exec_summary) > 0:
        net_pnl = exec_summary.groupby("bar_end_ts")["pnl_net"].sum().cumsum()
        net_pnl.index = pd.to_datetime(net_pnl.index)
        net_pnl.plot(ax=ax, label="Net PnL (с импактом)", color="red")
        ax.axhline(0, color="black", linestyle="--", alpha=0.5)
        ax.set_title("Кумулятивный PnL")
        ax.set_xlabel("Дата")
        ax.set_ylabel("PnL (руб)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "cumulative_pnl.png"), dpi=150)
        plt.close()

    # 2. Распределение дневных PnL (net)
    if len(exec_summary) > 0:
        exec_summary["date"] = pd.to_datetime(exec_summary["bar_end_ts"]).dt.date
        daily_pnl = exec_summary.groupby("date")["pnl_net"].sum()
        fig, ax = plt.subplots(figsize=(10, 5))
        daily_pnl.hist(bins=30, alpha=0.7, color="green", edgecolor="black")
        ax.axvline(0, color="red", linestyle="--")
        ax.set_title("Распределение дневного Net PnL")
        ax.set_xlabel("PnL (руб)")
        ax.set_ylabel("Частота")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "daily_pnl_dist.png"), dpi=150)
        plt.close()

    # 3. Временной ряд сигналов (активность)
    signal_activity = signal[signal["value"] != 0].copy()
    if len(signal_activity) > 0:
        signal_activity["date"] = pd.to_datetime(signal_activity["bar_end_ts"]).dt.date
        daily_activity = signal_activity.groupby("date").size()
        fig, ax = plt.subplots(figsize=(12, 4))
        daily_activity.plot(ax=ax, marker="o", linestyle="-", alpha=0.7)
        ax.set_title("Количество активных сигналов по дням")
        ax.set_ylabel("Количество")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "signal_activity.png"), dpi=150)
        plt.close()

    # 4. Тепловая карта корреляции между доходностями активов (mid PnL)
    if len(backtest_mid["seccode"].unique()) > 1:
        # Преобразуем timestamp в datetime для pivot
        backtest_mid_copy = backtest_mid.copy()
        backtest_mid_copy["bar_end_ts"] = pd.to_datetime(backtest_mid_copy["bar_end_ts"])
        pnl_pivot = backtest_mid_copy.pivot(index="bar_end_ts", columns="seccode", values="pnl_mid")
        corr = pnl_pivot.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Корреляция mid PnL по тикерам")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "pnl_correlation.png"), dpi=150)
        plt.close()

    print(f"Графики сохранены в {output_dir}")


# ----------------------------------------------------------------------
# 7. MAIN PIPELINE
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Полный пайплайн торговой стратегии")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Папка с входными parquet-файлами (open_, close_, ...)")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Папка для сохранения результатов и графиков")
    parser.add_argument("--lookback", type=int, default=30,
                        help="Период скользящего среднего для сигнала")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Порог Z-score для сигнала")
    parser.add_argument("--aum", type=float, default=1_000_000,
                        help="Размер портфеля (AUM)")
    parser.add_argument("--participation_max", type=float, default=0.3,
                        help="Максимальная доля участия в минуту")
    parser.add_argument("--base_impact", type=float, default=0.01,
                        help="Базовый коэффициент импакта (константа)")
    args = parser.parse_args()

    main_app_func(args.data_dir, args.output_dir, args.lookback, args.threshold, args.aum, args.participation_max)

def main_app_func(data_dir: str = "./data", output_dir: str = "./results", lookback: int = 30, threshold: float = 2.0, aum: float = 1_000_000, participation_max: float = 0.3, base_impact: float = 0.01):

    print("=" * 70)
    print("ЗАПУСК ПОЛНОГО ПАЙПЛАЙНА")
    print("=" * 70)

    # Шаг 1: загрузка сырых данных (1-минутные)
    print("\n1. Загрузка сырых 1-минутных данных...")
    raw_df = load_raw_data(data_dir)
    print(f"   Загружено {len(raw_df)} записей (timestamp, ticker)")

    # Шаг 2: ресемплинг в 5-минутные бары
    print("\n2. Ресемплинг в 5-минутные бары...")
    bars_5min = resample_to_5min(raw_df)
    print(f"   Получено {len(bars_5min)} 5-минутных баров")

    # Шаг 3: генерация сигнала
    print("\n3. Генерация mean-reversion сигнала...")
    signal = generate_signal(bars_5min, lookback=lookback, threshold=threshold)
    print(f"   Сигнал: {len(signal)} строк, ненулевых: {(signal['value'] != 0).mean():.2%}")

    # Шаг 4: бэктест без импакта
    print("\n4. Бэктест по mid цене...")
    backtest_mid = backtest_signal(signal, bars_5min, threshold=0.2, aum=aum)
    print(f"   Total PnL mid: {backtest_mid['pnl_mid'].sum():.2f} руб.")

    # Шаг 5: создание модели импакта (константа)
    print("\n5. Создание модели импакта (константа a={})...".format(base_impact))
    impact_model = create_impact_model(raw_df, base_impact=base_impact)
    print(f"   Модель импакта: {len(impact_model)} записей, a = {impact_model['a'].iloc[0]}")

    # Шаг 6: оптимизация исполнения с динамическим participation
    print("\n6. Оптимизация исполнения с учётом импакта...")
    exec_summary = optimize_execution_profitable(
        signal, raw_df, impact_model,
        aum=aum,
        participation_max=participation_max
    )
    if len(exec_summary) > 0:
        print(f"   Net PnL: {exec_summary['pnl_net'].sum():.2f} руб.")
    else:
        print("   Не удалось исполнить ни одного сигнала.")

    # Шаг 7: визуализация
    print("\n7. Построение графиков...")
    plot_results(backtest_mid, exec_summary, signal, output_dir)

    # Сохраняем ключевые результаты
    os.makedirs(output_dir, exist_ok=True)
    signal.to_parquet(os.path.join(output_dir, "signal_5min.parquet"), index=False)
    backtest_mid.to_parquet(os.path.join(output_dir, "backtest_mid.parquet"), index=False)
    impact_model.to_parquet(os.path.join(output_dir, "impact_model.parquet"), index=False)
    if len(exec_summary) > 0:
        exec_summary.to_parquet(os.path.join(output_dir, "execution_summary.parquet"), index=False)

    print("\n✅ Пайплайн успешно завершён. Результаты и графики сохранены в:", output_dir)

@app.get("/main-app")
async def main_app_func_api():
    main_app_func(data_dir="./uploads", output_dir="./download")
    return {"status": "success"}

# if __name__ == "__main__":
#     main()