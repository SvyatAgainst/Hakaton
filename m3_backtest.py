import pandas as pd
import numpy as np

def backtest_signal(signal_5min: pd.DataFrame, 
                    bars_5min: pd.DataFrame,
                    threshold: float = 0.2) -> pd.DataFrame:
    """
    Бэктест сигнала с исполнением по mid цене (нулевой импакт)
    """
    
    # Подготовка данных
    bars = bars_5min.copy()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars['mid'] = bars['close']
    
    signal = signal_5min.copy()
    signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
    
    # Объединяем сигнал с ценами
    merged = signal.merge(bars, left_on=['bar_end_ts', 'seccode'], 
                          right_on=['timestamp', 'ticker'], how='inner')
    merged = merged.sort_values(['seccode', 'bar_end_ts'])
    
    # Позиция: только сигналы выше порога
    merged['pos'] = merged['value']
    merged.loc[merged['value'].abs() < threshold, 'pos'] = 0
    
    # Доходность на следующем баре
    merged['return'] = merged.groupby('seccode')['close'].pct_change().shift(-1)
    
    # PnL
    merged['pnl'] = merged['pos'] * merged['return']
    
    # Кумулятивный PnL по каждому инструменту
    merged['cum_pnl'] = merged.groupby('seccode')['pnl'].cumsum()
    
    result = merged[['bar_end_ts', 'seccode', 'pos', 'pnl', 'cum_pnl']].copy()
    
    return result

def calculate_metrics(backtest: pd.DataFrame):
    """Расчет метрик"""
    
    # Убираем NaN
    backtest = backtest.dropna(subset=['pnl'])
    
    total_pnl = backtest['pnl'].sum()
    avg_pnl = backtest['pnl'].mean()
    std_pnl = backtest['pnl'].std()
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252 * 78) if std_pnl > 0 else 0
    hit_rate = (backtest['pnl'] > 0).mean()
    num_trades = (backtest['pos'] != 0).sum()
    
    # PnL по long и short
    long_pnl = backtest[backtest['pos'] > 0]['pnl'].sum()
    short_pnl = backtest[backtest['pos'] < 0]['pnl'].sum()
    
    return {
        'total_pnl': total_pnl,
        'avg_pnl': avg_pnl,
        'sharpe_ratio': sharpe,
        'hit_rate': hit_rate,
        'num_trades': num_trades,
        'long_pnl': long_pnl,
        'short_pnl': short_pnl
    }

if __name__ == "__main__":
    print("="*60)
    print("M3: БЭКТЕСТ СИГНАЛА (BASELINE)")
    print("="*60)
    
    # Загрузка
    print("\nЗагрузка данных...")
    signal = pd.read_parquet("signal_5min.parquet")
    bars_5min = pd.read_parquet("bars_5min.parquet")
    
    print(f"Сигнал: {len(signal)} строк")
    print(f"Бары: {len(bars_5min)} строк")
    
    # Бэктест
    print("\nЗапуск бэктеста...")
    backtest = backtest_signal(signal, bars_5min, threshold=0.2)
    
    # Метрики
    metrics = calculate_metrics(backtest)
    
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ БЭКТЕСТА")
    print("="*60)
    print(f"Total PnL: {metrics['total_pnl']:.6f}")
    print(f"Средний PnL на сделку: {metrics['avg_pnl']:.6f}")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Hit Rate: {metrics['hit_rate']:.2%}")
    print(f"Количество сделок: {metrics['num_trades']}")
    print(f"PnL Long: {metrics['long_pnl']:.6f}")
    print(f"PnL Short: {metrics['short_pnl']:.6f}")
    
    # Сохраняем
    backtest.to_parquet("backtest_baseline.parquet", index=False)
    print(f"\n✅ Сохранено в backtest_baseline.parquet")
    
    # Показываем пример
    print("\n" + "="*60)
    print("ПРИМЕР СДЕЛОК")
    print("="*60)
    trades = backtest[backtest['pos'] != 0].head(15)
    print(trades.to_string())
