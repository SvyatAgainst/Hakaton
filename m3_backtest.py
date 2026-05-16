import pandas as pd
import numpy as np

def backtest_signal(signal_5min: pd.DataFrame, 
                    bars_5min: pd.DataFrame,
                    threshold: float = 0.2) -> pd.DataFrame:
    """
    Бэктест сигнала с исполнением по mid цене (нулевой импакт)
    """
    
    bars = bars_5min.copy()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars['mid'] = bars['close']
    
    signal = signal_5min.copy()
    signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
    
    merged = signal.merge(bars, left_on=['bar_end_ts', 'seccode'], 
                          right_on=['timestamp', 'ticker'], how='inner')
    merged = merged.sort_values(['seccode', 'bar_end_ts'])
    
    merged['pos'] = merged['value']
    merged.loc[merged['value'].abs() < threshold, 'pos'] = 0
    
    merged['return'] = merged.groupby('seccode')['close'].pct_change().shift(-1)
    merged['pnl_mid'] = merged['pos'] * merged['return']
    merged['cum_pnl_mid'] = merged.groupby('seccode')['pnl_mid'].cumsum()
    
    result = merged[['bar_end_ts', 'seccode', 'pos', 'pnl_mid', 'cum_pnl_mid']].copy()
    result['bar_end_ts'] = result['bar_end_ts'].astype('int64') // 10**9
    
    return result

if __name__ == "__main__":
    print("="*60)
    print("M3: БЭКТЕСТ СИГНАЛА (BASELINE)")
    print("="*60)
    
    print("\nЗагрузка данных...")
    signal = pd.read_parquet("data/results/signal_5min.parquet")
    bars_5min = pd.read_parquet("data/processed/bars_5min.parquet")
    
    print(f"Сигнал: {len(signal)} строк")
    print(f"Бары: {len(bars_5min)} строк")
    
    print("\nЗапуск бэктеста...")
    backtest = backtest_signal(signal, bars_5min, threshold=0.2)
    
    print(f"\nРезультаты бэктеста:")
    print(f"  Строк: {len(backtest)}")
    print(f"  Колонки: {list(backtest.columns)}")
    
    print(f"\nTotal PnL: {backtest['pnl_mid'].sum():.6f}")
    print(f"Sharpe Ratio: {backtest['pnl_mid'].mean() / backtest['pnl_mid'].std() * np.sqrt(252 * 78) if backtest['pnl_mid'].std() > 0 else 0:.2f}")
    print(f"Hit Rate: {(backtest['pnl_mid'] > 0).mean():.2%}")
    
    backtest.to_parquet("data/results/backtest_baseline.parquet", index=False)
    print(f"\n✅ Сохранено в data/results/backtest_baseline.parquet")
    
    print("\nПример первых 10 строк:")
    print(backtest.head(10))
