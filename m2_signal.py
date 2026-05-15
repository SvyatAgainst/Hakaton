import pandas as pd
import numpy as np

def generate_signal(bars_5min: pd.DataFrame, 
                    lookback: int = 20,
                    threshold: float = 2.0) -> pd.DataFrame:
    """
    Mean Reversion сигнал (возврат к среднему)
    Сигнал положительный когда цена ниже средней - покупаем
    Сигнал отрицательный когда цена выше средней - продаём
    """
    
    df = bars_5min.sort_values(['ticker', 'timestamp']).copy()
    signals = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if len(ticker_df) < 50:
            continue
        
        # Скользящая средняя и стандартное отклонение
        ticker_df['ma'] = ticker_df['close'].rolling(lookback).mean()
        ticker_df['std'] = ticker_df['close'].rolling(lookback).std()
        
        # Z-score отклонения от средней
        ticker_df['z_score'] = (ticker_df['close'] - ticker_df['ma']) / ticker_df['std']
        
        # Сигнал: минус tanh (возврат к среднему)
        ticker_df['value'] = -np.tanh(ticker_df['z_score'] / threshold)
        
        # Отсечка слабых сигналов
        ticker_df.loc[ticker_df['value'].abs() < 0.1, 'value'] = 0.0
        
        # Заполняем NaN и сдвигаем
        ticker_df['value'] = ticker_df['value'].fillna(0.0)
        ticker_df['value'] = ticker_df['value'].shift(1).fillna(0.0)
        
        signals.append(ticker_df[['timestamp', 'ticker', 'value']].copy())
    
    if not signals:
        return pd.DataFrame(columns=['bar_end_ts', 'seccode', 'value'])
    
    result = pd.concat(signals, ignore_index=True)
    result = result.rename(columns={
        'timestamp': 'bar_end_ts',
        'ticker': 'seccode',
        'value': 'value'
    })
    
    return result

if __name__ == "__main__":
    print("Загрузка данных...")
    bars_5min = pd.read_parquet("bars_5min.parquet")
    print(f"Загружено {len(bars_5min)} строк")
    
    print("Генерация Mean Reversion сигнала...")
    signal = generate_signal(bars_5min, lookback=20, threshold=2.0)
    
    print(f"Сигнал сохранён: {len(signal)} строк")
    print(f"Ненулевых сигналов: {(signal['value'] != 0).mean():.2%}")
    print(f"Диапазон value: [{signal['value'].min():.4f}, {signal['value'].max():.4f}]")
    print(f"\nПервые 20 строк с ненулевыми сигналами:")
    print(signal[signal['value'] != 0].head(20))
    
    signal.to_parquet("signal_5min.parquet", index=False)
    print("\n✅ Сохранено в signal_5min.parquet")