import pandas as pd
import numpy as np

def generate_signal(bars_5min: pd.DataFrame, 
                    lookback: int = 20,
                    ema_span: int = 5,
                    signal_threshold: float = 0.2) -> pd.DataFrame:
    
    df = bars_5min.sort_values(['ticker', 'timestamp']).copy()
    signals = []
    
    for ticker in df['ticker'].unique():
        ticker_df = df[df['ticker'] == ticker].copy()
        
        if len(ticker_df) < 70:
            continue
        
        ticker_df['roc'] = ticker_df['close'].pct_change(lookback)
        ticker_df = ticker_df.dropna(subset=['roc'])
        
        if len(ticker_df) < 50:
            continue
        
        roc_mean = ticker_df['roc'].rolling(50).mean()
        roc_std = ticker_df['roc'].rolling(50).std()
        ticker_df['signal_raw'] = (ticker_df['roc'] - roc_mean) / roc_std
        ticker_df = ticker_df.dropna(subset=['signal_raw'])
        
        ticker_df['signal_smoothed'] = ticker_df['signal_raw'].ewm(span=ema_span, adjust=False).mean()
        ticker_df['value'] = np.tanh(ticker_df['signal_smoothed'])
        ticker_df.loc[ticker_df['value'].abs() < signal_threshold, 'value'] = 0.0
        ticker_df['value'] = ticker_df['value'].shift(1)
        ticker_df['value'] = ticker_df['value'].fillna(0.0)
        
        signals.append(ticker_df[['timestamp', 'ticker', 'value']].copy())
    
    if not signals:
        return pd.DataFrame(columns=['bar_end_ts', 'seccode', 'value'])
    
    result = pd.concat(signals, ignore_index=True)
    result = result.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode', 'value': 'value'})
    return result

if __name__ == "__main__":
    print("Загрузка данных...")
    bars_5min = pd.read_parquet("bars_5min.parquet")
    print(f"Загружено {len(bars_5min)} строк")
    
    print("Генерация сигнала...")
    signal = generate_signal(bars_5min)
    
    print(f"Сигнал сохранён: {len(signal)} строк")
    print(f"Ненулевых сигналов: {(signal['value'] != 0).mean():.2%}")
    print(f"Диапазон value: [{signal['value'].min():.4f}, {signal['value'].max():.4f}]")
    print(f"\nПервые 20 строк:")
    print(signal.head(20))
    
    signal.to_parquet("signal_5min.parquet", index=False)
    print("\n✅ Сохранено в signal_5min.parquet")
