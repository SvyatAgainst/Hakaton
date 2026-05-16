import numpy as np
import pandas as pd

def generate_signal(df: pd.DataFrame):
    """
    Рассчитывает торговый сигнал от -1 до 1 на основе стакана, сделок и тренда.
    """
    print("Генерация сигналов (M2)...")
    
    df['close'] = df.groupby('symbol')['close'].ffill()
    df = df.fillna(0)
    
    # 1. ДИСБАЛАНС СТАКАНА
    df['obi'] = (df['bid_vol'] - df['ask_vol']) / (df['bid_vol'] + df['ask_vol'] + 1e-9)
    
    # 2. ДИСБАЛАНС СДЕЛОК
    df['trade_imb'] = (df['buy_size'] - df['sell_size']) / (df['buy_size'] + df['sell_size'] + 1e-9)
    
    # 3. ТРЕНД
    df['ma_fast'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma_slow'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(15).mean())
    df['trend'] = (df['ma_fast'] - df['ma_slow']) / (df['ma_slow'] + 1e-9)
    
    df['trend_z'] = df.groupby('symbol')['trend'].transform(
        lambda x: (x - x.rolling(30).mean()) / (x.rolling(30).std() + 1e-9)
    )
    df['trend_norm'] = np.tanh(df['trend_z'])
    
    # 4. ИТОГОВЫЙ СИГНАЛ
    df['raw_weight'] = 0.4 * df['obi'] + 0.4 * df['trade_imb'] + 0.2 * df['trend_norm']
    df['weight'] = np.clip(df['raw_weight'], -1.0, 1.0).fillna(0).round(4)
    
    output = df[["datetime", "symbol", "weight"]]
    output.to_csv("./output/m2_output.csv", index=False)

    return df