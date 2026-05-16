import pandas as pd
import numpy as np

def estimate_impact_coefficient(bars_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Оценка коэффициента импакта a(t)
    cost(x) = a * x, где x = participation_rate
    """
    
    print("="*60)
    print("M4: МОДЕЛЬ РЫНОЧНОГО ИМПАКТА")
    print("="*60)
    
    # Агрегируем данные по 1-минутным барам
    df = bars_1min.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Считаем спред как proxy для ликвидности
    # Чем шире спред - тем выше импакт
    df['spread'] = (df['high'] - df['low']) / df['close']
    
    # Объём торгов за минуту
    df['volume_mkt'] = df['total_bid_volume'] + df['total_ask_volume']
    
    # Нормированный объём (доля от среднего)
    for ticker in df['ticker'].unique():
        mask = df['ticker'] == ticker
        avg_volume = df.loc[mask, 'volume_mkt'].median()
        if avg_volume > 0:
            df.loc[mask, 'volume_norm'] = df.loc[mask, 'volume_mkt'] / avg_volume
        else:
            df.loc[mask, 'volume_norm'] = 1.0
    
    # Коэффициент импакта
    # a(t) = базовый_импакт + волатильность_фактор
    df['volatility'] = df['close'].pct_change().rolling(5).std()
    df['a'] = 0.01 + df['volatility'].fillna(0) * df['volume_norm'].fillna(1)
    
    # Клиппируем в разумные пределы [0.005, 0.05]
    df['a'] = df['a'].clip(0.005, 0.05)
    
    result = df[['timestamp', 'ticker', 'volume_mkt', 'a']].copy()
    result = result.rename(columns={'timestamp': 'bar_end_ts', 'ticker': 'seccode'})
    
    print(f"\nОценка коэффициента импакта:")
    print(f"  Средний a: {result['a'].mean():.4f}")
    print(f"  Медианный a: {result['a'].median():.4f}")
    print(f"  Диапазон a: [{result['a'].min():.4f}, {result['a'].max():.4f}]")
    
    return result

if __name__ == "__main__":
    # Загружаем 1-минутные данные
    bars_1min = pd.read_parquet("bars_1min.parquet")
    print(f"Загружено {len(bars_1min)} строк")
    
    # Оцениваем импакт
    impact_model = estimate_impact_coefficient(bars_1min)
    
    # Сохраняем
    impact_model.to_parquet("impact_model.parquet", index=False)
    print(f"\n✅ Сохранено в impact_model.parquet")
    
    print(f"\nПример:")
    print(impact_model.head(10))
