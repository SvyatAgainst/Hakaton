import pandas as pd
import numpy as np

def load_and_prepare_data():
    """
    Загружает parquet файлы и собирает их в единый DataFrame.
    Предполагается, что индекс файлов — это время (datetime), 
    а колонки — названия акций (symbols).
    """
    print("Загрузка данных...")
    
    # Список файлов и названий колонок, которые мы из них получим
    files = {
        'close': 'close_.parquet',
        'buy_size': 'buy_size_.parquet',
        'sell_size': 'sell_size_.parquet',
        'bid_vol': 'total_bid_volume_.parquet',
        'ask_vol': 'total_ask_volume_.parquet'
    }
    
    dfs = []
    for col_name, filename in files.items():
        # Читаем файл
        df = pd.read_parquet(filename)
        
        # Если время находится в индексе, переносим его в колонку
        if df.index.name != 'datetime' and 'datetime' not in df.columns:
            df.index.name = 'datetime'
        df = df.reset_index()
        
        # Переводим из "широкого" формата в "длинный" (datetime, symbol, value)
        df_melted = df.melt(id_vars=['datetime'], var_name='symbol', value_name=col_name)
        dfs.append(df_melted)
    
    # Объединяем все таблицы по времени и тикеру
    print("Слияние таблиц...")
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on=['datetime', 'symbol'], how='outer')
    
    final_df_sorted = final_df.sort_values(by=['symbol', 'datetime']).reset_index(drop=True)
    print(final_df_sorted)
    return final_df_sorted


def generate_signal(df):
    """
    Рассчитывает торговый сигнал от -1 до 1
    """
    print("Генерация сигнала...")
    
    # Заполняем пропуски нулями для объемов и предыдущими значениями для цены
    df['close'] = df.groupby('symbol')['close'].ffill()
    df = df.fillna(0)
    
    # ---------------------------------------------------------
    # 1. ДИСБАЛАНС СТАКАНА (Order Book Imbalance)
    # Показывает, кого больше в заявках: покупателей или продавцов
    # ---------------------------------------------------------
    # Формула: (Bids - Asks) / (Bids + Asks). Дает значение от -1 до 1.
    df['obi'] = (df['bid_vol'] - df['ask_vol']) / (df['bid_vol'] + df['ask_vol'] + 1e-9)
    
    # ---------------------------------------------------------
    # 2. ДИСБАЛАНС РЕАЛЬНЫХ СДЕЛОК (Trade Imbalance)
    # Показывает, кто агрессивнее бьет по рынку прямо сейчас
    # ---------------------------------------------------------
    df['trade_imb'] = (df['buy_size'] - df['sell_size']) / (df['buy_size'] + df['sell_size'] + 1e-9)
    
    # ---------------------------------------------------------
    # 3. ЦЕНОВОЙ ТРЕНД (Momentum)
    # ---------------------------------------------------------
    # Считаем короткую скользящую среднюю цены
    df['ma_fast'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(5).mean())
    df['ma_slow'] = df.groupby('symbol')['close'].transform(lambda x: x.rolling(15).mean())
    df['trend'] = (df['ma_fast'] - df['ma_slow']) / (df['ma_slow'] + 1e-9)
    
    # Нормализуем тренд (считаем z-score)
    df['trend_z'] = df.groupby('symbol')['trend'].transform(
        lambda x: (x - x.rolling(30).mean()) / (x.rolling(30).std() + 1e-9)
    )
    # Сжимаем тренд в диапазон [-1, 1] с помощью тангенса
    df['trend_norm'] = np.tanh(df['trend_z'])
    
    # ---------------------------------------------------------
    # 4. ИТОГОВЫЙ СИГНАЛ (WEIGHT)
    # ---------------------------------------------------------
    # Смешиваем три фактора: стакан (40%), сделки (40%) и тренд цены (20%)
    df['raw_weight'] = 0.4 * df['obi'] + 0.4 * df['trade_imb'] + 0.2 * df['trend_norm']
    
    # Финальное сглаживание: убеждаемся, что всё строго от -1 до 1, и округляем
    df['weight'] = np.clip(df['raw_weight'], -1.0, 1.0)
    df['weight'] = df['weight'].fillna(0).round(4)
    
    return df[['datetime', 'symbol', 'weight']]

if __name__ == "__main__":
    # 1. Загружаем ваши parquet-файлы
    market_data = load_and_prepare_data()
    
    # 2. Считаем веса
    signals = generate_signal(market_data)
    
    # 3. Выводим пример того, как выглядит таблица
    print("\nФрагмент готового сигнала:")
    print(signals.dropna().tail(10).to_string(index=False))
    
    # 4. Сохраняем в требуемый формат
    output_filename = 'signal_30min.csv'
    signals.to_csv(output_filename, index=False)
    print(f"\nФайл успешно сохранен: {output_filename}")