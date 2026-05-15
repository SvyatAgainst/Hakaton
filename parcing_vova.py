import pandas as pd
import numpy as np
from pathlib import Path

def parse_and_reshape_data():
    """
    Парсинг parquet-файлов и преобразование в удобный формат
    Оптимизированная версия - без циклов по строкам
    """
    print("="*60)
    print("ПАРСИНГ PARQUET ФАЙЛОВ (M1)")
    print("="*60)
    
    data_folder = Path("data")
    
    if not data_folder.exists():
        print(f"\n❌ Папка 'data' не найдена!")
        return None
    
    # Загружаем все файлы
    files = {
        'open': 'open_.parquet',
        'high': 'high_.parquet',
        'low': 'low_.parquet',
        'close': 'close_.parquet',
        'buy_size': 'buy_size_.parquet',
        'sell_size': 'sell_size_.parquet',
        'total_ask_volume': 'total_ask_volume_.parquet',
        'total_bid_volume': 'total_bid_volume_.parquet'
    }
    
    data_dict = {}
    
    for name, filename in files.items():
        file_path = data_folder / filename
        if file_path.exists():
            df = pd.read_parquet(file_path)
            print(f"✅ Загружен {name}: {df.shape}")
            data_dict[name] = df
        else:
            print(f"⚠️ Файл не найден: {filename}")
            return None
    
    print("\n🔄 Преобразование данных...")
    
    # Используем stack() для преобразования в длинный формат (быстро!)
    result_dfs = []
    
    for name, df in data_dict.items():
        # stack() превращает колонки в строки
        stacked = df.stack().reset_index()
        stacked.columns = ['timestamp', 'ticker', name]
        result_dfs.append(stacked)
    
    # Объединяем все данные
    print("🔄 Объединение данных...")
    merged = result_dfs[0]
    for df in result_dfs[1:]:
        merged = merged.merge(df, on=['timestamp', 'ticker'], how='inner')
    
    # Переименовываем колонки
    merged = merged.rename(columns={
        'timestamp': 'timestamp',
        'ticker': 'ticker',
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'buy_size': 'buy_size',
        'sell_size': 'sell_size',
        'total_ask_volume': 'total_ask_volume',
        'total_bid_volume': 'total_bid_volume'
    })
    
    # Удаляем строки с NaN
    before = len(merged)
    merged = merged.dropna()
    after = len(merged)
    
    print(f"\n✅ Итоговый DataFrame:")
    print(f"   Строк до удаления NaN: {before}")
    print(f"   Строк после: {after}")
    print(f"   Удалено: {before - after}")
    print(f"   Колонок: {len(merged.columns)}")
    print(f"   Инструментов: {merged['ticker'].nunique()}")
    
    if after > 0:
        print(f"   Временной диапазон: {merged['timestamp'].min()} - {merged['timestamp'].max()}")
    
    return merged

def aggregate_to_5min(df_1min):
    """
    Агрегация 1-минутных данных в 5-минутные бары (оптимизированная)
    """
    print("\n" + "="*60)
    print("АГРЕГАЦИЯ В 5-МИНУТНЫЕ БАРЫ")
    print("="*60)
    
    # Копируем
    df = df_1min.copy()
    
    # Конвертируем timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Создаем группировку по 5 минутам
    df['time_5min'] = df['timestamp'].dt.floor('5min')
    
    # Агрегируем (группируем по тикеру и 5-мин интервалу)
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'buy_size': 'sum',
        'sell_size': 'sum',
        'total_ask_volume': 'sum',
        'total_bid_volume': 'sum'
    }
    
    bars_5min = df.groupby(['ticker', 'time_5min']).agg(agg_dict).reset_index()
    bars_5min = bars_5min.rename(columns={'time_5min': 'timestamp'})
    
    print(f"\n✅ 5-минутные бары:")
    print(f"   Строк: {len(bars_5min)}")
    print(f"   Инструментов: {bars_5min['ticker'].nunique()}")
    print(f"   Временной диапазон: {bars_5min['timestamp'].min()} - {bars_5min['timestamp'].max()}")
    
    return bars_5min

def main():
    # Парсим и преобразуем данные
    print("\n🔄 Начинаем парсинг...")
    df_1min = parse_and_reshape_data()
    
    if df_1min is None or df_1min.empty:
        print("\n❌ Парсинг не удался!")
        return
    
    # Сохраняем 1-минутные данные
    df_1min.to_parquet("bars_1min.parquet", index=False)
    print(f"\n💾 Сохранено: bars_1min.parquet")
    print(f"   Размер: {len(df_1min)} строк, {len(df_1min.columns)} колонок")
    
    # Показываем пример
    print("\n" + "="*60)
    print("ПРИМЕР 1-МИНУТНЫХ ДАННЫХ")
    print("="*60)
    print(df_1min.head(10))
    
    # Агрегируем в 5-минутные бары
    df_5min = aggregate_to_5min(df_1min)
    
    # Сохраняем 5-минутные данные
    df_5min.to_parquet("bars_5min.parquet", index=False)
    print(f"\n💾 Сохранено: bars_5min.parquet")
    print(f"   Размер: {len(df_5min)} строк, {len(df_5min.columns)} колонок")
    
    # Показываем пример
    print("\n" + "="*60)
    print("ПРИМЕР 5-МИНУТНЫХ ДАННЫХ")
    print("="*60)
    print(df_5min.head(10))
    
    # Выбираем самый ликвидный инструмент для примера
    print("\n" + "="*60)
    print("СТАТИСТИКА ПО ИНСТРУМЕНТАМ")
    print("="*60)
    ticker_counts = df_5min.groupby('ticker').size().sort_values(ascending=False)
    print(f"Топ-5 инструментов по количеству баров:")
    for ticker, count in ticker_counts.head(5).items():
        print(f"   {ticker}: {count} баров")
    
    print("\n" + "="*60)
    print("✅ ГОТОВО ДЛЯ M2!")
    print("="*60)
    print("\nТеперь у вас есть:")
    print("  📁 bars_1min.parquet (для М4 - модель импакта)")
    print("  📁 bars_5min.parquet (для М2 и М3 - сигнал и бэктест)")

if __name__ == "__main__":
    main()