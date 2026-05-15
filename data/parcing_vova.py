import pandas as pd
import numpy as np
from pathlib import Path
import os

def parse_and_merge_data(data_path: str = "Hakaton/data") -> pd.DataFrame:
    """
    Парсинг всех parquet-файлов и объединение в один DataFrame
    
    Parameters:
    -----------
    data_path : str
        Путь к папке с данными
    
    Returns:
    --------
    pd.DataFrame
        Объединённый DataFrame со всеми данными
    """
    
    # Словарь для хранения загруженных данных
    data_frames = {}
    
    # Все файлы, которые нужно загрузить
    files = {
        'buy_size': 'buy_size_parquet',
        'sell_size': 'sell_size_parquet',
        'open': 'open_parquet',
        'high': 'high_parquet',
        'low': 'low_parquet',
        'close': 'close_parquet',
        'total_ask_volume': 'total_ask_volume_parquet',
        'total_bid_volume': 'total_bid_volume_parquet'
    }
    
    # Загружаем каждый файл
    for name, filename in files.items():
        file_path = Path(data_path) / filename
        
        if file_path.exists():
            print(f"Загружаю {filename}...")
            
            # Загружаем parquet
            df = pd.read_parquet(file_path)
            
            # Удаляем полностью пустые строки и столбцы
            df = df.dropna(how='all')  # строки, где все значения NaN
            df = df.dropna(axis=1, how='all')  # столбцы, где все значения NaN
            
            data_frames[name] = df
            print(f"  - Размер: {df.shape}")
            print(f"  - Null значений: {df.isnull().sum().sum()}")
        else:
            print(f"Файл не найден: {file_path}")
    
    # Объединяем все DataFrame по индексу (предполагается одинаковый индекс)
    if data_frames:
        # Проверяем, что индексы совпадают
        first_df = list(data_frames.values())[0]
        merged_df = pd.DataFrame(index=first_df.index)
        
        for name, df in data_frames.items():
            # Добавляем колонки с префиксом
            for col in df.columns:
                merged_df[f'{name}_{col}'] = df[col]
        
        # Удаляем строки, где есть хотя бы один NaN (очистка)
        before_count = len(merged_df)
        merged_df = merged_df.dropna()
        after_count = len(merged_df)
        
        print(f"\nПосле удаления NaN:")
        print(f"  - Было строк: {before_count}")
        print(f"  - Стало строк: {after_count}")
        print(f"  - Удалено: {before_count - after_count} строк")
        
        return merged_df
    else:
        print("Не загружено ни одного файла!")
        return pd.DataFrame()

# Альтернативный вариант, если файлы имеют одинаковую структуру
def parse_and_merge_alternative(data_path: str = "Hakaton/data") -> pd.DataFrame:
    """
    Альтернативный парсинг, если все файлы имеют одинаковую структуру
    и нужно объединить их по timestamp
    """
    
    data_path = Path(data_path)
    
    # Находим все parquet файлы
    all_files = list(data_path.glob("*_parquet"))
    
    if not all_files:
        all_files = list(data_path.glob("*.parquet"))
    
    print(f"Найдено файлов: {len(all_files)}")
    
    merged_data = {}
    
    for file_path in all_files:
        print(f"\nОбработка: {file_path.name}")
        
        # Загружаем файл
        df = pd.read_parquet(file_path)
        print(f"  Размер: {df.shape}")
        
        # Удаляем пустые строки
        df = df.dropna(how='all')
        
        # Определяем тип данных по имени файла
        file_name = file_path.stem.replace('_parquet', '')
        
        # Если это single column файл (как на скриншоте)
        if len(df.columns) == 1:
            col_name = df.columns[0]
            merged_data[file_name] = df[col_name]
        else:
            # Если несколько колонок
            for col in df.columns:
                merged_data[f'{file_name}_{col}'] = df[col]
    
    # Объединяем все в один DataFrame
    result_df = pd.DataFrame(merged_data)
    
    # Удаляем строки с NaN
    before = len(result_df)
    result_df = result_df.dropna()
    after = len(result_df)
    
    print(f"\nИтоговый результат:")
    print(f"  - Размер: {result_df.shape}")
    print(f"  - Удалено строк с NaN: {before - after}")
    print(f"  - Колонки: {list(result_df.columns)}")
    
    return result_df

# Функция для сохранения результата
def save_merged_data(df: pd.DataFrame, output_path: str = "merged_data.parquet"):
    """
    Сохранение объединённых данных
    """
    df.to_parquet(output_path, index=True)
    print(f"\nДанные сохранены в {output_path}")
    
    # Также сохраняем в CSV для удобного просмотра (первые 1000 строк)
    csv_path = output_path.replace('.parquet', '_sample.csv')
    df.head(1000).to_csv(csv_path)
    print(f"Сэмпл сохранён в {csv_path}")

# Основная функция
def main():
    # Укажите правильный путь к данным
    # Варианты:
    # data_path = "Hakaton/data"  
    # data_path = "data"          
    # data_path = "."             
    
    data_path = "Hakaton/data"  # ← ИЗМЕНИТЕ ПРИ НЕОБХОДИМОСТИ
    
    print("=" * 50)
    print("ПАРСИНГ PARQUET ФАЙЛОВ")
    print("=" * 50)
    
    # Парсим данные
    merged_df = parse_and_merge_alternative(data_path)
    
    if not merged_df.empty:
        print("\n" + "=" * 50)
        print("СТАТИСТИКА ПО ДАННЫМ")
        print("=" * 50)
        print(f"Всего строк: {len(merged_df)}")
        print(f"Всего колонок: {len(merged_df.columns)}")
        print(f"\nПервые 5 строк:")
        print(merged_df.head())
        print(f"\nТипы данных:")
        print(merged_df.dtypes)
        print(f"\nСтатистика по null (должно быть 0):")
        print(merged_df.isnull().sum())
        
        # Сохраняем результат
        save_merged_data(merged_df)
    else:
        print("Не удалось загрузить данные!")

if __name__ == "__main__":
    main()