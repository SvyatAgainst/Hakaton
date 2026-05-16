import pandas as pd
import numpy as np

def load_and_prepare_data(data_dir='.', drop_empty_prices=True, fill_volumes_with_zero=True):
    """
    Загружает все 8 parquet файлов (из структуры parcing.py), объединяет их 
    в единый DataFrame и проводит глубокую очистку от NaN-значений.
    
    Параметры:
    ----------
    data_dir : str
        Путь к директории с parquet файлами.
    drop_empty_prices : bool
        Если True, удаляет строки, где цена закрытия (close) осталась NaN 
        даже после применения forward fill (например, в самом начале истории).
    fill_volumes_with_zero : bool
        Если True, заполняет пропуски в объемах и сделках нулями.
    """
    print("==================================================")
    print("ЗАПУСК СБОРА И ОЧИСТКИ ДАННЫХ")
    print("==================================================")
    
    # Полный список 8 файлов, как в вашем скрипте parcing.py
    files = {
        'open': f'{data_dir}/open_.parquet',
        'high': f'{data_dir}/high_.parquet',
        'low': f'{data_dir}/low_.parquet',
        'close': f'{data_dir}/close_.parquet',
        'buy_size': f'{data_dir}/buy_size_.parquet',
        'sell_size': f'{data_dir}/sell_size_.parquet',
        'bid_vol': f'{data_dir}/total_bid_volume_.parquet',
        'ask_vol': f'{data_dir}/total_ask_volume_.parquet'
    }
    
    dfs = []
    for col_name, filepath in files.items():
        try:
            df = pd.read_parquet(filepath)
        except FileNotFoundError:
            print(f"Ошибка: Файл {filepath} не найден. Пропустите или проверьте путь.")
            continue
            
        # Приводим индекс/колонку времени к единому стандарту
        if df.index.name != 'datetime' and 'datetime' not in df.columns:
            df.index.name = 'datetime'
        df = df.reset_index()
        
        # Переводим из широкого формата (тикеры в колонках) в длинный (строки)
        df_melted = df.melt(id_vars=['datetime'], var_name='symbol', value_name=col_name)
        dfs.append(df_melted)
    
    if not dfs:
        raise ValueError("Критическая ошибка: Не удалось загрузить ни один файл.")

    print("Слияние всех метрик в единую матрицу...")
    final_df = dfs[0]
    for df in dfs[1:]:
        final_df = pd.merge(final_df, df, on=['datetime', 'symbol'], how='outer')
    
    # Сортируем для корректной работы временных рядов и ffill
    final_df = final_df.sort_values(by=['symbol', 'datetime']).reset_index(drop=True)
    
    # ==================================================
    # БЛОК ОБРАБОТКИ И ВЫРЕЗАНИЯ NaN (Фильтрация данных)
    # ==================================================
    initial_rows = len(final_df)
    print(f"Исходное количество записей после слияния: {initial_rows:,}")
    
    # 1. Заполнение цен методом Forward Fill (протягиваем последнюю известную цену вперед)
    price_cols = ['open', 'high', 'low', 'close']
    for col in price_cols:
        if col in final_df.columns:
            final_df[col] = final_df.groupby('symbol')[col].ffill()
            
    # 2. Вырезание строк, где нет критически важных данных (цена close отсутствует)
    if drop_empty_prices and 'close' in final_df.columns:
        # Удаляем строки, где close остался NaN (обычно это самое начало торгов, до первой сделки)
        final_df = final_df.dropna(subset=['close'])
        dropped_prices = initial_rows - len(final_df)
        if dropped_prices > 0:
            print(f"  -> Удалено строк из-за отсутствия базовой цены (NaN в начале истории): {dropped_prices:,}")

    # 3. Обработка объемов и стакана
    volume_cols = ['buy_size', 'sell_size', 'bid_vol', 'ask_vol']
    if fill_volumes_with_zero:
        for col in volume_cols:
            if col in final_df.columns:
                final_df[col] = final_df[col].fillna(0.0)
        print("  -> Пропуски в объемах сделок и стаканах успешно заполнены нулями (0.0).")
        
    # 4. Считаем агрегированный рыночный объем для М4/М5
    if 'buy_size' in final_df.columns and 'sell_size' in final_df.columns:
        final_df['market_volume'] = final_df['buy_size'] + final_df['sell_size']
        
    final_rows = len(final_df)
    print(f"Итоговое количество чистых записей для бэктеста: {final_rows:,}")
    print("Данные готовы к передаче в модуль генерации сигналов M2.\n")
    
    final_df.to_csv("./output/data_loader_output.csv", index=False)
    return final_df

if __name__ == "__main__":
    # Локальный тест модуля загрузки
    # Функция автоматически подтянет все 8 файлов из текущей папки
    df_clean = load_and_prepare_data(data_dir='.')
    print(df_clean.head(10))