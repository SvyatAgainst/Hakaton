import pandas as pd
import numpy as np

def load_data():
    """
    Загружаем сигналы (результат M2) и цены закрытия (историю)
    """
    print("Загрузка данных...")
    
    # 1. Загружаем наши сигналы
    signals = pd.read_csv('signal_30min.csv')
    signals['datetime'] = pd.to_datetime(signals['datetime'])
    
    # 2. Загружаем цены из parquet
    # Преобразуем "широкую" таблицу в "длинную" (как делали в M2)
    prices_wide = pd.read_parquet('close_.parquet')
    if prices_wide.index.name != 'datetime' and 'datetime' not in prices_wide.columns:
        prices_wide.index.name = 'datetime'
    prices_wide = prices_wide.reset_index()
    prices = prices_wide.melt(id_vars=['datetime'], var_name='symbol', value_name='close')
    
    return signals, prices

def run_backtest(signals, prices):
    """
    Прогоняет идеальный бэктест и считает PnL Mid
    """
    print("Расчет идеальной доходности (бэктест)...")
    
    # Объединяем сигналы и цены в одну таблицу
    df = pd.merge(signals, prices, on=['datetime', 'symbol'], how='inner')
    df = df.sort_values(by=['symbol', 'datetime'])
    
    # ---------------------------------------------------------
    # 1. СЧИТАЕМ БУДУЩУЮ ДОХОДНОСТЬ (Forward Return)
    # ---------------------------------------------------------
    # Нам нужно узнать, как изменится цена в СЛЕДУЮЩИЕ 30 минут.
    # Метод shift(-1) берет цену из следующей строки и подтягивает к текущей.
    df['next_close'] = df.groupby('symbol')['close'].shift(-1)
    
    # Считаем процентное изменение цены за следующие 30 минут
    df['future_return'] = (df['next_close'] - df['close']) / df['close']
    
    # ---------------------------------------------------------
    # 2. СЧИТАЕМ ИДЕАЛЬНУЮ ПРИБЫЛЬ (PnL Mid)
    # ---------------------------------------------------------
    # Если weight = 1 (купили на 100%), а цена выросла на 2% -> прибыль +2%
    # Если weight = -1 (продали), а цена выросла на 2% -> убыток -2%
    # Если weight = 0.5 (купили на 50%), а цена упала на 2% -> убыток -1%
    df['pnl_mid'] = df['weight'] * df['future_return']
    
    # Удаляем последнюю строку для каждой акции, так как для нее нет "будущего"
    df = df.dropna(subset=['pnl_mid']).copy()
    
    # ---------------------------------------------------------
    # 3. АГРЕГИРУЕМ ПРИБЫЛЬ ПО ВСЕМУ ПОРТФЕЛЮ
    # ---------------------------------------------------------
    # У нас много акций. Сложим прибыль по всем акциям для каждого момента времени.
    # Если портфель нужно делить поровну на все активы, берем среднее (mean).
    # Для хакатона часто просят просто сумму PnL по всем позициям (sum).
    portfolio = df.groupby('datetime').agg(
        pnl_mid=('pnl_mid', 'sum'),
        total_exposure=('weight', lambda x: x.abs().sum()) # общий размер позиций на рынке
    ).reset_index()
    
    # Считаем накопленную прибыль (Cumulative PnL) для оценки
    portfolio['cumulative_pnl'] = portfolio['pnl_mid'].cumsum()
    
    return portfolio, df

def print_metrics(portfolio):
    """
    Выводит базовые метрики успешности стратегии
    """
    total_return = portfolio['cumulative_pnl'].iloc[-1]
    
    # Коэффициент Шарпа (Sharpe Ratio) - отношение доходности к риску (волатильности)
    # Умножаем на sqrt(кол-во 30-мин периодов в году), примерно 48 периодов в день * 252 дня = ~12000
    mean_pnl = portfolio['pnl_mid'].mean()
    std_pnl = portfolio['pnl_mid'].std()
    
    if std_pnl > 0:
        sharpe = (mean_pnl / std_pnl) * np.sqrt(12096)
    else:
        sharpe = 0
        
    print("\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА (ИДЕАЛЬНЫЕ УСЛОВИЯ) ---")
    print(f"Итоговая накопленная доходность: {total_return * 100:.2f}%")
    print(f"Коэффициент Шарпа (Annualized):  {sharpe:.2f}")
    print("-------------------------------------------------")

if __name__ == "__main__":
    # Запуск
    signals, prices = load_data()
    portfolio_results, detailed_results = run_backtest(signals, prices)
    
    print_metrics(portfolio_results)
    
    # Сохраняем в требуемый формат
    # Формат: datetime, pnl_mid, position (мы назовем position как total_exposure)
    output = portfolio_results[['datetime', 'pnl_mid', 'total_exposure']].rename(
        columns={'total_exposure': 'position'}
    )
    
    output_filename = 'backtest_baseline.csv'
    output.to_csv(output_filename, index=False)
    
    print(f"\nФрагмент файла {output_filename}:")
    print(output.head(10).to_string(index=False))