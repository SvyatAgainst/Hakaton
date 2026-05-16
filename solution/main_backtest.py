import pandas as pd
import numpy as np
from data_loader import load_and_prepare_data
from m2_signals import generate_signal
from m4_m5_execution import MarketImpactModel, TWAPExecutor

# Настройки бэктеста
PORTFOLIO_CAPITAL = 1_000_000  # Допустим, мы торгуем капиталом 1 млн $ на один актив
EXECUTION_MINUTES = 30         # Окно исполнения TWAP (в минутах)

def run_realistic_backtest(df: pd.DataFrame):
    print("\nЗапуск реалистичного бэктеста с учетом Market Impact (M4/M5)...")
    
    impact_model = MarketImpactModel(a_coef=0.03)
    twap_executor = TWAPExecutor(impact_model=impact_model, num_slices=EXECUTION_MINUTES)
    
    # 1. Считаем будущую цену (на 30 минут вперед) для расчета идеального PnL Mid
    df = df.sort_values(by=['symbol', 'datetime'])
    df['next_close'] = df.groupby('symbol')['close'].shift(-EXECUTION_MINUTES)
    df['future_return'] = (df['next_close'] - df['close']) / df['close']
    df['pnl_mid'] = df['weight'] * df['future_return']
    
    # Удаляем концы, где нет будущего (нельзя посчитать PnL и исполнить TWAP)
    df = df.dropna(subset=['next_close']).copy()

    # Списки для сохранения результатов исполнения
    total_is_list = []
    vwap_fills = []

    print("Симуляция исполнения ордеров (это может занять время)...")
    
    # Проходим по всем строкам (для продакшена это оптимизируют, но для теста пойдет)
    # Группируем по тикеру, чтобы было проще "заглядывать в будущее"
    for symbol, group in df.groupby('symbol'):
        group = group.reset_index(drop=True)
        prices = group['close'].values
        market_vols = group['market_volume'].values
        weights = group['weight'].values
        
        for i in range(len(group)):
            # Если сигнала нет, пропускаем
            if weights[i] == 0:
                total_is_list.append(0.0)
                vwap_fills.append(prices[i])
                continue
                
            # Переводим вес [-1; 1] в количество акций для покупки/продажи
            # Объем = (Доля капитала) / Цена
            target_q = (weights[i] * PORTFOLIO_CAPITAL) / (prices[i] + 1e-9)
            
            # Собираем данные стакана на следующие 30 минут для M5
            # Срез данных: от текущей минуты i до i + 30
            end_idx = min(i + EXECUTION_MINUTES, len(group))
            
            minute_bars_real = []
            for j in range(i, end_idx):
                minute_bars_real.append({
                    'price': prices[j],
                    'market_volume': market_vols[j]
                })
            
            # Запускаем алгоритм исполнения M5
            exec_res = twap_executor.execute_order(target_q, minute_bars_real)
            
            total_is_list.append(exec_res['total_is'])
            vwap_fills.append(exec_res['vwap_fill'])

    # Добавляем результаты исполнения обратно в DataFrame
    df['implementation_shortfall_usd'] = total_is_list
    df['vwap_fill'] = vwap_fills
    
    # 2. СЧИТАЕМ РЕАЛИСТИЧНЫЙ PNL (PnL Post)
    # Штраф за исполнение переводим в проценты (IS_USD / Выделенный капитал)
    df['is_penalty_pct'] = df['implementation_shortfall_usd'] / PORTFOLIO_CAPITAL
    
    # PnL Post = Идеальный PnL - Штраф за влияние на рынок
    df['pnl_post'] = df['pnl_mid'] - df['is_penalty_pct']

    # 3. Агрегация портфеля
    portfolio = df.groupby('datetime').agg(
        pnl_mid=('pnl_mid', 'sum'),
        pnl_post=('pnl_post', 'sum'),
        total_exposure=('weight', lambda x: x.abs().sum()),
        total_is_usd=('implementation_shortfall_usd', 'sum')
    ).reset_index()
    
    portfolio['cumulative_pnl_mid'] = portfolio['pnl_mid'].cumsum()
    portfolio['cumulative_pnl_post'] = portfolio['pnl_post'].cumsum()
    
    portfolio.to_csv("./output/backtest_output.csv", index=False)

    return portfolio, df

def print_metrics(portfolio):
    print("\n--- РЕЗУЛЬТАТЫ БЭКТЕСТА ---")
    print(f"Накопленная ИДЕАЛЬНАЯ доходность (PnL Mid): {portfolio['cumulative_pnl_mid'].iloc[-1] * 100:.2f}%")
    print(f"Накопленная РЕАЛЬНАЯ доходность (PnL Post):  {portfolio['cumulative_pnl_post'].iloc[-1] * 100:.2f}%")
    
    mean_post = portfolio['pnl_post'].mean()
    std_post = portfolio['pnl_post'].std()
    sharpe = (mean_post / std_post) * np.sqrt(12096) if std_post > 0 else 0
    print(f"Коэффициент Шарпа (PnL Post, Ann.):        {sharpe:.2f}")
    print(f"Суммарные потери на Market Impact ($):       ${portfolio['total_is_usd'].sum():,.2f}")
    print("---------------------------")

if __name__ == "__main__":
    # 1. Загружаем данные (теперь одной строкой)
    market_data = load_and_prepare_data(data_dir='./data') # Укажи путь к папке с паркетами, если они не в текущей
    
    # 2. Генерируем сигналы
    market_data = generate_signal(market_data)
    
    # 3. Запускаем интегрированный бэктест
    portfolio_results, detailed_results = run_realistic_backtest(market_data)
    
    # 4. Выводим метрики
    print_metrics(portfolio_results)
    
    # 5. Сохраняем результат
    output = portfolio_results[['datetime', 'pnl_mid', 'pnl_post', 'total_exposure']]
    output.to_csv('backtest_realistic.csv', index=False)
    print("\nСохранен файл: backtest_realistic.csv")