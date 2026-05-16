import numpy as np
import pandas as pd

# --- Классы M4 (MarketImpactModel) и M5 (TWAPExecutor) остаются без изменений ---
# ... (вставьте их сюда из предыдущего ответа) ...

class MarketImpactModel:
    def __init__(self, a_coef=0.03):
        self.a = a_coef

    def get_participation_rate(self, my_volume, market_volume):
        if market_volume <= 0 or pd.isna(market_volume):
            return 0.0
        return abs(my_volume) / market_volume

    def calculate_impact_cost(self, x):
        return self.a * x

    def get_execution_price(self, ideal_price, my_volume, market_volume):
        if my_volume == 0 or market_volume <= 0 or pd.isna(market_volume):
            return ideal_price, 0.0
        x = self.get_participation_rate(my_volume, market_volume)
        cost_fraction = self.calculate_impact_cost(x)
        direction = np.sign(my_volume)
        real_price = ideal_price * (1 + (direction * cost_fraction))
        return real_price, cost_fraction

class TWAPExecutor:
    def __init__(self, impact_model: MarketImpactModel, num_slices=30):
        self.impact_model = impact_model
        self.K = num_slices

    def execute_order(self, target_volume, minute_bars_data):
        if target_volume == 0 or len(minute_bars_data) == 0:
            return {"vwap_fill": 0.0, "total_is": 0.0, "executed_volume": 0.0}

        slice_volume = target_volume / self.K
        total_value_traded = 0.0  
        total_is = 0.0            
        executed_abs_volume = 0.0 

        for k_data in minute_bars_data:
            ideal_price = k_data['price']
            market_vol_1min = k_data['market_volume']
            
            real_price, cost_fraction = self.impact_model.get_execution_price(
                ideal_price=ideal_price, 
                my_volume=slice_volume, 
                market_volume=market_vol_1min
            )
            
            slice_is = abs(slice_volume) * ideal_price * cost_fraction
            
            total_value_traded += abs(slice_volume) * real_price
            total_is += slice_is
            executed_abs_volume += abs(slice_volume)

        vwap_fill = total_value_traded / executed_abs_volume if executed_abs_volume > 0 else 0.0

        return {
            "vwap_fill": vwap_fill,
            "total_is": total_is,
            "target_volume": target_volume
        }

# ---------------------------------------------------------
# ИНТЕГРАЦИЯ С РЕАЛЬНЫМИ ДАННЫМИ (.parquet)
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Загрузка данных из parquet файлов...")
    
    # 1. Загружаем нужные датафреймы
    # Обычно в таких данных индексом выступает DateTime, а колонками - тикеры (инструменты)
    try:
        df_close = pd.read_parquet('close_.parquet')
        df_buy = pd.read_parquet('buy_size_.parquet')
        df_sell = pd.read_parquet('sell_size_.parquet')
    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл. Убедитесь, что parquet-файлы лежат в папке со скриптом.\n{e}")
        exit()

    # 2. Выбираем инструмент (тикер) и временное окно
    # Для теста возьмем первый попавшийся тикер и первые 30 минут (индексов)
    ticker = df_close.columns[0]
    
    # Предполагаем, что данные отсортированы по времени. Берем первые 30 минут.
    # В реальном бэктесте (М3) этот срез будет передаваться динамически для каждого бара.
    slice_start = 0
    slice_end = 30
    
    # 3. Формируем список словарей minute_bars_data для M5
    minute_bars_real = []
    for i in range(slice_start, slice_end):
        # Цена закрытия за эту минуту
        price = df_close[ticker].iloc[i]
        
        # Общий объем рынка за минуту (покупки + продажи). 
        # Если данных нет (NaN), считаем объем нулевым.
        buy_vol = df_buy[ticker].iloc[i]
        sell_vol = df_sell[ticker].iloc[i]
        
        buy_vol = 0 if pd.isna(buy_vol) else buy_vol
        sell_vol = 0 if pd.isna(sell_vol) else sell_vol
        market_vol = buy_vol + sell_vol
        
        minute_bars_real.append({
            'price': price,
            'market_volume': market_vol
        })

    # 4. Инициализируем модели
    impact_model = MarketImpactModel(a_coef=0.03)
    twap_executor = TWAPExecutor(impact_model=impact_model, num_slices=30)
    
    # 5. Задаем сигнал (например, из M2 пришел сигнал КУПИТЬ 500 лотов/акций за эти 30 минут)
    target_q = 500.0 
    
    # 6. Запускаем исполнение на реальных данных
    execution_results = twap_executor.execute_order(
        target_volume=target_q, 
        minute_bars_data=minute_bars_real
    )
    
    print("\n--- Результаты работы модуля M5 (TWAP) на реальных данных ---")
    print(f"Инструмент: {ticker}")
    print(f"Целевой объем (Q_s): {execution_results['target_volume']} шт.")
    print(f"Объем исполнения на 1 минуту (q_k): {target_q/30:.2f} шт.")
    print(f"Средняя реальная цена покупки (VWAP fill): {execution_results['vwap_fill']:.4f}")
    print(f"Суммарный штраф за импакт (Implementation Shortfall): {execution_results['total_is']:.2f}")