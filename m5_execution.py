import pandas as pd
import numpy as np

def optimize_execution(signal_5min: pd.DataFrame, 
                        bars_1min: pd.DataFrame,
                        impact_model: pd.DataFrame,
                        aum: float = 1000000,
                        participation_max: float = 0.1) -> tuple:
    
    print("="*60)
    print("M5: ОПТИМИЗАЦИЯ ИСПОЛНЕНИЯ")
    print("="*60)
    
    signal = signal_5min.copy()
    signal['bar_end_ts'] = pd.to_datetime(signal['bar_end_ts'])
    
    bars = bars_1min.copy()
    bars['timestamp'] = pd.to_datetime(bars['timestamp'])
    bars['volume_mkt'] = bars['total_bid_volume'] + bars['total_ask_volume']
    
    impact = impact_model.copy()
    impact['bar_end_ts'] = pd.to_datetime(impact['bar_end_ts'])
    
    active_signals = signal[signal['value'] != 0].copy()
    print(f"\nАктивных сигналов: {len(active_signals)}")
    
    execution_slices = []
    execution_summaries = []
    
    for idx, sig in active_signals.iterrows():
        ticker = sig['seccode']
        pos = sig['value']
        signal_time = sig['bar_end_ts']
        
        execution_start = signal_time + pd.Timedelta(minutes=5)
        execution_end = execution_start + pd.Timedelta(minutes=5)
        
        next_bar = bars[(bars['timestamp'] >= execution_start) & 
                        (bars['timestamp'] < execution_start + pd.Timedelta(minutes=5)) &
                        (bars['ticker'] == ticker)]
        
        if len(next_bar) == 0:
            continue
        
        entry_price = next_bar['open'].iloc[0]
        exit_price = next_bar['close'].iloc[-1]
        
        price_return = (exit_price - entry_price) / entry_price
        pnl_mid = pos * price_return * aum
        
        slices = bars[(bars['timestamp'] >= execution_start) & 
                      (bars['timestamp'] < execution_end) &
                      (bars['ticker'] == ticker)].copy()
        
        if len(slices) == 0:
            continue
        
        q_target = abs(pos) * aum / entry_price
        
        total_volume = slices['volume_mkt'].sum()
        if total_volume == 0:
            continue
        
        slices['q_slice'] = q_target * slices['volume_mkt'] / total_volume
        slices['participation_rate'] = slices['q_slice'] / slices['volume_mkt']
        slices['participation_rate'] = slices['participation_rate'].clip(0, participation_max)
        
        slices = slices.merge(impact[['bar_end_ts', 'seccode', 'a']], 
                               left_on=['timestamp', 'ticker'],
                               right_on=['bar_end_ts', 'seccode'],
                               how='left')
        slices['a'] = slices['a'].fillna(0.01)
        slices['impact_cost_rel'] = slices['a'] * slices['participation_rate']
        
        is_cost = (slices['impact_cost_rel'] * slices['q_slice'] * entry_price).sum()
        pnl_net = pnl_mid - is_cost
        
        fill_price = entry_price * (1 + np.sign(pos) * slices['impact_cost_rel'])
        vwap = (fill_price * slices['q_slice']).sum() / slices['q_slice'].sum() if slices['q_slice'].sum() > 0 else entry_price
        twap = slices['close'].mean()
        
        for _, row in slices.iterrows():
            execution_slices.append({
                'bar_end_ts_5min': signal_time,
                'bar_end_ts_1min': row['timestamp'],
                'seccode': ticker,
                'q_slice': row['q_slice'],
                'participation_rate': row['participation_rate'],
                'impact_cost_rel': row['impact_cost_rel']
            })
        
        execution_summaries.append({
            'bar_end_ts_5min': signal_time,
            'seccode': ticker,
            'Q_executed': slices['q_slice'].sum(),
            'vwap_fill': vwap,
            'twap_bench': twap,
            'implementation_shortfall': is_cost,
            'pnl_mid': pnl_mid,
            'pnl_net': pnl_net,
            'pos': pos
        })
    
    exec_slices_df = pd.DataFrame(execution_slices)
    exec_summary_df = pd.DataFrame(execution_summaries)
    
    if len(exec_summary_df) > 0:
        print(f"\nРезультаты оптимизации:")
        print(f"  Всего исполнений: {len(exec_summary_df)}")
        print(f"  Суммарный PnL mid: {exec_summary_df['pnl_mid'].sum():.2f}")
        print(f"  Суммарный IS: {exec_summary_df['implementation_shortfall'].sum():.2f}")
        print(f"  Суммарный PnL net: {exec_summary_df['pnl_net'].sum():.2f}")
        
        profitable = (exec_summary_df['pnl_net'] > 0).sum()
        print(f"  Прибыльных сделок: {profitable} / {len(exec_summary_df)} ({profitable/len(exec_summary_df)*100:.1f}%)")
    
    return exec_slices_df, exec_summary_df

if __name__ == "__main__":
    print("Загрузка данных...")
    signal = pd.read_parquet("signal_5min.parquet")
    bars_1min = pd.read_parquet("bars_1min.parquet")
    impact = pd.read_parquet("impact_model.parquet")
    
    exec_slices, exec_summary = optimize_execution(
        signal, bars_1min, impact, 
        aum=1000000,
        participation_max=0.1
    )
    
    if len(exec_slices) > 0:
        exec_slices.to_parquet("execution_schedule.parquet", index=False)
        exec_summary.to_parquet("execution_summary.parquet", index=False)
        print(f"\n✅ Сохранено: execution_schedule.parquet, execution_summary.parquet")
