import pandas as pd

filenames = {
    "open": "./open_.parquet",
    "close": "./close_.parquet",
    "high": "./high_.parquet",
    "low": "./low_.parquet",
    "buy_size": "./buy_size_.parquet",
    "sell_size": "./sell_size_.parquet"
}
#если нужно добавить файлы то сюда
dataframes = {
    key: pd.read_parquet(path) 
    for key, path in filenames.items()
}
#если нужно добавить датафреймы то сюда

df_3d = pd.concat(list(dataframes.values()), axis=1, keys=list(dataframes.keys()))
df_swap = df_3d.swaplevel(0, 1, axis=1).sort_index(axis=1) #выход

# данные будем передавать в pandas dataframe'ах с несколькими уровнями колонок
# 0 уровень – название акции
# 1 уровень – по параметру на название акции

# примерный вид:

# АКЦИЯ
# buy_size close high low open sell_size
# число    число число число число число

print(df_swap)
