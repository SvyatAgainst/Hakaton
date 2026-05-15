import pandas as pd

filenames = {
    "open": "./open_.parquet",
    "close": "./close_.parquet",
    "high": "./high_.parquet",
    "low": "./low_.parquet",
    "buy_size": "./buy_size_.parquet",
    "sell_size": "./sell_size_.parquet"
}

open_df = pd.read_parquet(filenames["open"])
close_df = pd.read_parquet(filenames["close"])
high_df = pd.read_parquet(filenames["high"])
low_df = pd.read_parquet(filenames["low"])
buy_size_df = pd.read_parquet(filenames["buy_size"])
sell_size_df = pd.read_parquet(filenames["sell_size"])

df_3d = pd.concat([open_df, close_df, high_df, low_df, buy_size_df, sell_size_df], axis=1, keys=list(filenames.keys()))
df_swap = df_3d.swaplevel(0, 1, axis=1).sort_index(axis=1) #result

print(df_swap)
