from operator import truediv
import pandas as pd
import numpy as np


#data = pd.read_csv("dataset.csv")

chunk = pd.read_csv('dataset.csv', sep=",", chunksize=1000, on_bad_lines="skip")

df = pd.concat(chunk)

df.drop('platform_name', inplace=True, axis=1)
df.drop('industry_name', inplace=True, axis=1)
df.drop('capitalization_change_1_day', inplace=True, axis=1)
df.drop('USD_price_change_1_day', inplace=True, axis=1)
df.drop('max_supply', inplace=True, axis=1)
df.drop('crypto_type', inplace=True, axis=1)
df.drop('ticker', inplace=True, axis=1)
df.drop('site_url', inplace=True, axis=1)
df.drop('github_url', inplace=True, axis=1)
df.drop('minable', inplace=True, axis=1)

allowed_names = [ "Cosmos", "Bitcoin", "Ethereum", "Cardano", "Solana" ]

new_df = df
flag = True

for name in allowed_names:
    temp_df = df[df["crypto_name"] == name]
    if flag == True:
        new_df = temp_df        
        flag = False
    else:
        new_df = pd.concat([new_df, temp_df])


print(new_df.sample(100))

new_df.to_csv("clean_dataset.csv", index=False)

