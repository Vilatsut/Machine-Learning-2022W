import pandas as pd
import os

df = pd.read_csv(os.getcwd() + "/data/adult_income_data.csv")

df["native.country"] = df["native.country"].replace(to_replace="?", value="Unknown")
df["occupation"] = df["occupation"].replace(to_replace="?", value="Unemployed")
df["workclass"] = df["workclass"].replace(to_replace="?", value="Unemployed")

print(df.head())

df.to_csv("data/clean_adults.csv")

