import pandas as pd
import os

print(os.getcwd())
df = pd.read_csv(os.getcwd() + "/data/adult.csv")

df["native-country"] = df["native.country"].replace(to_replace=" ?", value=" Unknown")
df["occupation"] = df["occupation"].replace(to_replace=" ?", value=" Unemployed")
df["workclass"] = df["workclass"].replace(to_replace=" ?", value=" Unemployed")

print(df.sample(100))

df.to_csv("data/clean_adults.csv")

