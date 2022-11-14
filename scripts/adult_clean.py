import pandas as pd
import os

from sklearn import preprocessing

df = pd.read_csv(os.getcwd() + "/data/adult_income_data.csv")

label_encoder = preprocessing.LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"])
df["workclass"] = label_encoder.fit_transform(df["workclass"])
df["education"] = label_encoder.fit_transform(df["education"])
df["marital.status"] = label_encoder.fit_transform(df["marital.status"])
df["occupation"] = label_encoder.fit_transform(df["occupation"])
df["relationship"] = label_encoder.fit_transform(df["relationship"])
df["race"] = label_encoder.fit_transform(df["race"])
df["sex"] = label_encoder.fit_transform(df["sex"])
df["native.country"] = label_encoder.fit_transform(df["native.country"])

df.to_csv("data/clean_adults.csv")

