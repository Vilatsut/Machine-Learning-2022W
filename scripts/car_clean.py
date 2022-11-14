import pandas as pd
import os

from sklearn import preprocessing

df = pd.read_csv(os.getcwd() + "/data/car_data.csv")

label_encoder = preprocessing.LabelEncoder()
df["buying"] = label_encoder.fit_transform(df["buying"])
df["maint"] = label_encoder.fit_transform(df["maint"])
df["lug_boot"] = label_encoder.fit_transform(df["lug_boot"])
df["safety"] = label_encoder.fit_transform(df["safety"])

df.to_csv("data/clean_car.csv")