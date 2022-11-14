import os

import pandas as pd

from sklearn import preprocessing

df = pd.read_csv(os.getcwd() + "/data/car_data.csv")

# Encode labels
label_encoder = preprocessing.LabelEncoder()
columns = ["buying", "maint","lug_boot", "safety"]
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])

df["persons"] = df["persons"].replace("more", 5)
df["doors"] = df["doors"].replace("5more", 5)

df.to_csv("data/clean_car.csv")