import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.getcwd() + "/data/car/car_data/car-data.csv")


df["persons"] = df["persons"].replace("more", 5)
df["doors"] = df["doors"].replace("5more", 5)

columns = ["buying", "maint","lug_boot", "safety"]
df = pd.get_dummies(df, columns = columns)

X_train, X_test, y_train, y_test = train_test_split(df[df.columns], df["class"], test_size=0.3, random_state=5)

train_df = pd.DataFrame(data=X_train, columns=X_train.columns, index=X_train.index)
train_df["class"] = y_train
test_df = pd.DataFrame(data=X_test, columns=X_test.columns, index=X_test.index)
test_df["class"] = y_test

label_encoder = preprocessing.LabelEncoder()
train_df["class"] = label_encoder.fit_transform(train_df["class"])
test_df["class"] = label_encoder.fit_transform(test_df["class"])





train_df.to_csv("data/car/clean_train_car.csv", index=False)
test_df.to_csv("data/car/clean_test_car.csv", index=False)


#df.to_csv("data/car/clean_car.csv")