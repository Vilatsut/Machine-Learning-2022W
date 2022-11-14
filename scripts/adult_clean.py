import os

import pandas as pd
from sklearn import preprocessing

df = pd.read_csv(os.getcwd() + "/data/adult_income_data.csv")

# Encode labels
label_encoder = preprocessing.LabelEncoder()
columns = ["income", "workclass", "education", "marital.status", "occupation", "relationship", "race", "sex", "native.country"]
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])

df.to_csv("data/clean_adults.csv")

