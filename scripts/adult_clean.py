import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.getcwd() + "/data/adult/clean_adult.csv", index_col=0)

#Clean the missing values
df["native-country"] = df["native-country"].replace(to_replace=" ?", value="Unknown")
df["occupation"] = df["occupation"].replace(to_replace=" ?", value="Unemployed")
df["workclass"] = df["workclass"].replace(to_replace=" ?", value="Unemployed")
df["class"] = df["class"].replace(to_replace=" <=50K", value=0)
df["class"] = df["class"].replace(to_replace=" >50K", value=1)



columns = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


encoded_columns = []
for column in columns:
    encoded_columns.append(column + "_n")
    encoded_columns.append(column + "_unknown")
    encoded_columns.append(column + "_y")

df = pd.get_dummies(df, columns = columns)

# Fill the dataframes with columns missing due to the variable not being present before dummy encoding.
for e_column in encoded_columns:
    if e_column not in df.columns:
        df[e_column] = 0



X_train, X_test, y_train, y_test = train_test_split(df[df.columns], df["class"], test_size=0.3, random_state=42)

train_df = pd.DataFrame(data=X_train, columns=X_train.columns, index=X_train.index)
train_df["class"] = y_train
test_df = pd.DataFrame(data=X_test, columns=X_test.columns, index=X_test.index)
test_df["class"] = y_test

print(test_df.sample(5))
print(train_df.sample(5))
print(df.columns)

# Encode columns
label_encoder = preprocessing.LabelEncoder()
train_df["class"] = label_encoder.fit_transform(train_df["class"])
test_df["class"] = label_encoder.fit_transform(test_df["class"])

train_df.to_csv("data/adult/clean_train_adult.csv")
test_df.to_csv("data/adult/clean_test_adult.csv")


