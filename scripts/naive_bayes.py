import os
import numpy as np
import pandas as pd

# Importing library
import math
import random
import csv


def Accuracy(y, prediction):

    # Function to calculate accuracy
    y = list(y)
    prediction = list(prediction)
    score = 0

    for i, j in zip(y, prediction):
        if i == j:
            score += 1

    return score / len(y)


def count(data, colname, label, target):
    condition = (data[colname] == label) & (data["Class"] == target)
    return len(data[condition])


# train_df = pd.read_csv(os.getcwd() + "/data/clean_train_adult.csv", index_col=0)
# test_df = pd.read_csv(os.getcwd() + "/data/clean_test_adult.csv", index_col=0)

data = pd.read_csv(os.getcwd() + "/data/diabetes.csv")

labels = ["low", "medium", "high"]

for j in data.columns[:-1]:
    mean = data[j].mean()
    data[j] = data[j].replace(0, mean)  # Replace 0 with mean
    data[j] = pd.cut(data[j], bins=len(labels), labels=labels)

train_percent = 70
train_len = int((train_percent * len(data)) / 100)
train_X = data.iloc[:train_len, :]
test_X = data.iloc[train_len + 1 :, :-1]
test_y = data.iloc[train_len + 1 :, -1]

# categorical = [var for var in train_df.columns if train_df[var].dtype == "O"]
# categorical1 = [var for var in test_df.columns if test_df[var].dtype == "O"]

# categorial data to numeric
# cat_columns = train_df.select_dtypes(["0"]).columns
# train_df[cat_columns] = train_df[cat_columns].apply(lambda x: x.cat.codes)

# cat_columns1 = train_df.select_dtypes(["0"]).columns
# test_df[cat_columns1] = test_df[cat_columns1].apply(lambda x: x.cat.codes)

# y_train = train_df["class"]
# x_train = train_df.drop("class", axis=1)
# y_test = test_df["class"]
# x_test = test_df.drop("class", axis=1)

# prior = train_df.groupby("class").size().div(len(train_df))
# print(count(train_df, "class", 0, 0))
count_0 = count(train_X, "Class", 0, 0)
count_1 = count(train_X, "Class", 1, 1)
# print(count(train_df, "class", 1, 1))
prob_0 = count_0 / len(train_X)
prob_1 = count_1 / len(train_X)
print(count_0)
print(count_1)
# print(prior)

predicted = []
probabilities = {0: {}, 1: {}}

for col in train_X.columns[:-1]:
    probabilities[0][col] = {}
    probabilities[1][col] = {}
    for category in labels:
        count_ct_0 = count(train_X, col, category, 0)
        count_ct_1 = count(train_X, col, category, 1)

        probabilities[0][col][category] = count_ct_0 / count_0
        probabilities[1][col][category] = count_ct_1 / count_1
# # print(
# #     "There are {} categorical variables\n".format(len(categorical) + len(categorical1))
# # )


for row in range(0, len(test_X)):
    prod_0 = prob_0
    prod_1 = prob_1
    for feature in test_X.columns:
        prod_0 *= probabilities[0][feature][test_X[feature].iloc[row]]
        prod_1 *= probabilities[1][feature][test_X[feature].iloc[row]]

    # Predict the outcome
    if prod_0 > prod_1:
        predicted.append(0)
    else:
        predicted.append(1)  #

tp, tn, fp, fn = 0, 0, 0, 0
for j in range(0, len(predicted)):
    if predicted[j] == 0:
        if test_y.iloc[j] == 0:
            tp += 1
        else:
            fp += 1
    else:
        if test_y.iloc[j] == 1:
            tn += 1
        else:
            fn += 1

print(
    "Accuracy: ",
    ((tp + tn) / len(test_y)) * 100,
)
print(probabilities)
# Calculate min and standard deviation
