import os
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator


class OwnNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, test_X, test_y):
        self.labels = ["low", "medium", "high"]
        self.predicted = []
        self.probabilities = {0: {}, 1: {}}
        self.test_y = test_y
        self.test_X = test_X

    # function to traverse through feautes and count the occurances of a nominal value
    def count(self, data, colname, label, target):
        condition = (data[colname] == label) & (data["Class"] == target)
        return len(data[condition])

    def fit(self, X):
        # count the probability of 1 or 0 in our train data
        count_0 = self.count(train_X, "Class", 0, 0)
        count_1 = self.count(train_X, "Class", 1, 1)
        prob_0 = count_0 / len(train_X)
        prob_1 = count_1 / len(train_X)

        # traversing through all the features and calculate the probabilities of all unique values
        # aka training the model
        for col in train_X.columns[:-1]:
            self.probabilities[0][col] = {}
            self.probabilities[1][col] = {}
            for category in self.labels:
                count_ct_0 = self.count(train_X, col, category, 0)
                count_ct_1 = self.count(train_X, col, category, 1)

                self.probabilities[0][col][category] = count_ct_0 / count_0
                self.probabilities[1][col][category] = count_ct_1 / count_1
        # test the model
        for row in range(0, len(test_X)):
            # final probabilites of which we will choose the highest probability
            prod_0 = prob_0
            prod_1 = prob_1
            # P(row) = P(0)* P(feature) * P(feature) ...
            for feature in test_X.columns:
                prod_0 *= self.probabilities[0][feature][test_X[feature].iloc[row]]
                prod_1 *= self.probabilities[1][feature][test_X[feature].iloc[row]]

            # Predict the outcome
            if prod_0 > prod_1:
                self.predicted.append(0)
            else:
                self.predicted.append(1)  #

    def predict(self, X):
        # "confusion matrix"
        # true positive, true negative, falsely positive, falsely negative
        # compare the rtest set and our predicted results
        tp, tn, fp, fn = 0, 0, 0, 0
        for j in range(0, len(self.predicted)):
            if self.predicted[j] == 0:
                if self.test_y.iloc[j] == 0:
                    tp += 1
                else:
                    fp += 1
            else:
                if self.test_y.iloc[j] == 1:
                    tn += 1
                else:
                    fn += 1
        return self.predicted
        # accuracy = (tp+tn) / (p+n)
        # print(
        #     "Accuracy: ",
        #     ((tp + tn) / len(self.test_y)) * 100,
        # )
        # # print(self.probabilities)
        # print()


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

# %%
if __name__ == "__main__":
    # temp = "/Users/tiimie/Desktop/tu/ml/ML-Crypto-Car-Project/EX2/data/diabetes.csv"
    data = pd.read_csv(os.getcwd() + "/data/diabetes.csv")
    # data = pd.read_csv(temp)
    labels = ["low", "medium", "high"]

    for j in data.columns[:-1]:
        mean = data[j].mean()
        # replacing the zero value with the mean value of the column
        data[j] = data[j].replace(0, mean)
        # using pd's cut to convert numeric data to nominal storing them in bins that are labeled above
        data[j] = pd.cut(data[j], bins=len(labels), labels=labels)

    train_percent = 70
    train_len = int((train_percent * len(data)) / 100)
    train_X = data.iloc[:train_len, :]
    test_X = data.iloc[train_len + 1 :, :-1]
    test_y = data.iloc[train_len + 1 :, -1]

    nb = OwnNaiveBayes(test_X, test_y)
    nb.fit(train_X)
    print(nb.score(test_X, test_y))
    # nb.predict()


# %%
