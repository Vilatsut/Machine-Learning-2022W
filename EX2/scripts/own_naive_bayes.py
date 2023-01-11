import os
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator


class OwnNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self, test_X, test_y):
        # self.data = pd.read_csv(os.getcwd() + "/data/diabetes.csv")
        self.labels = ["low", "medium", "high"]
        self.predicted = []
        self.probabilities = {0: {}, 1: {}}
        self.test_y = test_y
        self.test_X = test_X

    def count(self, data, colname, label, target):
        condition = (data[colname] == label) & (data["Class"] == target)
        return len(data[condition])

    def fit(self, X):
        # prior = train_df.groupby("class").size().div(len(train_df))
        count_0 = self.count(train_X, "Class", 0, 0)
        count_1 = self.count(train_X, "Class", 1, 1)
        prob_0 = count_0 / len(train_X)
        prob_1 = count_1 / len(train_X)
        # print(count_0)
        # print(count_1)

        for col in train_X.columns[:-1]:
            self.probabilities[0][col] = {}
            self.probabilities[1][col] = {}
            for category in self.labels:
                count_ct_0 = self.count(train_X, col, category, 0)
                count_ct_1 = self.count(train_X, col, category, 1)

                self.probabilities[0][col][category] = count_ct_0 / count_0
                self.probabilities[1][col][category] = count_ct_1 / count_1

        for row in range(0, len(test_X)):
            prod_0 = prob_0
            prod_1 = prob_1
            for feature in test_X.columns:
                prod_0 *= self.probabilities[0][feature][test_X[feature].iloc[row]]
                prod_1 *= self.probabilities[1][feature][test_X[feature].iloc[row]]

            # Predict the outcome
            if prod_0 > prod_1:
                self.predicted.append(0)
            else:
                self.predicted.append(1)  #

    def predict(self, X):
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

    # df = pd.read_csv(os.getcwd() + "/data/diabetes.csv")

    # X = df[df.columns[:-1]]
    # y = df["Class"]

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
    # clf = GaussianNB()
    # clf.fit(X_train, y_train)
    # print(
    #     "Model accuracy score: {0:0.4f}".format(
    #         accuracy_score(y_test, clf.predict(X_test))
    #     )
    # )

    data = pd.read_csv(os.getcwd() + "/data/diabetes.csv")
    labels = ["low", "medium", "high"]

    for j in data.columns[:-1]:
        mean = data[j].mean()
        data[j] = data[j].replace(0, mean)
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
# Calculate min and standard deviation
