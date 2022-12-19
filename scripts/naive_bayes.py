import os
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, BaseEstimator


class OwnNaiveBayes(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.data = pd.read_csv(os.getcwd() + "/data/diabetes.csv")
        self.labels = ["low", "medium", "high"]
        self.predicted = []
        self.probabilities = {0: {}, 1: {}}
        self.test_y = None

    # def Accuracy(y, prediction):

    #     # Function to calculate accuracy
    #     y = list(y)
    #     prediction = list(prediction)
    #     score = 0

    #     for i, j in zip(y, prediction):
    #         if i == j:
    #             score += 1

    #     return score / len(y)

    def count(self, data, colname, label, target):
        condition = (data[colname] == label) & (data["Class"] == target)
        return len(data[condition])

    def fit(self):
        for j in self.data.columns[:-1]:
            mean = self.data[j].mean()
            self.data[j] = self.data[j].replace(0, mean)
            self.data[j] = pd.cut(
                self.data[j], bins=len(self.labels), labels=self.labels
            )

        train_percent = 70
        train_len = int((train_percent * len(self.data)) / 100)
        train_X = self.data.iloc[:train_len, :]
        test_X = self.data.iloc[train_len + 1 :, :-1]
        self.test_y = self.data.iloc[train_len + 1 :, -1]

        # prior = train_df.groupby("class").size().div(len(train_df))
        count_0 = self.count(train_X, "Class", 0, 0)
        count_1 = self.count(train_X, "Class", 1, 1)
        prob_0 = count_0 / len(train_X)
        prob_1 = count_1 / len(train_X)
        print(count_0)
        print(count_1)

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

    def predict(self):
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

        print(
            "Accuracy: ",
            ((tp + tn) / len(self.test_y)) * 100,
        )
        print(self.probabilities)


# %%
if __name__ == "__main__":
    nb = OwnNaiveBayes()
    nb.fit()
    nb.predict()

# %%
# Calculate min and standard deviation
