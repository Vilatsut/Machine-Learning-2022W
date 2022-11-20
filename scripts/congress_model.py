import os
import sys

import pandas as pd

from helpers import experiment_DT_classifier, experiment_KN_classifier, experiment_SVM_classifier, dummy


orig_stdout = sys.stdout
f = open('out.txt', 'w')
sys.stdout = f

df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

# Create the training data
X_train = df[df.columns[:-1]]
y_train = df["class"]
X_test = test_df[test_df.columns[:-1]]
y_test = test_df["class"]

# DECISION TREE CLASSIFIER
experiment_DT_classifier(X_train, y_train, X_test,y_test)


# Best classifier based on f1-score running the Kaggle competition data

# from sklearn.tree import DecisionTreeClassifier
# competition_df = pd.read_csv(os.getcwd() + "/data/congress/clean_comp_congress.csv", index_col=0)
# dt = DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=10)
# # Training
# dt.fit(X_train, y_train)
# # Predicting
# dt_pred = dt.predict(competition_df)
# f = open('competition.csv', 'w')
# new_df = pd.DataFrame(data=dt_pred, index=competition_df.index, columns=["class"])
# new_df["class"] = new_df["class"].replace(0, "democrat")
# new_df["class"] = new_df["class"].replace(1, "republican")
# new_df.to_csv("data/congress/comp_result.csv")


# KNN CLASSIFIER
experiment_KN_classifier(X_train, y_train, X_test,y_test)

# SUPPORT VECTOR MACHINE
experiment_SVM_classifier(X_train, y_train, X_test,y_test)

# DUMMY
dummy(X_train, y_train, X_test,y_test)

sys.stdout = orig_stdout
f.close()
