import os

import pandas as pd

from helpers import experiment_DT_classifier, experiment_KN_classifier, experiment_SVM_classifier, dummy

df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

# Create the training data
X_train = df[df.columns[:-1]]
y_train = df["class"]
X_test = test_df[test_df.columns[:-1]]
y_test = test_df["class"]

# DECISION TREE CLASSIFIER
experiment_DT_classifier(X_train, y_train, X_test,y_test)

# KNN CLASSIFIER
experiment_KN_classifier(X_train, y_train, X_test,y_test)

# SUPPORT VECTOR MACHINE
experiment_SVM_classifier(X_train, y_train, X_test,y_test)

# DUMMY
dummy(X_train, y_train, X_test,y_test)