import pandas as pd
import os
import torch.nn as nn


class NaiveBayesClassifier():
  def __init__(self):
    ###################################
    # Write your own code here #
    pass

    ###################################

  def forward(self, x):
    ###################################
    # Write your own code here #


    ###################################
    return x


df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

# Create the training data
X_train = df[df.columns[:-1]]
y_train = df["class"]
X_test = test_df[test_df.columns[:-1]]
y_test = test_df["class"]

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

for train_index, test_index in skf.split(X,y_binned):
    pass