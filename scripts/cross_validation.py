import os

import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB


if __name__ == '__main__':
  df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
  test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

  # Create the training data
  X_train = df[df.columns[:-1]]
  X_test = test_df[test_df.columns[:-1]]
  x = pd.concat([X_train, X_test])
  y_train = df["class"]
  y_test = test_df["class"]
  y = pd.concat([y_train, y_test])

  skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)

  for train_index, test_index in skf.split(x, y):
 
    naive_bayes = GaussianNB()
    
    naive_bayes.fit(x.iloc[train_index] , y.iloc[train_index])
    print(naive_bayes.score(x.iloc[test_index], y.iloc[test_index]))