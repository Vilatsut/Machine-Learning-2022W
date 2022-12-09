import pandas as pd
import os




df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

# Create the training data
X_train = df[df.columns[:-1]]
y_train = df["class"]
X_test = test_df[test_df.columns[:-1]]
y_test = test_df["class"]