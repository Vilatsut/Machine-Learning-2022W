import os
from matplotlib import pyplot as plt
import numpy as np

import pandas as pd
from sklearn import preprocessing

from seaborn import pairplot

df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.lrn.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.tes.csv", index_col=0) 
sol_df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.sol.ex.csv", index_col=0) 


# # Missing values handling, deleting rows drops the accuracy 10%, maybe something else? Imputing?
# df = df.replace('unknown',np.nan).dropna(axis = 0, how = 'any')
# test_df = test_df.replace('unknown',np.nan).dropna(axis = 0, how = 'any')
# sol_df = sol_df[sol_df.index.isin(test_df.index)]

# Encode labels
columns = ["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-crporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]
df = pd.get_dummies(df, columns = columns)
test_df = pd.get_dummies(test_df, columns = columns)

label_encoder = preprocessing.LabelEncoder()
df["class"] = label_encoder.fit_transform(df["class"])
sol_df["class"] = label_encoder.fit_transform(sol_df["class"])

# Save to files
df.to_csv("data/congress/clean_congress.csv")
test_df.to_csv("data/congress/clean_test_congress.csv")
sol_df.to_csv("data/congress/clean_sol_congress.csv")

