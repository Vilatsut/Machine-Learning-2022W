import os

import pandas as pd
from sklearn import preprocessing

df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.lrn.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.tes.csv", index_col=0) 
sol_df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.sol.ex.csv", index_col=0) 

# Missing values handling


# Encode labels
label_encoder = preprocessing.LabelEncoder()
columns = ["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-crporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]
for column in columns:
    df[column] = label_encoder.fit_transform(df[column])
    test_df[column] = label_encoder.fit_transform(test_df[column])

# Save to files
df.to_csv("data/congress/clean_congress.csv")
test_df.to_csv("data/congress/clean_test_congress.csv")
sol_df.to_csv("data/congress/clean_sol_congress.csv")

