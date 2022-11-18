import os

import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.lrn.csv", index_col=0)

columns = ["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-crporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]

# Encode data
df = pd.get_dummies(df, columns = columns)

# Create the training data
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[1:-1]], df["class"], test_size=0.3, random_state=5)

train_df = pd.DataFrame(data=X_train, columns=X_train.columns, index=X_train.index)
train_df["class"] = y_train
test_df = pd.DataFrame(data=X_test, columns=X_test.columns, index=X_test.index)
test_df["class"] = y_test

# Encode columns
label_encoder = preprocessing.LabelEncoder()
train_df["class"] = label_encoder.fit_transform(train_df["class"])
test_df["class"] = label_encoder.fit_transform(test_df["class"])


# # Figures
# plt.pie(df["class"].value_counts().values, labels = np.flip(df["class"].unique()), autopct='%0.0f%%')
# plt.show()

# Save to files
train_df.to_csv("data/congress/clean_train_congress.csv")
test_df.to_csv("data/congress/clean_test_congress.csv")

