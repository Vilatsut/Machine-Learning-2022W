import os

import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.lrn.csv", index_col=0)
comp_df = pd.read_csv(os.getcwd() + "/data/congress/CongressionalVotingID.shuf.tes.csv", index_col=0)

columns = ["handicapped-infants","water-project-cost-sharing","adoption-of-the-budget-resolution","physician-fee-freeze","el-salvador-aid","religious-groups-in-schools","anti-satellite-test-ban","aid-to-nicaraguan-contras","mx-missile","immigration","synfuels-crporation-cutback","education-spending","superfund-right-to-sue","crime","duty-free-exports","export-administration-act-south-africa"]

encoded_columns = []
for column in columns:
    encoded_columns.append(column + "_n")
    encoded_columns.append(column + "_unknown")
    encoded_columns.append(column + "_y")

# One-hot encoding
df = pd.get_dummies(df, columns = columns)
comp_df = pd.get_dummies(comp_df, columns = columns)

# Fill the dataframes with columns missing due to the variable not being present before dummy encoding.
for e_column in encoded_columns:
    if e_column not in df.columns:
        df[e_column] = 0
    if e_column not in comp_df.columns:
        comp_df[e_column] = 0

# Create the training data
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[1:]], df["class"], test_size=0.3, random_state=5)

train_df = pd.DataFrame(data=X_train, columns=X_train.columns, index=X_train.index)
train_df["class"] = y_train
test_df = pd.DataFrame(data=X_test, columns=X_test.columns, index=X_test.index)
test_df["class"] = y_test

# Encode columns
label_encoder = preprocessing.LabelEncoder()
train_df["class"] = label_encoder.fit_transform(train_df["class"])
test_df["class"] = label_encoder.fit_transform(test_df["class"])


# For label encoding the data columns
# for column in columns:
#     train_df[column] = label_encoder.fit_transform(train_df[column])
#     test_df[column] = label_encoder.fit_transform(test_df[column])
#     comp_df[column] = label_encoder.fit_transform(comp_df[column])


# For creating fgures of the data
# fig, ax = plt.subplots(1,3)
# sns.countplot(x=columns[0], data=df, palette='rainbow', hue='class', ax = ax[0])
# sns.countplot(x=columns[1], data=df, palette='rainbow', hue='class', ax = ax[1])
# sns.countplot(x=columns[2], data=df, palette='rainbow', hue='class', ax = ax[2])
# plt.pie(df["class"].value_counts().values, labels = np.flip(df["class"].unique()), autopct='%0.0f%%')
# plt.show()


# Save to files
train_df.to_csv("data/congress/clean_train_congress.csv")
test_df.to_csv("data/congress/clean_test_congress.csv")
comp_df.to_csv("data/congress/clean_comp_congress.csv")