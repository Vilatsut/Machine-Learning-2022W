# %%
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score
# %%
def print_scores(pred, y_test, classifier):
    print("\nAccuracy of " + classifier +  ": ", accuracy_score(y_test, pred))
# %%
df_amazon_raw = pd.read_csv(r'C:\Users\beregszaszim\Desktop\TU_Wien\ML\Exercise_1\ML-Adult-Car-Project\data\amazon\184702-tu-ml-ws-22-amazon-commerce-reviews\amazon_review_ID.shuf.lrn.csv')
df_amazon_test = pd.read_csv(r'C:\Users\beregszaszim\Desktop\TU_Wien\ML\Exercise_1\ML-Adult-Car-Project\data\amazon\184702-tu-ml-ws-22-amazon-commerce-reviews\amazon_review_ID.shuf.tes.csv')
# %%
frequency = {}
for item in df_amazon_raw['Class'].tolist():
   if item in frequency:
      frequency[item] += 1
   else:
      frequency[item] = 1
# %%
print(np.sqrt(np.var(list(frequency.values()))))
# %% Create the training data
X_train, X_test, y_train, y_test = train_test_split(
    df_amazon_raw.iloc[:, 1:10001], df_amazon_raw['Class'], test_size=0.3, random_state=42)

# %% DUMMY
dummy_clf = DummyClassifier()
# Training
dummy_clf.fit(X_train, y_train)
# Predicting
dummy_clf_pred = dummy_clf.predict(X_test)
# Accuracy
print_scores(dummy_clf_pred, y_test, "dummy")

# %%
dt_clf_grid_params = {
    'criterion':  ['gini', 'entropy', 'log_loss'],
    'max_depth':  [None, 2, 4, 6, 8, 10],
    'max_features': [None, 'sqrt', 'log2', 0.2, 0.4, 0.6, 0.8],
    'splitter': ['best', 'random']
}

dt_clf_grid = GridSearchCV(
    estimator=DecisionTreeClassifier(),
    param_grid=dt_clf_grid_params,
    cv=5,
    n_jobs=-1
)

dt_clf_grid.fit(X_train, y_train)
print(dt_clf_grid.best_params_)
# %% Decision Tree
dt = DecisionTreeClassifier(max_features=0.4)
# Training
dt.fit(X_train, y_train)
# Predicting
dt_pred = dt.predict(X_test)
# Accuracy
print_scores(dt_pred, y_test, "DT")

# %% Scaling (if needed)
scaler = StandardScaler()
# Fit only on X_train
scaler.fit(X_train)

# Scale both X_train and X_test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% KNN GridSearchCV
knn_clf_grid_params = {
    'n_neighbors': list(range(0, 51)),
}

knn_clf_grid = GridSearchCV(
    estimator=KNeighborsClassifier(),
    param_grid=knn_clf_grid_params,
    scoring = 'accuracy',
    cv=5,
    n_jobs = -1
)

knn_clf_grid.fit(X_train, y_train)
print(knn_clf_grid.best_params_)
# %% KNN
knn_clf = KNeighborsClassifier(n_neighbors = 1)
# Training
knn_clf.fit(X_train, y_train)
# Prediction
knn_clf_pred = knn_clf.predict(X_test)
# Accuracy
print("Accuracy of KNN: ", accuracy_score(y_test, knn_clf_pred))

# %% Logistic Regression GridSearchCV
logreg_clf_grid_params = {
    'penalty' : ['l1','l2'], 
    'solver'  : ['newton-cg', 'lbfgs', 'liblinear'],
}

logreg_clf_grid = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=logreg_clf_grid_params,
    scoring = 'accuracy',
    cv=5,
    n_jobs = -1
)

logreg_clf_grid.fit(X_train_scaled, y_train)
print(logreg_clf_grid.best_params_)
# %% Logistic regression
log_reg_clf = LogisticRegression(penalty='l2', solver='liblinear')
# Training
log_reg_clf.fit(X_train_scaled, y_train)
# Predicting
log_reg_clf_pred = log_reg_clf.predict(X_test_scaled)
# Accuracy
print_scores(log_reg_clf_pred, y_test, "Logistic Regression")

# %% KAGGLE
# %%
X_train = df_amazon_raw.iloc[:, 1:10001]
y_train = df_amazon_raw.iloc[:, 10001]

X_test = df_amazon_test.iloc[:, 1:10001]
# %% Scaling
scaler.fit(X_train)

# Scale both X_train and X_test
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%
log_reg_clf.fit(X_train_scaled, y_train)
# %% Predicting
log_reg_clf_pred = log_reg_clf.predict(X_test_scaled)
# %%
df_kaggle = pd.DataFrame(columns=['ID', 'Class'])
df_kaggle['ID'] = df_amazon_test['ID']
df_kaggle['Class'] = log_reg_clf_pred
# %%
df_kaggle.to_csv(r'C:\Users\beregszaszim\Desktop\TU_Wien\ML\Exercise_1\ML-Adult-Car-Project\data\amazon\amazon_kaggle.csv', index=False)
# %%
