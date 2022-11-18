import os

import pandas as pd

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

# Create the training data
X_train = df[df.columns[:-1]]
y_train = df["class"]
X_test = test_df[test_df.columns[:-1]]
y_test = test_df["class"]

# DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier()
# Training
dt.fit(X_train, y_train)
# Predicting
dt_pred = dt.predict(X_test)
# Accuracy
print("\nAccuracy of DT: ", accuracy_score(y_test, dt_pred))
print("F1-score of DT: ", f1_score(y_test, dt_pred))
print("Confusion Matrix of DT: \n", confusion_matrix(y_test, dt_pred))

# KNN CLASSIFIER
knn_clf = KNeighborsClassifier(n_neighbors = 20)
# Training
knn_clf.fit(X_train, y_train)
# Predicting
knn_clf_pred = knn_clf.predict(X_test)
# Accuracy
print("\nAccuracy of KNN: ", accuracy_score(y_test, knn_clf_pred))
print("F1-score of KNN: ", f1_score(y_test, knn_clf_pred))
print("Confusion Matrix of KNN: \n", confusion_matrix(y_test, knn_clf_pred))

# SUPPORT VECTOR MACHINE
svm_clf = SVC()
# Training
svm_clf.fit(X_train, y_train)
# Predicting
svm_clf_pred = svm_clf.predict(X_test)
# Accuracy
print("\nAccuracy of SVM: ", accuracy_score(y_test, svm_clf_pred))
print("F1-score of SVM: ", f1_score(y_test, svm_clf_pred))
print("Confusion Matrix of SVM: \n", confusion_matrix(y_test, svm_clf_pred))

# DUMMY
dummy_clf = DummyClassifier()
# Training
dummy_clf.fit(X_train, y_train)
# Predicting
dummy_clf_pred = dummy_clf.predict(X_test)
# Accuracy
print("\nAccuracy of dummy: ", accuracy_score(y_test, dummy_clf_pred))
print("F1-score of dummy: ", f1_score(y_test, dummy_clf_pred))
print("Confusion Matrix of dummy: \n", confusion_matrix(y_test, dummy_clf_pred))
