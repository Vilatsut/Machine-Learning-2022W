import os

import pandas as pd

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv(os.getcwd() + "/data/clean_adults.csv", index_col=0)

# Create the training data
X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[0:-1]], df["income"], test_size=0.3, random_state=1)

# GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
# Training
gnb.fit(X_train, y_train)
# Predicting
gnb_pred = gnb.predict(X_test)
# Accuracy
print("Accuracy of Gaussian Naive Bayes: ", accuracy_score(y_test, gnb_pred))
 
 # DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(random_state=0)
# Training
dt.fit(X_train, y_train)
# Predicting
dt_pred = dt.predict(X_test)
# Accuracy
print("Accuracy of Decision Tree Classifier: ", accuracy_score(y_test, dt_pred))
 
 # KNN CLASSIFIER
knn_clf = KNeighborsClassifier(n_neighbors = 20)
# Training
knn_clf.fit(X_train, y_train)
# Prediction
knn_clf_pred = knn_clf.predict(X_test)
# Accuracy
print("Accuracy of KNN: ", accuracy_score(y_test, knn_clf_pred))

# SUPPORT VECTOR MACHINE
svm_clf = svm.SVC()  # Linear Kernel
# Training
svm_clf.fit(X_train, y_train) # Takes approx 10min
# Predicting
svm_clf_pred = svm_clf.predict(X_test)
# Accuracy
print("Accuracy of Support Vector Machine: ", accuracy_score(y_test, svm_clf_pred))
