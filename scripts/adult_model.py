import pandas as pd
import numpy as np
import scipy as sp
import os
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


df = pd.read_csv(os.getcwd() + "/data/clean_adults.csv", index_col=0)

# print(df.head())
label_encoder = preprocessing.LabelEncoder()
df["income"] = label_encoder.fit_transform(df["income"])
df["workclass"] = label_encoder.fit_transform(df["workclass"])
df["education"] = label_encoder.fit_transform(df["education"])
df["marital.status"] = label_encoder.fit_transform(df["marital.status"])
df["occupation"] = label_encoder.fit_transform(df["occupation"])
df["relationship"] = label_encoder.fit_transform(df["relationship"])
df["race"] = label_encoder.fit_transform(df["race"])
df["sex"] = label_encoder.fit_transform(df["sex"])
df["native.country"] = label_encoder.fit_transform(df["native.country"])


X_train, X_test, y_train, y_test = train_test_split(
    df[df.columns[0:-1]], df["income"], test_size=0.3, random_state=1)

# GAUSSIAN NAIVE BAYES
gnb = GaussianNB()
# train the model
gnb.fit(X_train, y_train)
# make predictions
gnb_pred = gnb.predict(X_test)
# print the accuracy
print("Accuracy of Gaussian Naive Bayes: ", accuracy_score(y_test, gnb_pred))
 
 # DECISION TREE CLASSIFIER
dt = DecisionTreeClassifier(random_state=0)
# train the model
dt.fit(X_train, y_train)
# make predictions
dt_pred = dt.predict(X_test)
# print the accuracy
print("Accuracy of Decision Tree Classifier: ", accuracy_score(y_test, dt_pred))
 
# SUPPORT VECTOR MACHINE
svm_clf = svm.SVC()  # Linear Kernel
# train the model
svm_clf.fit(X_train, y_train)
# make predictions
svm_clf_pred = svm_clf.predict(X_test)
# print the accuracy
print("Accuracy of Support Vector Machine: ",
      accuracy_score(y_test, svm_clf_pred))