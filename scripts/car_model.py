import os
import sys
import pandas as pd
from helpers import experiment_DT_classifier, experiment_KN_classifier, experiment_SVM_classifier, dummy

from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler



orig_stdout = sys.stdout
f = open('out-car.txt', 'w')
sys.stdout = f

train_df = pd.read_csv(os.getcwd() + "/data/car/clean_train_car.csv")
test_df = pd.read_csv(os.getcwd() + "/data/car/clean_test_car.csv")

# Create the training data
X_train = train_df[train_df.columns]
y_train = train_df["class"]
X_test = test_df[test_df.columns]
y_test = test_df["class"]

# DECISION TREE CLASSIFIER
experiment_DT_classifier(X_train, y_train, X_test,y_test)

# KNN CLASSIFIER
experiment_KN_classifier(X_train, y_train, X_test,y_test)

# GAUSSIAN PROCESS
gnb = GaussianNB()
# Training
gnb.fit(X_train, y_train)
#Predicting
gnb_pred = gnb.predict(X_test)
# Accuracy
print("Accuracy of Gaussian Naive Bayes: ", accuracy_score(y_test, gnb_pred))


# DUMMY
dummy(X_train, y_train, X_test,y_test)

sys.stdout = orig_stdout
f.close()
