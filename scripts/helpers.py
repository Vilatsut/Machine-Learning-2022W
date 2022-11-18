
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier

def experiment_DT_classifier(X_train, y_train, X_test, y_test):
    dt = DecisionTreeClassifier()
    # Training
    dt.fit(X_train, y_train)
    # Predicting
    dt_pred = dt.predict(X_test)
    # Accuracy
    print_scores(dt_pred, y_test, "DT")

def experiment_KN_classifier(X_train, y_train, X_test, y_test):

    knn_clf = KNeighborsClassifier(n_neighbors = 20)
    # Training
    knn_clf.fit(X_train, y_train)
    # Predicting
    knn_clf_pred = knn_clf.predict(X_test)
    # Accuracy
    print_scores(knn_clf_pred, y_test, "KN")

def experiment_SVM_classifier(X_train, y_train, X_test, y_test):

    svm_clf = SVC()
    # Training
    svm_clf.fit(X_train, y_train)
    # Predicting
    svm_clf_pred = svm_clf.predict(X_test)
    # Accuracy
    print_scores(svm_clf_pred, y_test, "SVM")

def dummy(X_train, y_train, X_test,y_test):
    dummy_clf = DummyClassifier()
    # Training
    dummy_clf.fit(X_train, y_train)
    # Predicting
    dummy_clf_pred = dummy_clf.predict(X_test)
    # Accuracy
    print_scores(dummy_clf_pred, y_test, "dummy")

def print_scores(pred, y_test, classifier):
    
    print("\nAccuracy of " + classifier +  ": ", accuracy_score(y_test, pred))
    print("F1-score of " + classifier +  ": ", f1_score(y_test, pred))
    print("Confusion Matrix of " + classifier +  ": \n", confusion_matrix(y_test, pred))