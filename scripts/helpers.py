import sys

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier



def experiment_DT_classifier(X_train, y_train, X_test, y_test):
    for criterion in ["gini", "entropy", "log_loss"]:
        for splitter in ["best", "random"]:
            for max_depth in [3, 10, None]:
                dt = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth)
                # Training
                dt.fit(X_train, y_train)
                # Predicting
                dt_pred = dt.predict(X_test)
                # Accuracy
                args = [criterion, splitter, max_depth]
                print_scores(dt_pred, y_test, "DT", args)

def experiment_KN_classifier(X_train, y_train, X_test, y_test):
    for n_neighbours in [3, 5, 10, 20]:
        for weights in ["uniform", "distance"]:
            for algorithm in ["auto", "ball_tree", "kd_tree", "brute"]:
                for leaf_size in [1,5,10,30,50,100]:
                    knn_clf = KNeighborsClassifier(
                        n_neighbors = n_neighbours, 
                        weights=weights, 
                        algorithm=algorithm,
                        leaf_size=leaf_size
                        )
                    # Training
                    knn_clf.fit(X_train, y_train)
                    # Predicting
                    knn_clf_pred = knn_clf.predict(X_test)
                    # Accuracy
                    args = [n_neighbours, weights, algorithm, leaf_size]
                    print_scores(knn_clf_pred, y_test, "KN", args)


def experiment_SVM_classifier(X_train, y_train, X_test, y_test):
    for C in [0.5, 1, 10]:
        for kernel in ["linear", "poly", "rbf", "sigmoid"]:
            for gamma in ["scale", "auto"]:
                svm_clf = SVC(C=C, kernel=kernel, gamma=gamma)
                # Training
                svm_clf.fit(X_train, y_train)
                # Predicting
                svm_clf_pred = svm_clf.predict(X_test)
                # Accuracy
                args = [C, kernel, gamma]
                print_scores(svm_clf_pred, y_test, "SVM", args)

def dummy(X_train, y_train, X_test,y_test):
    dummy_clf = DummyClassifier()
    # Training
    dummy_clf.fit(X_train, y_train)
    # Predicting
    dummy_clf_pred = dummy_clf.predict(X_test)
    # Accuracy
    print_scores(dummy_clf_pred, y_test, "dummy")

best_scores = { "DT": {"acc_score": 0, "f1_score": 0, "args": []}, 
                "KN": {"acc_score": 0, "f1_score": 0, "args": []},
                "SVM": {"acc_score": 0, "f1_score": 0, "args": []},
                "dummy": {"acc_score": 0, "f1_score": 0, "args": []}
                }
def print_scores(pred, y_test, classifier, args = []):
    acc_scr = accuracy_score(y_test, pred)
    f1_scr = f1_score(y_test, pred)
    print("\nArguments used: " + str(args))
    print("Accuracy of " + classifier +  ": ", acc_scr)
    print("F1-score of " + classifier +  ": ", f1_scr)
    print("Confusion Matrix of " + classifier +  ": \n", confusion_matrix(y_test, pred))
    # with open(classifier + ".data", "a") as f:
    #     f.write(str(f1_scr) + "\n")


    if acc_scr > best_scores[classifier]["acc_score"] and f1_scr > best_scores[classifier]["f1_score"]:
        best_scores[classifier]["acc_score"] = acc_scr
        best_scores[classifier]["f1_score"] = f1_scr
        best_scores[classifier]["args"] = args
        # print("THEBEST: ")
        # print(best_scores[classifier])