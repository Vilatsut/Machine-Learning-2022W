import os

import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.datasets import load_diabetes


from timeit import default_timer

from linear_regression import OwnLinearRegression
from naive_bayes import OwnNaiveBayes

from sklearn.datasets import load_diabetes

# Kfold parameters
n_splits = 10
shuffle = True
random_state = None

def run_linear_cv(model, X, y, params = None):

	kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

	start = default_timer()
	linear_reg_model = model()
	scores = cross_val_score(linear_reg_model, X, y, cv=kf, scoring='neg_mean_absolute_error')
	end = default_timer()
	print(f"\n{model.__name__} RUNTIME: {end-start}")
	print(f"{model.__name__} scores: {scores}")
	print(f"{model.__name__} mean of scores: {scores.mean()}")

	print("\nFinding the best params")
	grid_clf = GridSearchCV(linear_reg_model, param_grid=params)
	grid_clf.fit(X, y)
	print(f'Best parameters: {grid_clf.best_params_}')
	print(f"Optimized {model.__name__} mean of scores: {grid_clf.best_score_}")

	start = default_timer()
	clf_model = model(**grid_clf.best_params_)
	scores = cross_val_score(clf_model, X, y, cv=kf)
	end = default_timer()
	print(f"RUNTIME for optimized: {end-start}")

def run_classification_cv(model, X, y, params = None):

	skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=None)

	start = default_timer()
	clf_model = model()
	scores = cross_val_score(clf_model, X, y, cv=skf, scoring="recall")
	end = default_timer()

	print(f"\n{model.__name__} RUNTIME: {end-start}")
	print(f"{model.__name__} scores with default params: {scores}")
	print(f"{model.__name__} mean of scores with default params: {scores.mean()}")

	print("\nFinding the best params")
	grid_clf = GridSearchCV(clf_model, param_grid=params)
	grid_clf.fit(X, y)
	print(f'Best parameters: {grid_clf.best_params_}')
	print(f"Optimized {model.__name__} mean of scores: {grid_clf.best_score_}")

	start = default_timer()
	clf_model = model(**grid_clf.best_params_)
	scores = cross_val_score(clf_model, X, y, cv=skf)
	end = default_timer()
	print(f"RUNTIME for optimized: {end-start}")


if __name__ == '__main__':

	# REGRESSION
	data = load_diabetes()
	X = data.data
	y = data.target

	# own_linear_regressions__params = {"learning_rate": [0.1, 0.5, 0.8], "n_iters": [10, 100, 1000]}
	# run_linear_cv(OwnLinearRegression, X, y, own_linear_regressions__params)

	# linear_regressions__params = {"copy_X": [False, True], "fit_intercept": [False, True], "n_jobs": [1, 5, 10], "positive": [False, True]}
	# run_linear_cv(LinearRegression, X, y, linear_regressions__params)

	# ridge_params = {}
	# run_linear_cv(Ridge, X, y, ridge_params)

	# CLASSIFICATION	
	df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
	test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)
	X_train = df[df.columns[:-1]]
	X_test = test_df[test_df.columns[:-1]]
	x = pd.concat([X_train, X_test])
	y_train = df["class"]
	y_test = test_df["class"]
	y = pd.concat([y_train, y_test])

	df = pd.read_csv(os.getcwd() + "/data/diabetes.csv")

	X = df[df.columns[:-1]]
	y = df["Class"]

	# run_classification_cv(OwnNaiveBayes, X, y)
	# start = default_timer()
	# nb = OwnNaiveBayes()
	# nb.fit()
	# nb.predict()
	# end = default_timer()
	# print(f"\n{OwnNaiveBayes.__name__} RUNTIME: {end-start}")

	gnb_params = {"var_smoothing": [0.000001, 0.00001, 0.0001, 0.001, 0.01]}
	rfc_params = {"n_estimators": [10, 100, 100], "criterion": ["gini", "entropy", "log_loss"]}
	dtc_params = {"criterion": ["gini", "entropy", "log_loss"], "splitter": ["best", "random"]}

	run_classification_cv(GaussianNB, X, y, gnb_params)
	run_classification_cv(RandomForestClassifier, X, y, rfc_params)
	run_classification_cv(DecisionTreeClassifier, X, y, dtc_params)