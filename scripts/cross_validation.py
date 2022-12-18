import os

import pandas as pd
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_diabetes

from timeit import default_timer
from linear_regression import OwnLinearRegression

from sklearn.datasets import load_diabetes

data = load_diabetes()
X = data.data
y = data.target


def run_skf(x, y):
	start = default_timer()
	skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=None)


	for train_index, test_index in skf.split(x, y):

		# x_fold_train, y_fold_train = x.iloc[train_index], y.iloc[train_index]
		# x_fold_test,  y_fold_test  = x.iloc[test_index],  y.iloc[test_index]
		x_fold_train, y_fold_train = x[train_index], y[train_index]
		x_fold_test,  y_fold_test  = x[test_index],  y[test_index]
		
	# 	naive_bayes = GaussianNB()
		linear_regression = LinearRegression()
		own_linear_regression = OwnLinearRegression()

	# 	naive_bayes.fit(x_fold_train, y_fold_train)
		linear_regression.fit(x_fold_train , y_fold_train)
		own_linear_regression.fit(x_fold_train, y_fold_train)

	# 	# print(naive_bayes.score(x_fold_test, y_fold_test))
		print(own_linear_regression.predict(x_fold_test))
		print(own_linear_regression.score(x_fold_test, y_fold_test))
		print(linear_regression.score(x_fold_test, y_fold_test))


	end = default_timer()
	print("RUNTIME: ", end-start)

	data = load_diabetes()
	X = data.data
	y = data.target
	own_linear_regression = OwnLinearRegression()
	own_linear_regression.fit()
	print(cross_val_score(own_linear_regression, X, y, cv=skf))

if __name__ == '__main__':

	df = pd.read_csv(os.getcwd() + "/data/congress/clean_train_congress.csv", index_col=0)
	test_df = pd.read_csv(os.getcwd() + "/data/congress/clean_test_congress.csv", index_col=0)

	# Create the training data
	# X_train = df[df.columns[:-1]]
	# X_test = test_df[test_df.columns[:-1]]
	# x = pd.concat([X_train, X_test])
	# y_train = df["class"]
	# y_test = test_df["class"]
	# y = pd.concat([y_train, y_test])

	run_skf(X,y)

