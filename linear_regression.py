import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import utils

RACE_BLIND = True

def train(X, y):
	return LinearRegression().fit(X, y)

def predict(reg, x):
	return reg.predict(x.reshape(1, -1))

def evaluate(X, y, reg):
	correct = 0
	false = 0
	for i in range(len(X)):
		prediction = predict(reg, X[i])[0]
		if abs(prediction - y[i]) < abs(prediction - (1 - y[i])):
			correct += 1
		else:
			false += 1
	return float(correct)/(correct + false)

def main():
	if RACE_BLIND:
		X_train, y_train = utils.get_data(utils.TRAIN_PATH_RACE_BLIND)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH_RACE_BLIND)
		X_test, y_test = utils.get_data(utils.TEST_PATH_RACE_BLIND)
		feature_names = utils.get_feature_names_race_blind()
	else:
		X_train, y_train = utils.get_data(utils.TRAIN_PATH)
		X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH)
		X_test, y_test = utils.get_data(utils.TEST_PATH)
		feature_names = utils.get_feature_names()

	reg = train(X_train, y_train)
	print(feature_names)
	coeff = reg.coef_
	named_coeffs = []
	for i in range(len(coeff)):
		named_coeffs.append((feature_names[i], coeff[i]))

	named_coeffs = sorted(named_coeffs, key=lambda pair: abs(pair[1]), reverse=True)
	print("Coefficients:", named_coeffs)

	print("Train Accuracy: ", evaluate(X_train, y_train, reg))
	print("Validation Accuracy: ", evaluate(X_validate, y_validate, reg))
	# print("Test Accuracy: ", evaluate(X_test, y_test, reg))

if __name__ == '__main__':
	main()
