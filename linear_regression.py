import csv
import numpy as np
from sklearn.linear_model import LinearRegression
import math
import utils

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
	X_train, y_train = utils.get_data(utils.TRAIN_PATH)
	X_validate, y_validate = utils.get_data(utils.VALIDATE_PATH)
	X_test, y_test = utils.get_data(utils.TEST_PATH)

	reg = train(X_train, y_train)
	print("Coefficients:", reg.coef_)


	print("Train Accuracy: ", evaluate(X_train, y_train, reg))
	print("Validation Accuracy: ", evaluate(X_validate, y_validate, reg))
	print("Test Accuracy: ", evaluate(X_test, y_test, reg))

if __name__ == '__main__':
	main()
